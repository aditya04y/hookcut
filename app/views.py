"""Streamlit views for a validated RunResult."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from hookcut.runlog import RunLog
from hookcut.schema import RunResult, Segment
from hookcut.textfmt import ep_tc, tc

WAVES = {1: "Cold audiences", 2: "Escalation", 3: "Big reveals"}
WAVE_COLORS = {1: "blue", 2: "orange", 3: "violet"}
PICKUPS = {
    "immediate": "Immediate",
    "recap_then_advance": "After a recap",
    "delayed": "Delayed",
    "unrelated": "Unrelated",
    "not_applicable": "First episode",
}


def segment_range(segment: Segment) -> str:
    return f"{ep_tc(segment.episode, segment.start)}–{tc(segment.end)}"


def show_promos(result: RunResult) -> None:
    if not result.promos:
        st.info("This run has no ranked promos.")
        return
    wave = st.selectbox("Post wave", [None, *WAVES], format_func=lambda w: "All waves" if w is None else f"{w} · {WAVES[w]}")
    promos = [promo for promo in sorted(result.promos, key=lambda promo: promo.rank)
              if wave is None or promo.post_wave == wave]
    if not promos:
        st.info("This wave has no ranked promos.")
    verdicts = {verdict.candidate_id: verdict for verdict in result.verdicts}
    for promo in promos:
        with st.container(border=True):
            st.markdown(f"### #{promo.rank:02d} · {promo.duration:g} s")
            color = WAVE_COLORS[promo.post_wave]
            st.markdown(f":{color}-background[Wave {promo.post_wave} · {WAVES[promo.post_wave]}]")
            for segment in promo.segments:
                st.code(segment_range(segment), language=None)
            opening, ending = st.columns(2)
            with opening:
                st.markdown("**Opening hook**")
                st.caption(promo.hook_time)
                st.text(promo.hook_text)
            with ending:
                st.markdown("**Cliffhanger**")
                st.caption(promo.cliffhanger_time)
                st.text(promo.cliffhanger_text)
            st.markdown("**Open question**")
            st.text(promo.open_question or "—")
            st.markdown("**Why it works**")
            st.text(promo.why_it_works or "—")
            verdict = promo.cold_viewer or verdicts.get(promo.candidate_id)
            if verdict:
                st.markdown(f"**Cold-viewer test · {verdict.verdict.title()}**")
                st.text(verdict.premise)
                st.caption("Viewer confusions")
                st.text(" · ".join(verdict.confusions) or "None reported")
            else:
                st.caption("Cold-viewer test unavailable")
            if promo.flags:
                st.caption("Flags")
                st.text(" · ".join(promo.flags))
            with st.expander(f"Variants ({len(promo.variants)})"):
                if not promo.variants:
                    st.caption("No variants recorded.")
                for variant in promo.variants:
                    st.text(variant.get("label") or "Alternate cut")
                    if variant.get("duration") is not None:
                        st.caption(f"{variant['duration']} s")
                    for segment in variant.get("segments") or []:
                        if isinstance(segment, dict) and {"episode", "start", "end"} <= segment.keys():
                            st.text(f"{ep_tc(segment['episode'], segment['start'])}–{tc(segment['end'])}")
                    if variant.get("note"):
                        st.text(variant["note"])


def show_season_summary(result: RunResult) -> None:
    """Season view for the script editor: averages, weakest openings/endings, candidate unlock episodes."""
    try:
        from hookcut.report import season_summary
        summary = season_summary(result.episode_report)
    except Exception:
        return
    if not summary.get("analyzed"):
        return
    st.subheader("Season summary")
    a, b, c = st.columns(3)
    a.metric("Avg opening hook", summary.get("avg_opening_hook"))
    b.metric("Avg ending cliffhanger", summary.get("avg_ending_cliff"))
    c.metric("Episodes analyzed", f"{summary['analyzed']} / {summary['episodes']}")
    fmt = lambda items: ", ".join(f"E{i['episode']:02d} ({i['score']})" for i in items) or "—"
    st.markdown(f"**Weakest openings:** {fmt(summary.get('weakest_openings', []))}  \n"
                f"**Weakest endings:** {fmt(summary.get('weakest_endings', []))}")
    unlock = summary.get("unlock_candidates") or []
    if unlock:
        st.markdown("**Candidate unlock episodes** (strongest cliffhangers where free episodes usually run out):")
        for u in unlock:
            st.markdown(f"- E{u['episode']:02d} · cliffhanger {u['score']}/5 · {u.get('open_question') or ''}")
        st.caption(summary.get("unlock_assumption", ""))


def show_episode_report(result: RunResult) -> None:
    saved = {row["episode"]: row for row in result.episode_report if isinstance(row.get("episode"), int)}
    analyses = {analysis.episode: analysis for analysis in result.analyses}
    rows = []
    for episode in sorted(result.episodes, key=lambda episode: episode.number):
        row = saved.get(episode.number, {})
        analysis = analyses.get(episode.number)
        pickup = row.get("pickup_status", analysis.pickup.status if analysis else None)
        rows.append({
            "Episode": f"E{episode.number:02d}",
            "Opening hook": row.get("opening_hook_score", analysis.opening_hook.score if analysis else None),
            "Ending cliffhanger": row.get("ending_cliff_score", analysis.ending_cliffhanger.score if analysis else None),
            "Pickup": PICKUPS.get(pickup, pickup) or "—",
            "Delay (s)": row.get("pickup_delay_s", analysis.pickup.delay_s if analysis else None),
            "Opening note": row.get("opening_hook_note", analysis.opening_hook.note if analysis else ""),
            "Ending note": analysis.ending_cliffhanger.note if analysis else "",
            "Pickup note": analysis.pickup.note if analysis else "",
            "Ending question": row.get("ending_open_question", analysis.ending_open_question if analysis else ""),
            "Flags": " · ".join(row.get("flags") or []),
        })
    if not rows:
        st.info("This run has no episodes to report.")
        return
    frame = pd.DataFrame(rows)
    show_season_summary(result)
    st.caption("Pickup describes how this episode continues the previous ending.")
    st.dataframe(frame, hide_index=True)
    scores = frame.melt(id_vars="Episode", value_vars=["Opening hook", "Ending cliffhanger"],
                        var_name="Measure", value_name="Score").dropna(subset=["Score"])
    if scores.empty:
        st.info("No opening or ending scores are available.")
        return
    chart = alt.Chart(scores).mark_bar().encode(
        x=alt.X("Episode:N", sort=frame["Episode"].tolist()),
        xOffset="Measure:N",
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5]), axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("Measure:N", title=None),
        tooltip=["Episode:N", "Measure:N", "Score:Q"],
    )
    st.altair_chart(chart.properties(height=260))


def show_timeline(result: RunResult) -> None:
    if not result.episodes:
        st.info("This run has no episode timelines.")
        return
    episode_index = st.selectbox("Timeline episode", range(len(result.episodes)),
                                 format_func=lambda index: f"E{result.episodes[index].number:02d}")
    episode = result.episodes[episode_index]
    analysis = next((analysis for analysis in result.analyses if analysis.episode == episode.number), None)
    windows = [
        {"Start": segment.start, "End": segment.end, "Promo": f"#{promo.rank} · {segment_range(segment)}"}
        for promo in result.promos for segment in promo.segments if segment.episode == episode.number
    ]
    layers = []
    x = alt.X("Start:Q", title="Episode time (seconds)",
              scale=alt.Scale(domain=[0, episode.duration]) if episode.duration > 0 else alt.Scale())
    if windows:
        layers.append(alt.Chart(pd.DataFrame(windows)).mark_rect(color="#f5a524", opacity=0.19).encode(
            x=x, x2="End:Q", tooltip=["Promo:N", "Start:Q", "End:Q"],
        ))
        st.caption("Shading marks the windows used by ranked promos.")
    if analysis and episode.beats:
        scores = {score.beat_id: score for score in analysis.beats}
        rows = []
        for beat in episode.beats:
            score = scores.get(beat.id)
            rows.extend([
                {"Start": beat.start, "Beat": beat.id, "Text": beat.text, "Measure": "Hook", "Score": score.hook if score else 0},
                {"Start": beat.start, "Beat": beat.id, "Text": beat.text, "Measure": "Cliffhanger", "Score": score.cliff if score else 0},
            ])
        layers.append(alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
            x=x,
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5]), axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("Measure:N", title=None),
            order=alt.Order("Beat:N"),
            tooltip=["Start:Q", "Beat:N", "Text:N", "Measure:N", "Score:Q"],
        ))
    else:
        st.info("No beat scores are available for this episode.")
    if layers:
        st.altair_chart(alt.layer(*layers).properties(height=290))


def show_candidates(result: RunResult) -> None:
    if not result.candidates:
        st.info("This run has no candidate clips.")
        return
    ranks = {promo.candidate_id: promo.rank for promo in result.promos}
    verdicts = {verdict.candidate_id: verdict for verdict in result.verdicts}
    verdicts.update({promo.candidate_id: promo.cold_viewer for promo in result.promos if promo.cold_viewer})
    rows = []
    for candidate in result.candidates:
        verdict = verdicts.get(candidate.id)
        rows.append({
            "Candidate": candidate.id,
            "Shortlist": f"#{ranks[candidate.id]}" if candidate.id in ranks else "Not shortlisted",
            "Cold-viewer verdict": verdict.verdict.title() if verdict else "Unavailable",
            "Confusions": " · ".join(verdict.confusions) if verdict else "",
            "Segments": " + ".join(map(segment_range, candidate.segments)),
            "Duration (s)": candidate.duration,
            "Hook": candidate.hook_score,
            "Cliffhanger": candidate.cliff_score,
            "Standalone": verdict.standalone if verdict else None,
            "Hook lands": verdict.hook_lands if verdict else None,
            "Viewer premise": verdict.premise if verdict else "",
            "Viewer conflict": verdict.conflict if verdict else "",
            "Viewer question": verdict.open_question if verdict else "",
            "Context needed": " · ".join(candidate.needs_context),
            "Notes": " · ".join(candidate.notes),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_run_log(result: RunResult, prices: dict, redact) -> None:
    if not result.run_log:
        st.info("This run has no call log.")
        return
    log = RunLog(prices=result.config.get("prices_usd_per_mtok") or prices)
    usage_fields = ("input_tokens", "output_tokens", "latency_s")
    log.rows = [
        {**row, **{key: row[key] if isinstance(row.get(key), (int, float)) else 0 for key in usage_fields}}
        for row in result.run_log if row.get("stage")
    ]
    by_stage = {}
    for stage_model, totals in log.summary()["by_stage_model"].items():
        stage, _, _ = stage_model.partition("|")
        current = by_stage.setdefault(stage, dict.fromkeys(totals, 0))
        for key, value in totals.items():
            current[key] += value
    if by_stage:
        st.caption("Tokens, request latency and estimated list-price cost include uncached calls.")
        st.dataframe(pd.DataFrame([
            {"Stage": stage, "Calls": row["calls"], "Cached": row["cached"], "Errors": row["errors"],
             "Input tokens": row["input_tokens"], "Output tokens": row["output_tokens"],
             "Latency (s)": round(row["latency_s"], 2), "Est. cost (USD)": round(row["est_cost_usd"], 4)}
            for stage, row in by_stage.items()
        ]), hide_index=True)
    with st.expander("Call details"):
        columns = ("stage", "item", "provider", "model", "input_tokens", "output_tokens", "latency_s",
                   "cached", "attempts", "message", "error")
        details = [{key: redact(value) if isinstance(value, str) else value for key, value in row.items() if key in columns}
                   for row in result.run_log]
        st.dataframe(pd.DataFrame(details).reindex(columns=columns).dropna(axis=1, how="all"), hide_index=True)


def show_downloads(result: RunResult, folder: Path | None, source: str) -> None:
    files = {}
    if folder:
        root = folder.resolve()
        files = {path.relative_to(folder).as_posix(): path for path in sorted(folder.rglob("*"))
                 if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)}
    result_json = result.model_dump_json(indent=2).encode("utf-8")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, path in files.items():
            bundle.write(path, name)
        if "result.json" not in files:
            bundle.writestr("result.json", result_json)
    st.download_button("Download run ZIP", archive.getvalue(), file_name=f"{source}.zip", mime="application/zip")
    singles = {"result.json": "application/json", "promos.csv": "text/csv", "promos.json": "application/json",
               "promos.md": "text/markdown", "episode_report.csv": "text/csv"}
    for name, mime in singles.items():
        if name in files or name == "result.json":
            data = files[name].read_bytes() if name in files else result_json
            st.download_button(name, data, file_name=name, mime=mime, key=f"download-{name}")
    for name, path in files.items():
        if name.startswith("editor/"):
            st.download_button(name, path.read_bytes(), file_name=path.name, key=f"download-{name}")
