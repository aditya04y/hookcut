"""Editor workbench for audience-informed promo decisions on a saved script run."""

from __future__ import annotations

from hashlib import sha256

import pandas as pd
import streamlit as st

from app.views import segment_range
from hookcut.audience import clip_keys, events_csv
from hookcut.schema import RunResult
from hookcut.studio import build_studio
from hookcut.textfmt import ep_tc

SIMULATE = "Simulated activity"
UPLOAD = "Upload activity CSV"
EDITORIAL = "Script only"
OBJECTIVES = {"starts": "Start the show", "hold": "Hold attention for 3 seconds"}
GOAL_LABELS = {"starts": "show starts", "hold": "3-second holds"}
PRIORS = {"neighbors": "Similar cuts", "genome": "Story-feature model"}


def rate(count: int, denominator: int) -> str:
    return f"{count / denominator:.0%}" if denominator else "—"


def show_upload(result: RunResult, run_key: str) -> str | None:
    with st.expander("Activity CSV format and clip keys"):
        st.write("Use one row per impression with an anonymous viewer ID, cohort, clip key, and seconds watched "
                 "from the start of the promo. Set continued to 1 when the viewer starts the show after "
                 "finishing the promo, or 0 otherwise. Reuse an impression ID only for a retry of the same event.")
        st.code(events_csv([]).strip(), language=None)
        st.caption("Use reported in the source column for playback observations and synthetic for generated events.")
        keys = clip_keys(result)
        catalog = pd.DataFrame([
            {"candidate_id": candidate.id, "clip_key": keys[candidate.id], "duration_s": candidate.duration,
             "cut": " + ".join(map(segment_range, candidate.segments))}
            for candidate in result.candidates
        ])
        left, right = st.columns(2)
        left.download_button("Download blank activity CSV", events_csv([]), "activity-template.csv", "text/csv")
        right.download_button("Download clip keys", catalog.to_csv(index=False), "clip-keys.csv", "text/csv")
    file = st.file_uploader("Activity file", type=["csv"], key=f"activity-file-{run_key}")
    if file is None:
        st.info("Choose an activity CSV for this show to calculate its audience ranking.")
        return None
    try:
        return file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("The activity file must be a UTF-8 encoded CSV.")
        return None


def report_for(result: RunResult, raw_csv: str | None, simulate: bool, objective: str, cohort: str | None,
               prior: str = "neighbors"):
    # Keep uploaded activity in this editor's Streamlit session. Widget reruns can reuse the
    # all-audience overview and one cohort view without caching anyone's data across sessions.
    context = (sha256(raw_csv.encode()).digest() if raw_csv is not None else None, simulate, objective, prior)
    cache = st.session_state.get("audience_report_cache")
    if not cache or cache["run"] is not result or cache["context"] != context:
        cache = {"run": result, "context": context, "reports": {}}
        st.session_state.audience_report_cache = cache
    reports = cache["reports"]
    if cohort not in reports:
        report = build_studio(result, events_csv=raw_csv, cohort=cohort, objective=objective, simulate=simulate,
                              prior=prior)
        if cohort is not None:
            for previous in list(reports):
                if previous is not None:
                    del reports[previous]
        reports[cohort] = report
    return reports[cohort]


def feature_label(genome: dict, feature_id: str) -> str:
    return genome["nodes"].get(feature_id, {}).get("label", feature_id)


def show_activity_matrix(report: dict) -> None:
    matrix = report["matrix"]
    if not matrix:
        return
    with st.expander("Viewer × cut activity matrix"):
        metric = st.selectbox("Cell outcome", ["held_3s", "completed", "continued"], format_func={
            "held_3s": "3-second hold", "completed": "Finished the promo", "continued": "Started the show",
        }.get)
        clip_ids = [row["candidate_id"] for row in report["rows"]][:8]
        preview = {}
        for cell in matrix:
            viewer, clip = cell["viewer_id"], cell["candidate_id"]
            if clip not in clip_ids or (viewer not in preview and len(preview) >= 12):
                continue
            entry = preview.setdefault(viewer, {"Viewer": f"Viewer {len(preview) + 1:02d}"})
            entry[clip] = f"{cell[metric]} / {cell['impressions']}"
        st.caption("Each cell shows outcomes / impressions. A dash means the viewer had no impressions of that cut. "
                   "The preview uses anonymous labels for up to 12 viewers and the first 8 ranked cuts.")
        if preview:
            table = pd.DataFrame(preview.values()).reindex(columns=["Viewer", *clip_ids])
            st.dataframe(table.astype("string").fillna("—"), hide_index=True)
        else:
            st.caption("This audience has no impressions of the displayed cuts.")


def show_patterns(report: dict) -> None:
    rows = [{
        "Story feature": feature_label(report["genome"], pattern["feature"]),
        "Cuts": pattern["clips"], "Impressions": pattern["impressions"],
        "Held 3s / impressions": rate(pattern["held_3s"], pattern["impressions"]),
        "Starts / finishes": rate(pattern["continued"], pattern["completed"]),
        "Starts / impressions": rate(pattern["continued"], pattern["impressions"]),
    } for pattern in report["patterns"] if pattern["impressions"]]
    if rows:
        with st.expander("Which story features held this audience?"):
            st.caption("Each row pools activity for cuts sharing a feature. A cut may appear in several rows. "
                       "Differences can reflect the audience or other elements in the same cuts.")
            st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_graph(result: RunResult, report: dict, row: dict) -> None:
    graph = report["genome"]
    clip_edges = [edge for edge in graph["edges"] if edge["source"] == f"clip:{row['candidate_id']}"]
    edges = [edge for edge in clip_edges if edge["relation"] == "has_feature"]
    beats = {b.id: b for ep in result.episodes for b in ep.beats}
    inside = {edge["target"].removeprefix("beat:") for edge in clip_edges if edge["relation"] == "contains"}
    records = []
    for edge in edges:
        evidence = []
        for beat_id in edge["beat_ids"][:2]:
            beat = beats.get(beat_id)
            if beat:
                evidence.append(f"{ep_tc(beat.episode, beat.start)} · {beat.text}")
        records.append({
            "Type": edge["target"].partition(":")[0].title(), "Story feature": feature_label(graph, edge["target"]),
            "Basis": edge.get("basis", ""), "Script evidence": "\n".join(evidence),
        })
    with st.expander("Content genome · features and source beats"):
        if records:
            st.dataframe(pd.DataFrame(records), hide_index=True)
            relationships = []
            for e in graph["edges"]:
                if (e["source"].startswith("character:") and e["target"].startswith("character:")
                        and inside.intersection(e["beat_ids"])):
                    relationships.append({
                        "Character": graph["nodes"][e["source"]]["label"],
                        "Relationship": e["relation"].replace("_", " "),
                        "Other character": graph["nodes"][e["target"]]["label"],
                        "Basis": e["basis"], "Source beat": ", ".join(sorted(inside.intersection(e["beat_ids"]))),
                    })
            if relationships:
                st.markdown("**Character relationships in this cut**")
                st.dataframe(pd.DataFrame(relationships), hide_index=True)
            for note in graph.get("limitations", []):
                st.caption(note)
        else:
            st.caption("This cut has no linked story evidence in the saved run.")


def inspect_cut(result: RunResult, report: dict, row: dict, objective: str) -> None:
    promo = next(p for p in result.promos if p.candidate_id == row["candidate_id"])
    st.caption(" + ".join(map(segment_range, promo.segments)))
    left, right = st.columns(2)
    with left:
        st.markdown("**Opening hook**")
        st.text(promo.hook_text)
    with right:
        st.markdown("**Cliffhanger**")
        st.text(promo.cliffhanger_text)
        if promo.open_question:
            st.caption(promo.open_question)
    impressions, held, finished, starts = (row[key] for key in ("impressions", "held_3s", "completed", "continued"))
    if report["source"] != "none":
        st.markdown("**Playback funnel for this cut**")
        stages = [
            ("Held for 3 seconds", held, impressions, "impressions"), ("Reached the ending", finished, impressions, "impressions"),
            ("Started the show after the ending", starts, finished, "finishes"),
        ]
        for column, (label, count, denominator, unit) in zip(st.columns(3), stages):
            column.metric(label, rate(count, denominator))
            column.caption(f"{count:,} / {denominator:,} {unit}")
        st.caption("A dash means the selected audience has no observations for that rate.")
    st.write(row["reason"])
    if row["neighbors"]:
        with st.expander("Evidence from similar cuts"):
            st.caption(f"These cuts share story features used to estimate {GOAL_LABELS[objective]} per impression.")
            st.dataframe(pd.DataFrame([
                {"Cut": n["candidate_id"], "Impressions": n["impressions"], "Outcome rate": f"{n['rate']:.1%}",
                 "Shared features": ", ".join(feature_label(report["genome"], f) for f in n["shared"])}
                for n in row["neighbors"]
            ]), hide_index=True)
    if len(report["cohorts"]) > 1:
        with st.expander("Compare this cut across cohorts"):
            by_cohort = []
            for cohort, stats in report["cohort_stats"].items():
                cell = stats.get(row["candidate_id"], {})
                n, held, completed, started = (cell.get(key, 0) for key in ("impressions", "held_3s", "completed", "continued"))
                by_cohort.append({"Cohort": cohort, "Impressions": n, "Held 3s": held, "Finished": completed,
                                  "Started show": started, "3s hold rate": rate(held, n),
                                  "Starts / finishes": rate(started, completed), "Starts / impressions": rate(started, n)})
            st.dataframe(pd.DataFrame(by_cohort), hide_index=True)
    show_graph(result, report, row)


def show_validation(report: dict) -> None:
    evaluation = report["evaluation"]
    with st.expander("How well did the forecast generalize?"):
        if evaluation["model_brier"] is None or evaluation["baseline_brier"] is None:
            st.write("There is too little eligible activity for the held-out episode check.")
        else:
            st.write("The model forecasts impressions from held-out episodes using the other episodes. "
                     "Lower Brier scores indicate more accurate probability estimates.")
            model, baseline = st.columns(2)
            model.metric("Audience model", f"{evaluation['model_brier']:.3f}")
            baseline.metric("Cohort average", f"{evaluation['baseline_brier']:.3f}")
            st.caption(f"{evaluation['evaluated_impressions']:,} held-out impressions across {evaluation['folds']} folds.")
    with st.expander("Workflow and model routing"):
        st.dataframe(pd.DataFrame(report["workflow"]).rename(columns={
            "stage": "Stage", "executor": "Runs on", "detail": "Work performed",
        }), hide_index=True)


def show_audience_studio(result: RunResult) -> None:
    if not result.promos:
        st.info("Rank promo cuts first to open the audience studio.")
        return
    run_key = sha256(f"{result.show}/{result.created_at}".encode()).hexdigest()[:16]
    is_demo = result.config.get("fixture") == "hand-authored"
    source_column, goal_column = st.columns([2, 1])
    source = source_column.radio("Activity source", [SIMULATE, UPLOAD, EDITORIAL], horizontal=True,
                                 index=0 if is_demo else 2, key=f"activity-source-{run_key}")
    objective = goal_column.selectbox("Ranking goal", OBJECTIVES, format_func=OBJECTIVES.get,
                                      disabled=source == EDITORIAL)
    prior = st.radio("Estimate for cuts with little data from", PRIORS, format_func=PRIORS.get, horizontal=True,
                     disabled=source == EDITORIAL, key=f"activity-prior-{run_key}")
    raw_csv = show_upload(result, run_key) if source == UPLOAD else None
    if source == UPLOAD and raw_csv is None:
        return
    try:
        overview = report_for(result, raw_csv, source == SIMULATE, objective, None, prior)
        cohort = st.selectbox("Audience cohort", [None, *overview["cohorts"]],
                              format_func=lambda value: "All viewers" if value is None else value,
                              disabled=not overview["cohorts"])
        report = overview if cohort is None else report_for(result, raw_csv, source == SIMULATE, objective, cohort,
                                                            prior)
    except ValueError as exc:
        st.error(f"Unable to use this activity data. {exc}")
        return
    if report["source"] == "simulated":
        st.info("Simulated audience activity. Rates and forecasts illustrate how the editor workflow responds to viewing behavior.")
    elif report["source"] == "imported":
        st.caption("Rates use playback activity reported in the uploaded CSV.")
    else:
        st.caption("Showing the script-based posting order. Choose an activity source to estimate audience response.")

    rows = report["rows"]
    if not rows:
        st.info("There are no eligible cuts for this ranking.")
        return
    promos = {p.candidate_id: p for p in result.promos}
    labels = {r["candidate_id"]: " + ".join(map(segment_range, promos[r["candidate_id"]].segments)) for r in rows}
    st.subheader("Posting order")
    st.caption("Wave 1 introduces the premise, wave 2 escalates the conflict, and wave 3 contains larger reveals.")
    if report["source"] != "none":
        st.caption(f"The forecast estimates {GOAL_LABELS[objective]} per impression and orders cuts within each wave.")
    ranking = []
    for row in rows:
        cid = row["candidate_id"]
        record = {"Rank": row["rank"], "Wave": row["wave"], "Cut": cid, "Timecode": labels[cid],
                  "Opening": promos[cid].hook_text, "Editorial rank": row["editorial_rank"]}
        if report["source"] != "none":
            record.update({"Impressions": row["impressions"],
                           "Forecast": f"{row['forecast']:.1%}" if row["forecast"] is not None else "—"})
        ranking.append(record)
    frame = pd.DataFrame(ranking)
    st.dataframe(frame, hide_index=True)
    export = frame.assign(clip_key=[r["clip_key"] for r in rows], source=report["source"], objective=objective,
                          cohort=cohort or "all", show=result.show)
    st.download_button("Download this posting order", export.to_csv(index=False), "hookcut-posting-order.csv", "text/csv")
    by_id = {row["candidate_id"]: row for row in rows}
    chosen = st.selectbox("Inspect a cut", list(by_id),
                          format_func=lambda cid: f"#{by_id[cid]['rank']} · {cid} · {labels[cid]}")
    inspect_cut(result, report, by_id[chosen], objective)
    show_patterns(report)
    show_activity_matrix(report)
    if experiment := report.get("experiment"):
        with st.expander("Test two openings with the same ending"):
            st.write(f"Compare {experiment['candidate_id']} with {experiment['challenger_id']}. "
                     f"Both cuts end at {experiment['cliff_beat_id']}.")
            st.text(experiment["goal"])
            st.text(experiment["allocation"])
            candidates = {c.id: c for c in result.candidates}
            for arm, key in (("Original", "candidate_id"), ("Alternate", "challenger_id")):
                candidate = candidates[experiment[key]]
                st.text(f"{arm} · {candidate.id} · {' + '.join(map(segment_range, candidate.segments))}")
    show_validation(report)
