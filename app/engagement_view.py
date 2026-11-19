"""Engagement view: viewer x cut and viewer x episode activity, the story-feature model and its decisions."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from hookcut.engagement import build_engagement
from hookcut.schema import RunResult

# Reference palette: categorical slots 1-4 in fixed order for cohorts, secondary ink for all viewers,
# an ordinal blue ramp (steps 250 -> 650) for activity states, blue/red poles for the sign of an effect.
COHORT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ALL_COLOR = "#52514e"
NEVER_COLOR = "#e1e0d9"
STATE_COLORS = ["#86b6ef", "#3987e5", "#256abf", "#104281"]
UP, DOWN, MUTED = "#2a78d6", "#e34948", "#898781"
OUTCOME_NAMES = {"hold": "3-second hold (promo)", "start": "Story start (promo)",
                 "complete": "Episode completed", "continue": "Next episode after the ending"}


def _report(result: RunResult, paywall: int) -> dict:
    cache = st.session_state.get("engagement_cache")
    if not cache or cache["run"] is not result:
        cache = st.session_state["engagement_cache"] = {"run": result, "reports": {}}
    if paywall not in cache["reports"]:
        cache["reports"][paywall] = build_engagement(result, simulate_activity=True, paywall=paywall, bootstrap=100)
    return cache["reports"][paywall]


def _title(report: dict, text: str) -> str:
    return f"{text} ({report['label']})"


def _heatmap(report: dict, grain: str, column: str, labels: list[str]) -> None:
    data = report.get("heatmaps", {}).get(grain)
    if not data or not data["cells"]:
        return
    states = ["never shown" if grain == "promo" else "never offered", *data["states"]]
    rows = [{"Viewer": viewer, column: label, "State": states[code + 1]}
            for viewer, cells in zip(data["viewers"], data["cells"]) for label, code in zip(labels, cells)]
    chart = alt.Chart(pd.DataFrame(rows)).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X(f"{column}:N", sort=labels, title=None, axis=alt.Axis(labelAngle=-60 if grain == "promo" else 0)),
        y=alt.Y("Viewer:N", sort=data["viewers"], title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("State:N", sort=states, title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=states, range=[NEVER_COLOR, *STATE_COLORS])),
        tooltip=["Viewer", column, "State"],
    ).properties(height=max(160, 7 * len(data["viewers"])))
    st.altair_chart(chart)


def show_matrices(result: RunResult, report: dict) -> None:
    sizes = report["matrices"]
    st.markdown("#### Activity matrices")
    for grain, unit in (("promo", "cut"), ("episode", "episode")):
        size = sizes.get(grain)
        if not isinstance(size, dict):
            continue
        cols = st.columns(4)
        cols[0].metric(f"Viewers x {unit}s", f"{size['viewers']:,} x {size.get('cuts', size.get('episodes'))}")
        cols[1].metric("Observed cells", f"{size['cells']:,}")
        cols[2].metric("Density", f"{size['density']:.1%}")
        if grain == "promo":
            cols[3].metric("Skips before 3 s", f"{size['observed_skips']:,}")
        else:
            cols[3].metric("Offered, not started", f"{size['observed_not_started']:,}")
    st.caption("A cell is one viewer and one cut (or episode). An observed skip is a cell; a cut never shown, or an "
               "episode never offered, is a missing cell. The model counts only observed cells.")
    heat = report.get("heatmaps", {})
    if heat.get("promo"):
        st.markdown(f"**{_title(report, 'Sampled viewers x top promos')}**")
        _heatmap(report, "promo", "Cut", heat["promo"]["cuts"])
    if heat.get("episode"):
        st.markdown(f"**{_title(report, 'Sampled viewers x episodes')}**")
        _heatmap(report, "episode", "Episode", [str(n) for n in heat["episode"]["episodes"]])


def show_retention(report: dict) -> None:
    episodes = report.get("episodes")
    if not episodes:
        return
    st.markdown(f"#### {_title(report, 'Retention by cohort')}")
    basis = st.radio("Curve", ["observed", "predicted"], horizontal=True, key="engagement-retention-basis",
                     format_func={"observed": "Observed", "predicted": "Model prediction"}.get)
    names = list(episodes["retention"])
    rows = [{"Cohort": name, "Episode": p["episode"], "Retained": p[basis], "Sessions": p["sessions"]}
            for name, curve in episodes["retention"].items() for p in curve if p[basis] is not None]
    if not rows:
        st.caption("No retention curve for this basis.")
        return
    cohorts = [n for n in names if n != "all"]
    chart = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30)).encode(
        x=alt.X("Episode:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Retained:Q", axis=alt.Axis(format="%"), title="Share of episode-1 starters still watching"),
        color=alt.Color("Cohort:N", sort=names, legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["all", *cohorts], range=[ALL_COLOR, *COHORT_COLORS])),
        tooltip=["Cohort", "Episode", alt.Tooltip("Retained:Q", format=".1%"), "Sessions"],
    ).properties(height=280)
    st.altair_chart(chart)
    st.caption("Each step multiplies the share who went from episode n to n+1, so viewers who join mid-season "
               "count from the episode where they joined.")


def show_effects(report: dict) -> str | None:
    models = report["models"]
    if not models:
        return None
    st.markdown(f"#### {_title(report, 'Why a story succeeds')}")
    outcome = st.selectbox("Outcome", list(models), format_func=lambda o: OUTCOME_NAMES.get(o, o),
                           key="engagement-outcome")
    model = models[outcome]
    scope = st.selectbox("Audience", ["pooled", *model["by_cohort"]], key="engagement-effect-scope",
                         format_func=lambda c: "All cohorts (cohort intercepts)" if c == "pooled" else c)
    effects = model["effects"] if scope == "pooled" else model["by_cohort"][scope]
    frame = pd.DataFrame(effects)
    if frame.empty:
        st.caption("No features vary in this data.")
        return outcome
    frame["Direction"] = ["raises" if v > 0 else "lowers" for v in frame["points"]]
    order = frame.sort_values("points")["label"].tolist()
    base = alt.Chart(frame).encode(y=alt.Y("label:N", sort=order, title=None))
    tips = [alt.Tooltip("label:N", title="Feature"), alt.Tooltip("change:N", title="Change"),
            alt.Tooltip("points:Q", title="Points", format="+.1f"),
            alt.Tooltip("points_lo:Q", title="90% CI low", format="+.1f"),
            alt.Tooltip("points_hi:Q", title="90% CI high", format="+.1f"),
            alt.Tooltip("log_odds:Q", title="Log-odds", format="+.2f"), alt.Tooltip("n:Q", title="Trials"),
            alt.Tooltip("units:Q", title="Cuts or episodes")]
    color = alt.Color("Direction:N", scale=alt.Scale(domain=["raises", "lowers"], range=[UP, DOWN]),
                      legend=alt.Legend(orient="top", title=None))
    chart = alt.layer(
        alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q"),
        base.mark_rule(strokeWidth=2).encode(x=alt.X("points_lo:Q", title="Change in probability (points)"),
                                             x2="points_hi:Q", color=color, tooltip=tips),
        base.mark_point(filled=True, size=80).encode(x="points:Q", color=color, tooltip=tips),
    ).properties(height=max(160, 22 * len(frame)))
    st.altair_chart(chart)
    st.caption(f"Base rate {model['base_rate']:.1%} ({model['outcome']}). Each dot is the change in probability at "
               "the base rate when the feature moves from no to yes, or by two standard deviations for a "
               "score; bars are 90% bootstrap intervals that resample whole cuts or episodes.")
    st.dataframe(_effect_table(frame), hide_index=True)
    screened = pd.DataFrame(model.get("screened", []))
    if scope == "pooled" and not screened.empty:
        with st.expander(f"Other story features, added one at a time ({len(screened)})"):
            st.caption("Story-graph numbers and genome tags are tested one at a time on top of the core model, "
                       "because a season cannot support every feature at once. About one in ten will show an "
                       "interval that excludes zero by chance; treat them as ideas for an A/B test.")
            st.dataframe(_effect_table(screened), hide_index=True)
    return outcome


def _effect_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {"label": "Feature", "change": "Change", "log_odds": "Log-odds", "lo": "CI low", "hi": "CI high",
               "points": "Points", "points_lo": "Points low", "points_hi": "Points high", "n": "Trials",
               "units": "Cuts or episodes", "support": "With the feature"}
    return frame[[c for c in columns if c in frame]].rename(columns=columns)


def show_recovery(report: dict, outcome: str | None) -> None:
    recovery = report.get("recovery")
    if not recovery or not outcome or not isinstance(recovery.get(outcome), dict):
        return
    r = recovery[outcome]
    st.markdown(f"#### {_title(report, 'Planted vs recovered effects')}")
    st.caption("This checks the pipeline on data whose answer is known. It is not evidence about any audience.")
    cols = st.columns(4)
    cols[0].metric("Sign agreement", f"{r['sign_agreement']:.0%}" if r["sign_agreement"] is not None else "-")
    cols[1].metric("Rank correlation (planted)", f"{r['spearman_planted']:.2f}" if r["spearman_planted"] is not None else "-")
    cols[2].metric("Planted inside 90% CI", f"{r['ci_coverage']:.0%}" if r["ci_coverage"] is not None else "-")
    cols[3].metric("Null features flagged", f"{r['null_false_alarms']} / {r['null_features']}")
    frame = pd.DataFrame(r["pairs"])
    frame["Fit"] = ["all cohorts" if c == "pooled" else "one cohort" for c in frame["cohort"]]
    lo, hi = float(min(frame[["planted", "lo"]].min())), float(max(frame[["planted", "hi"]].max()))
    tips = ["cohort", "feature", alt.Tooltip("planted:Q", format="+.2f"), alt.Tooltip("estimated:Q", format="+.2f"),
            alt.Tooltip("lo:Q", format="+.2f"), alt.Tooltip("hi:Q", format="+.2f")]
    color = alt.Color("Fit:N", scale=alt.Scale(domain=["all cohorts", "one cohort"], range=COHORT_COLORS[:2]),
                      legend=alt.Legend(orient="top", title=None))
    chart = alt.layer(
        alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(color=MUTED, strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q"),
        alt.Chart(frame).mark_rule(strokeWidth=1, opacity=0.5).encode(x="planted:Q", y="lo:Q", y2="hi:Q",
                                                                      color=color, tooltip=tips),
        alt.Chart(frame).mark_circle(size=64, stroke="white", strokeWidth=1).encode(
            x=alt.X("planted:Q", title="Planted effect (log-odds per change)"),
            y=alt.Y("estimated:Q", title="Recovered effect"), color=color, tooltip=tips),
    ).properties(height=300)
    st.altair_chart(chart)


def show_rankings(result: RunResult, report: dict) -> None:
    ranking = report.get("promo_ranking")
    if not ranking:
        return
    st.markdown(f"#### {_title(report, 'Which promo to show which cohort')}")
    left, right = st.columns(2)
    objective = left.radio("Goal", ["start", "hold"], horizontal=True, key="engagement-objective",
                           format_func={"start": "Story starts", "hold": "3-second holds"}.get)
    audiences = list(ranking[objective])
    cohort = right.selectbox("Cohort", audiences, key="engagement-cohort",
                             format_func=lambda c: "All viewers" if c == "all" else c)
    rows = [{"Rank": r["rank"], "Wave": r["wave"], "Cut": r["candidate_id"], "Change vs all": r["change_vs_all"],
             "Editorial rank": r["editorial_rank"], "Impressions": r["impressions"],
             "Observed": f"{r['hits'] / r['impressions']:.1%}" if r["impressions"] else "-",
             "Story-feature prior": f"{r['prior']:.1%}" if r["prior"] is not None else "-",
             "Score": f"{r['score']:.1%}" if r["score"] is not None else "-",
             "Cold start": "yes" if r["cold_start"] else ""} for r in ranking[objective][cohort]]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Score = (hits + m x prior) / (impressions + m), m up to {ranking['prior_strength']}. The prior is a "
               "logistic model of this cohort's outcome on genome features, so a cut with no impressions still gets "
               "a score. Cuts move only within their posting wave.")


def show_insights(result: RunResult, report: dict) -> None:
    episodes = report.get("episodes")
    if episodes:
        st.markdown(f"#### {_title(report, 'Where viewers drop')}")
        if episodes.get("dropoff"):
            st.dataframe(pd.DataFrame([{
                "Episode": d["episode"], "Predicted drop": f"{d['predicted_drop']:.0%}",
                "Observed drop": f"{d['observed_drop']:.0%}" if d["observed_drop"] is not None else "-",
                "Sessions": d["sessions"], "Where": d["where"], "Main reasons": ", ".join(d["drivers"]),
            } for d in episodes["dropoff"]]), hide_index=True)
        if episodes.get("endings"):
            with st.expander("Endings that keep viewers"):
                st.dataframe(pd.DataFrame([{
                    "Grouping": e["group"], "Value": e["value"], "Episodes": e["episodes"],
                    "Finished": e["completed"], "Went on": e["continued"],
                    "Rate": f"{e['rate']:.1%}" if e["rate"] is not None else "-"} for e in episodes["endings"]]),
                    hide_index=True)
                st.caption("Raw pooled rates; groups overlap with other features, so read the model effects for "
                           "one feature at a time.")
        paywall = episodes.get("paywall")
        if paywall and paywall["candidates"]:
            st.markdown(f"#### {_title(report, 'Which episode goes behind the paywall')}")
            st.dataframe(pd.DataFrame([{
                "Last free episode": c["last_free_episode"], "Paywall at": c["paywall_episode"],
                "Want the next episode": f"{c['continue_after_ending']:.1%}",
                "Reach the end of it": f"{c['reach_end']:.1%}", "Unlock demand": f"{c['unlock_demand']:.1%}",
            } for c in paywall["candidates"]]), hide_index=True)
            editor = ", ".join(map(str, paywall["season_summary_unlock"])) or "none"
            st.caption(f"Episode report (cliffhanger score only) suggests: {editor}. {paywall['note']}")
    ab = report.get("ab_test")
    if ab:
        with st.expander("A/B test: two openings, same ending"):
            if ab["pair"]:
                st.write(f"Test {ab['pair']['candidate_id']} against {ab['pair']['challenger_id']}; both end at "
                         f"{ab['pair']['cliff_beat_id']}.")
            else:
                st.write("No two shortlisted cuts share an ending with different openings yet.")
            for key in ("arms", "unit", "strata", "analysis"):
                st.write(ab[key])
            st.dataframe(pd.DataFrame([{"Outcome": k, "Window": v} for k, v in ab["windows"].items()]), hide_index=True)
            if ab["sample_size"]:
                st.dataframe(pd.DataFrame([{"Metric": s["metric"], "Base rate": f"{s['base_rate']:.1%}",
                                            "Lift": f"+{s['lift_points']:g} pts", "Viewers per arm": s["viewers_per_arm"]}
                                           for s in ab["sample_size"]]), hide_index=True)
    holdout = report.get("holdout") or {}
    rows = [{"Outcome": OUTCOME_NAMES.get(o, o), "Held-out trials": h["trials"], "Model Brier": h["model_brier"],
             "Base-rate Brier": h["base_rate_brier"], "Cohort-rate Brier": h["cohort_base_rate_brier"],
             "Model AUC": h["model_auc"], "Cohort-rate AUC": h["cohort_base_rate_auc"]}
            for o, h in holdout.items() if isinstance(h, dict)]
    if rows:
        with st.expander("Does it generalise to unseen episodes?"):
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(f"{holdout['folds']} interleaved episode folds. A cut is tested in the fold of its first "
                       "episode and never trained on with any episode of that fold. Lower Brier is better.")


def show_engagement(result: RunResult, report: dict | None = None) -> None:
    """Render a saved or reported engagement report; without one, simulate activity for this run."""
    if not result.candidates and not result.episodes:
        st.info("Load a run to see engagement.")
        return
    if report is None:
        numbers = sorted(ep.number for ep in result.episodes)
        options = [0, *numbers[1:]]
        paywall = st.selectbox("Simulated paywall at episode", options, key="engagement-paywall",
                               format_func=lambda n: "No paywall" if n == 0 else f"Episode {n}")
        try:
            report = _report(result, paywall)
        except ValueError as exc:
            st.error(f"Unable to build the engagement report. {exc}")
            return
    if report["label"] == "SIMULATED":
        st.warning("**SIMULATED ACTIVITY.** Every number below comes from the planted model in "
                   "hookcut/engagement.py. It shows the loop works end to end and that the model recovers effects "
                   "planted on purpose. It is not evidence about Kuku viewers.")
    else:
        st.caption(report["note"])
    st.caption(f"Episode features: {report['feature_source']}.")
    show_matrices(result, report)
    show_retention(report)
    outcome = show_effects(report)
    show_recovery(report, outcome)
    show_rankings(result, report)
    show_insights(result, report)
