"""Content genome view: the show's story graph (cast network, conflict pairs, profiles, open loops, tension,
tropes and story alerts). Works from a saved story_graph.json or, without one, from the deterministic layer."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from hookcut.schema import RunResult
from hookcut.story_graph import (CORE_CONFLICT_MIN, StoryGraph, build_story_graph, character_profile,
                                 top_conflict_pairs)

HEAT = [(CORE_CONFLICT_MIN, "#c0392b"), (30.0, "#e08e2b"), (0.0, "#9aa5b1")]   # conflict intensity -> edge colour


def _heat(intensity: float) -> str:
    return next(color for floor, color in HEAT if intensity >= floor)


def _quote(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def network_dot(graph: StoryGraph, max_nodes: int = 16) -> str:
    """Characters as boxes (name + archetype), edges sized by shared scenes and coloured by conflict intensity;
    a solid edge has a named bond (its label is shown), a dashed one is co-presence only."""
    names = {c.id: c for c in graph.characters}
    keep = [c.id for c in graph.characters[:max_nodes]]
    edges = [e for e in graph.relationships if e.a in keep and e.b in keep and (e.kind or e.shared_scenes >= 3)]
    linked = {x for e in edges for x in (e.a, e.b)}
    widest = max((e.shared_scenes for e in edges), default=1) or 1
    lines = ["graph story {", 'graph [layout=neato, overlap=false, splines=true, bgcolor="transparent"];',
             'node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", color="#5b6b7b", fontname="Helvetica", '
             'fontsize=11, fontcolor="#1f2933"];', 'edge [fontname="Helvetica", fontsize=9, fontcolor="#7b8794"];']
    for cid in (c for c in keep if c in linked):
        c = names[cid]
        sub = f"\\n{c.archetype}" if c.archetype and c.archetype != "other" else ""
        lines.append(f'"{_quote(cid)}" [label="{_quote(c.name)}{sub}", penwidth={1 + 4 * c.promo_pull:.1f}];')
    for e in edges:
        label = f', label="{_quote(e.label)}"' if e.label else ""
        style = "solid" if e.kind else "dashed"
        lines.append(f'"{_quote(e.a)}" -- "{_quote(e.b)}" [penwidth={1 + 6 * e.shared_scenes / widest:.1f}, '
                     f'color="{_heat(e.conflict_intensity)}", style={style}{label}];')
    lines.append("}")
    return "\n".join(lines)


def _metrics(graph: StoryGraph) -> None:
    loops = [l for l in graph.loops if l.scope != "scene"]
    cols = st.columns(5)
    cols[0].metric("Characters", len(graph.characters))
    cols[1].metric("Named bonds", sum(bool(e.kind) for e in graph.relationships))
    cols[2].metric("Core conflict pairs", sum(e.core for e in graph.relationships))
    cols[3].metric("Story questions answered / open", f"{sum(l.status == 'answered' for l in loops)} / "
                                                      f"{sum(l.status == 'open' for l in loops)}")
    cols[4].metric("Story alerts", len(graph.alerts))


def show_network(graph: StoryGraph) -> None:
    st.markdown("#### Characters and relationships")
    if not graph.relationships:
        st.info("No two characters share enough scenes to draw a network.")
        return
    st.graphviz_chart(network_dot(graph))
    st.caption(f"Edge width = shared scenes. Colour = conflict intensity: red is a core conflict pair "
               f"(≥ {CORE_CONFLICT_MIN:.0f}/100), amber 30–{CORE_CONFLICT_MIN:.0f}, grey below. Solid edges carry a "
               "named bond; dashed edges are co-presence only. Box border = promo pull.")
    pairs = top_conflict_pairs(graph, 10)
    st.markdown("**Strongest conflict pairs**: the relationships promos can lean on")
    st.dataframe(pd.DataFrame([{
        "Pair": f"{p['a_name']} – {p['b_name']}", "Bond": p["label"] or "—", "Shared scenes": p["shared_scenes"],
        "Co-presence (Jaccard)": p["jaccard"], "Strong lines a / b": f"{p['a_strong']} / {p['b_strong']}",
        "Hostile / warm beats": f"{p['hostile_beats']} / {p['warm_beats']}", "Intensity": p["conflict_intensity"],
        "Score parts": ", ".join(f"{k} {v:g}" for k, v in p["score_parts"].items()), "Core": "yes" if p["core"] else "",
    } for p in pairs]), hide_index=True)


def show_profile(graph: StoryGraph) -> None:
    st.markdown("#### Character profile")
    if not graph.characters:
        st.info("No characters were found in this run.")
        return
    options = [c.id for c in graph.characters]
    names = {c.id: c.name for c in graph.characters}
    who = st.selectbox("Character", options, format_func=lambda cid: names[cid], key="genome_profile")
    p = character_profile(graph, who)
    cols = st.columns(4)
    cols[0].metric("Episodes", len(p["episodes_present"]))
    cols[1].metric("Lines", p["lines"])
    cols[2].metric("Screen time", f"{p['screen_time_s'] / 60:.1f} min")
    cols[3].metric("Promo pull", f"{p['promo_pull']:.0%}")
    cols = st.columns(4)
    cols[0].metric("Hook lines delivered", p["hook_lines"])
    cols[1].metric("Cliffhangers involved", p["cliff_beats"])
    cols[2].metric("Candidate clips", p["clips"])
    cols[3].metric("Final promos", p["promos"])
    roles = ", ".join(f"{r['role']} (E{r['first_ep']}" + (f"–E{r['last_ep']})" if r["last_ep"] != r["first_ep"] else ")")
                      for r in p["roles"])
    st.caption(f"Archetype: {p['archetype'] or 'unknown'} · Episodes {p['first_ep']}–{p['last_ep']} · "
               f"Also called: {', '.join(p['aliases'][:6]) or '—'}" + (f" · Roles over time: {roles}" if roles else ""))
    if p["co_stars"]:
        st.dataframe(pd.DataFrame([{"Co-star": c["partner_name"], "Bond": c["label"] or "—",
                                    "Shared scenes": c["shared_scenes"], "Their strong lines": c["my_strong"],
                                    "Co-star's strong lines": c["partner_strong"],
                                    "Conflict intensity": c["conflict_intensity"], "Core": "yes" if c["core"] else ""}
                                   for c in p["co_stars"]]), hide_index=True)


def show_loops(graph: StoryGraph) -> None:
    st.markdown("#### Open loops: the questions that carry viewers to the next episode")
    scene = st.checkbox("Include in-episode questions", value=False, key="genome_scene_loops")
    last = max((e.ep for e in graph.episodes), default=1)
    rows = [{"Loop": l.id, "Opened": l.opened_ep, "Closes": l.answered_ep or last + 0.5,
             "Status": "open" if l.status == "open" else "answered", "Kind": l.kind, "Scope": l.scope,
             "Question": l.question, "Answered in": f"E{l.answered_ep:02d}" if l.answered_ep else "still open",
             "Basis": l.basis or "—"} for l in graph.loops if scene or l.scope != "scene"]
    if not rows:
        st.info("No open questions were recorded.")
        return
    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_rule(strokeWidth=3).encode(
        x=alt.X("Opened:Q", title="Episode", scale=alt.Scale(domain=[0.5, last + 0.5])), x2="Closes:Q",
        y=alt.Y("Loop:N", sort=alt.EncodingSortField("Opened"), title=None, axis=alt.Axis(labels=len(rows) <= 70)),
        color=alt.Color("Status:N", scale=alt.Scale(domain=["open", "answered"], range=["#c0392b", "#9aa5b1"])),
        tooltip=["Loop", "Question", "Kind", "Scope", "Opened", "Answered in", "Basis"])
    dots = alt.Chart(frame).mark_point(filled=True, size=40).encode(
        x="Opened:Q", y=alt.Y("Loop:N", sort=alt.EncodingSortField("Opened")),
        color=alt.Color("Status:N", legend=None,
                        scale=alt.Scale(domain=["open", "answered"], range=["#c0392b", "#9aa5b1"])))
    st.altair_chart((chart + dots).properties(height=min(900, 14 * len(rows) + 60)))
    st.caption("Each bar runs from the episode that raises a question to the one that answers it; red bars are "
               "still open at the end of the show. Basis: llm = linked by the show-level pass, pickup = the next "
               "episode picks the cliffhanger up (a proxy), analysis = answered inside the episode.")


def show_tension(graph: StoryGraph) -> None:
    st.markdown("#### Tension by episode")
    if not graph.episodes:
        return
    long = []
    for e in graph.episodes:
        long += [{"Episode": e.ep, "Series": "Tension", "Value": e.tension},
                 {"Episode": e.ep, "Series": "Opening hook", "Value": e.opening_hook / 5},
                 {"Episode": e.ep, "Series": "Ending cliffhanger", "Value": e.ending_cliff / 5}]
    frame = pd.DataFrame(long)
    color = alt.Color("Series:N", scale=alt.Scale(domain=["Tension", "Opening hook", "Ending cliffhanger"],
                                                  range=["#2f6fb0", "#2e9e6a", "#c0392b"]))
    base = alt.Chart(frame).encode(x=alt.X("Episode:Q", axis=alt.Axis(tickMinStep=1)),
                                   y=alt.Y("Value:Q", title="0–1", scale=alt.Scale(domain=[0, 1])), color=color,
                                   tooltip=["Episode", "Series", alt.Tooltip("Value:Q", format=".2f")])
    line = base.transform_filter(alt.datum.Series == "Tension").mark_line(point=True)
    marks = base.transform_filter(alt.datum.Series != "Tension").mark_point(filled=True, size=55, opacity=0.8)
    st.altair_chart((line + marks).properties(height=260))
    st.caption("Tension = 0.5 × top-3 cliff scores + 0.2 × top-3 hook scores + 0.3 × the ending cliffhanger "
               "(all / 5). Dots: the episode's own opening hook and ending cliffhanger scores.")


def show_tropes(graph: StoryGraph) -> None:
    st.markdown("#### Tropes")
    counts = pd.Series([t for e in graph.episodes for t in e.tropes]).value_counts()
    if counts.empty:
        st.info("No tropes were tagged.")
        return
    frame = counts.rename_axis("Trope").reset_index(name="Episodes")
    st.altair_chart(alt.Chart(frame).mark_bar(color="#2f6fb0").encode(
        x=alt.X("Episodes:Q"), y=alt.Y("Trope:N", sort="-x", title=None), tooltip=["Trope", "Episodes"])
        .properties(height=24 * len(frame) + 30))


def show_alerts(graph: StoryGraph) -> None:
    st.markdown("#### Story alerts")
    if not graph.alerts:
        st.info("No story alerts.")
        return
    types = sorted({a.type for a in graph.alerts})
    chosen = st.multiselect("Alert types", types, default=[t for t in types if t != "loop_closed"],
                            key="genome_alert_types")
    names = {c.id: c.name for c in graph.characters}
    st.dataframe(pd.DataFrame([{
        "Episode": a.ep, "Type": a.type.replace("_", " "), "Severity": a.severity, "Confidence": a.confidence,
        "Characters": ", ".join(names.get(c, c) for c in a.characters), "Summary": a.summary,
        "Evidence beats": ", ".join(a.beat_ids) or "—"} for a in graph.alerts if a.type in chosen]),
        hide_index=True)


def show_genome(result: RunResult, graph: StoryGraph | dict | None = None) -> None:
    if graph is None:
        graph = build_story_graph(result)
        st.info("Deterministic story graph (no model call): names merged by code, ending questions linked by "
                "the next episode's pickup. Run `python -m hookcut.story_graph --run <run folder>` for the "
                "show-level pass.")
    elif isinstance(graph, dict):
        graph = StoryGraph.model_validate(graph)
    st.caption(f"{graph.show} · story graph by {graph.model}"
               + (f" · {graph.prompt_version}" if graph.prompt_version else ""))
    _metrics(graph)
    show_network(graph)
    show_profile(graph)
    show_loops(graph)
    left, right = st.columns([3, 2])
    with left:
        show_tension(graph)
    with right:
        show_tropes(graph)
    show_alerts(graph)
    with st.expander("Per-episode genome"):
        st.dataframe(pd.DataFrame([{
            "Episode": e.ep, "Tension": e.tension, "Emotions": ", ".join(e.emotions), "Tropes": ", ".join(e.tropes),
            "Questions opened": len(e.loops_opened), "Closed": len(e.loops_closed), "Carried": len(e.loops_carried),
            "Characters": len(e.characters), "Pickup": e.pickup} for e in graph.episodes]), hide_index=True)
    if graph.notes or graph.warnings:
        with st.expander("Notes and checks"):
            for note in graph.notes:
                st.caption(note)
            for warning in graph.warnings:
                st.caption(f"Checked and repaired: {warning}")
