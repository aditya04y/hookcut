"""Streamlit blocks shared by the reviewer pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from hookcut.schema import Promo, RunResult

from . import components as C
from . import style


def show_promo(r: RunResult, p: Promo, key: str) -> None:
    """Timeline strip, marked-up script and the vertical promo frame for one promo."""
    from hookcut.validate import validate_promos
    ep_by = {e.number: e for e in r.episodes}
    an_by = {a.episode: a for a in r.analyses}
    try:
        issues = validate_promos([p], [ep_by[s.episode] for s in p.segments if s.episode in ep_by],
                                 (r.config or {}).get("promo") or {}, r.analyses)
    except Exception:
        issues = []
    episodes = list(dict.fromkeys(s.episode for s in p.segments if s.episode in ep_by))

    style.h2("Where it sits in the episode")
    for n in episodes:
        if len(episodes) > 1:
            st.caption(f"Episode {n}")
        st.markdown(C.timeline_svg(ep_by[n], an_by.get(n), p.segments), unsafe_allow_html=True)
    st.markdown(C.legend_html(), unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        style.h2("The script, marked up")
        whole = st.toggle("Show the whole episode", value=False, key=f"{key}-whole")
        for n in episodes:
            st.markdown(C.script_page_html(ep_by[n], an_by.get(n), p.segments, context_s=None if whole else 12.0,
                                           hook_beat=p.hook_beat_id, cliff_beat=p.cliff_beat_id), unsafe_allow_html=True)
    with right:
        style.h2("The promo")
        st.markdown(C.phone_html(p, ep_by) + C.facts_html(p, issues), unsafe_allow_html=True)


def show_candidates(r: RunResult, limit: int = 14) -> None:
    verdict = {v.candidate_id: v for v in r.verdicts}
    rows = []
    for c in sorted(r.candidates, key=lambda c: -c.prelim_score)[:limit]:
        v = verdict.get(c.id)
        rows.append({"Cut": C.cut_label(c.segments), "Seconds": round(c.duration),
                     "Hook (0–5)": c.hook_score, "Cliffhanger (0–5)": c.cliff_score,
                     "Stranger test": C.VERDICT_WORDS.get(v.verdict, "—") if v else "Not tested",
                     "What confused them": "; ".join(v.confusions[:2]) if v and v.confusions else ""})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.write("No cut passed the rules for this episode.")
