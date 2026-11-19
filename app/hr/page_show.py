"""A whole show: the ranked list, the season view for script editors, and the map of who the story is about."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.hr import components as C  # noqa: E402
from app.hr import data as D  # noqa: E402
from app.hr import style, views  # noqa: E402
from hookcut.schema import RunResult  # noqa: E402

style.title("A whole show at once", "Every episode goes through the same five steps. What comes out: promos ranked "
            "in the order to post them, a season view for the script editors, and a map of who the story is about.")

choices = ["The sample show"] + (["A run on this computer"] if D.local_runs() else []) + ["Upload a result.json"]
source = st.segmented_control("Run to show", choices, default=choices[0], key="show-source")
r, folder = None, None
if source in (None, "The sample show"):
    r = D.sample_result()
elif source == "A run on this computer":
    path = st.selectbox("Run", D.local_runs(), format_func=lambda p: p.parent.name, key="show-run")
    r, folder = D.load_result(path), path.parent
else:
    up = st.file_uploader("result.json from a HookCut run (it's in the raw outputs folder)", type=["json"], key="show-up")
    if up is not None:
        try:
            r = RunResult.model_validate_json(up.getvalue())
        except Exception:
            st.error("That file isn't a HookCut result.json. Use the result.json from a run's output folder.")
if r is None:
    st.stop()

st.markdown(f"**{r.show}**: {len(r.episodes)} episodes, {len(r.candidates)} possible cuts, "
            f"{len(r.verdicts)} shown to a stranger, {len(r.promos)} recommended.")

style.h2("Promos, in the order to post them")
rows = [{"#": p.rank, "When": C.WAVE_WORDS.get(p.post_wave, ""), "Cut": C.cut_label(p.segments),
         "Opens on": p.hook_text[:80], "Stranger test": C.VERDICT_WORDS.get(p.cold_viewer.verdict, "") if p.cold_viewer else ""}
        for p in r.promos]
picked = st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", on_select="rerun",
                      selection_mode="single-row", key="show-table")
sel = picked.selection.rows if picked and picked.selection else []
if r.promos:
    p = r.promos[sel[0]] if sel else r.promos[0]
    st.caption("Click a row to open that promo." if not sel else f"Promo #{p.rank}")
    views.show_promo(r, p, key=f"show-{p.candidate_id}")

style.h2("Season view for script editors")
chart = C.season_chart(r.episode_report)
if chart is not None:
    st.altair_chart(chart, width="stretch")
    from hookcut.report import season_summary
    s = season_summary(r.episode_report)
    weak_end = [f"E{x['episode']:02d} ({x['score']}/5)" for x in s.get("weakest_endings", []) if x["score"] < 4][:4]
    weak_open = [f"E{x['episode']:02d} ({x['score']}/5)" for x in s.get("weakest_openings", []) if x["score"] < 3][:4]
    unlock = ", ".join(f"E{x['episode']:02d}" for x in s.get("unlock_candidates", []))
    style.note("Endings worth a rewrite: " + (", ".join(weak_end) or "none, every ending scores 4 or more") +
               ". Openings worth a rewrite: " + (", ".join(weak_open) or "none below 3") +
               ". Strongest cliffhangers where free episodes usually run out (good places to start charging): " +
               (unlock or "none, the show is shorter than 5 episodes") + ".")

style.h2("Who the story is about")
graph = None
try:
    from hookcut import story_graph as SG
    saved = folder / "story_graph.json" if folder else None
    graph = SG.load_story_graph(saved) if saved and saved.exists() else SG.build_story_graph(r)
except Exception:
    graph = None
if graph is not None and graph.characters:
    st.graphviz_chart(C.story_dot(graph), width="content")
    style.note("Boxes are the main characters. Lines join people who share scenes; thicker means more scenes together. "
               "Red lines are feuds, the conflicts promos lean on.")
    reveals = [a for a in graph.alerts if a.type == "identity_reveal"][:3]
    if reveals:
        st.markdown("**Twists the ranking protects** (kept for people who already know the show):  \n" +
                    "  \n".join(f"Episode {a.ep}: {a.summary}" for a in reveals))
else:
    style.note("The story map needs episode analyses; this run doesn't have them.")
