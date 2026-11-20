"""Try one episode: pick a sample or paste a script, watch the five steps, see the marked-up result."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.hr import components as C  # noqa: E402
from app.hr import data as D  # noqa: E402
from app.hr import style, views  # noqa: E402

FORMAT = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: One sentence about the episode.

SCENE 01
Scene heading: INT. ROOFTOP BAR – NIGHT
Scene start: 00:00.00
Scene end: 01:05.00

**CAMERA:**
Start: (00:00.00) | End: (00:03.20)
Close-up on a ring dropping into a champagne glass.

**MAYA (Cold, Quiet):**
Start: (00:03.40) | End: (00:05.10)
"Keep it. You'll need it for her."
"""

style.title("HookCut", "Give it a microdrama script and it finds the promo: a line in the first 3 seconds that "
            "stops someone scrolling, and a cut that leaves them needing to know what happens next.")

sample = D.sample_result()
live_ok = D.live_models_available()
source = st.segmented_control("Script to use", ["A sample episode", "Paste a script", "Upload a script"],
                              default="A sample episode", key="try-source")
text, filename, run_live, number = None, "pasted-episode-001.txt", False, None
if source in (None, "A sample episode"):
    options = {e.number: f"Episode {e.number}: {e.logline}" for e in sample.episodes}
    number = st.selectbox("Episode", list(options), format_func=options.get, key="try-episode")
    st.caption("From a short made-up show written for this demo, so you can see real output without any setup.")
    run_live = st.toggle("Run it live with the AI models (about 10 seconds)", value=False, disabled=not live_ok,
                         help=None if live_ok else "This copy of the app has no model key set, so it shows the saved run.")
    if run_live:
        path = sorted((ROOT / "samples").glob(f"episode-{number:03d}*.txt"))
        if path:
            text, filename = path[0].read_text(encoding="utf-8"), path[0].name
elif source == "Paste a script":
    text = st.text_area("One episode, with timestamps", height=230, placeholder=FORMAT, key="try-paste") or None
    with st.expander("What the script needs to look like"):
        st.code(FORMAT, language=None)
else:
    upload = st.file_uploader("One episode as a .txt file", type=["txt"], key="try-upload")
    if upload is not None:
        text, filename = upload.getvalue().decode("utf-8-sig", errors="replace"), upload.name

needs_text = source in ("Paste a script", "Upload a script")
if st.button("Find the promo", type="primary", disabled=needs_text and not text, key="try-go"):
    st.session_state.pop("try-result", None)
    st.session_state.pop("try-error", None)
    if not needs_text and not run_live:
        st.session_state["try-result"] = D.episode_slice(sample, number)
    elif not live_ok:
        st.session_state["try-error"] = ("Running a new script needs a model key, and this copy of the app has none. "
                                         "The sample episodes still work without one.")
    else:
        steps = [{"title": t, "state": "waiting", "result": ""} for t in D.STEP_TITLES]
        board = st.empty()

        def on_step(i: int, state: str, message: str) -> None:
            steps[i].update(state=state, result=message)
            board.markdown(C.steps_html(steps), unsafe_allow_html=True)

        try:
            st.session_state["try-result"] = D.run_one_episode(filename, text, on_step)
        except Exception as exc:  # shown to the reviewer in plain words
            st.session_state["try-error"] = str(exc)[:400]
        board.empty()

if err := st.session_state.get("try-error"):
    st.error(err)

r = st.session_state.get("try-result")
if r is not None and r.episodes:
    steps = [{"title": t, "state": "done", "result": m} for t, m in zip(D.STEP_TITLES, D.step_summaries(r))]
    st.markdown(C.steps_html(steps), unsafe_allow_html=True)
    if r.promos:
        views.show_promo(r, r.promos[0], key="try")
    elif (fallback := D.review_only_promo(r)) is not None:
        st.info("No cut passed the stranger test this time (live judging varies from run to run). Here is the best "
                "cut it found, so you can see where it sits and what confused the stranger.")
        views.show_promo(r, fallback, key="try")
    else:
        st.info("No cut passed the rules for this episode. The full list is below.")
    with st.expander("Every cut it considered, and what the stranger said"):
        views.show_candidates(r)
