"""How HookCut works: who does what, how good it is, what it costs, where it falls short."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.hr import components as C  # noqa: E402
from app.hr import data as D  # noqa: E402
from app.hr import style  # noqa: E402

style.title("How HookCut works", "Code does the bookkeeping, AI makes the judgment calls, and an editor makes the final pick.")

st.markdown("""<ol class="hc-flow">
<li><span class="hc-who code">Code</span><span>Reads each episode into timestamped lines: dialogue, action, camera, sound.</span></li>
<li><span class="hc-who ai">AI</span><span>Scores every line: would it stop someone scrolling, and would cutting there leave a question hanging?</span></li>
<li><span class="hc-who code">Code</span><span>Builds every cut that follows the rules: 20–60 seconds, the hook in the first 3 seconds, no line cut in half, the answer kept after the cut.</span></li>
<li><span class="hc-who ai">AI</span><span>A second model, from a different company, watches each cut as a stranger and says what it understood.</span></li>
<li><span class="hc-who code">Code</span><span>Ranks the cuts, avoids near-duplicates, holds big twists back, and orders them for posting.</span></li>
<li><span class="hc-who human">Editor</span><span>Reviews the shortlist and opens the chosen cuts straight in Premiere or Resolve.</span></li>
</ol>""", unsafe_allow_html=True)

style.h2("Is it any good?")
stats = D.eval_stats()
if stats:
    st.altair_chart(C.eval_chart(stats), width="stretch")
    t = stats["tool"]
    style.note(f"On a real 50-episode show, a judge model from a third company passed {t['pass_rate']['mean']:.0%} of "
               f"HookCut's top 20, against {stats['last_window']['pass_rate']['mean']:.0%} for the last 45 seconds of each "
               f"episode and {stats['random']['pass_rate']['mean']:.0%} for random cuts. Red lines show the 90% range.")
style.note("Two judge models often disagree with each other, so these scores are one model's opinion. Hand-labelled "
           "episodes are the real test, and an editor reviewing the shortlist stays in the loop.")

style.h2("Plugged into Kuku's own stack")
st.markdown("""
<div class="hc-layer real"><b>Content genome</b><span class="hc-tag">Real, built from the scripts</span><br>
A map of the story: who the characters are, how they relate, which questions stay open and where the twists land.
The ranking uses it to keep twists for people who already know the show.</div>
<div class="hc-layer sim"><b>Viewer activity</b><span class="hc-tag">Simulated</span><br>
Tables of which viewer saw which promo and which episode, and a model that links story features to watching.
The demo data is made up with known preferences, and the model finds them. That proves the plumbing, not the
audience, until real data is plugged in.</div>
<div class="hc-layer real"><b>Model routing</b><span class="hc-tag">Real</span><br>
Sends each task to a suitable model, skips models that have run out of free quota, and lets any AI assistant use
HookCut's tools.</div>""", unsafe_allow_html=True)

style.h2("What it costs")
style.note("About $0.63 per 50-episode show at paid model prices, or $126 a month for 200 shows. Re-running a show is "
           "free because results are cached. An editor's estimated time drops from about 133 minutes a show (watch and "
           "pick) to about 30 (review the shortlist).")

style.h2("Where it falls short")
st.markdown("""
- A one-word reply like “No.” can be picked as a hook, though it needs the line before it.
- The stranger judge reads character names that a real viewer would never hear, so it is a little generous.
- A hook that is purely visual, like a raised fist, reads weakly on paper. Those are flagged for the editor to check on footage.
""")
if url := os.environ.get("HOOKCUT_REPO_URL"):
    st.markdown(f"[Code on GitHub]({url})")
