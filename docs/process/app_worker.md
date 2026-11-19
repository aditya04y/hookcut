You are the "app" worker on a small team building HookCut, a submission for Kuku's AI Generalist hiring
assignment (Problem 1: find promo clips in microdrama screenplays). The lead is another AI agent; three other
workers are building the engine in parallel. Work only inside the hookcut repo.

FIRST read: docs/process/team_board.md (goal, contracts, ownership, ground rules), hookcut/schema.py (data models),
hookcut/pipeline.py (run_pipeline signature), config.yaml, docs/research/competitors.md (pitch).

YOUR FILES (create/edit only these): app/** , samples/** , tests/test_app.py . Do not edit anything else, do not
git commit (the lead commits), do not install packages (use .venv/bin/python; streamlit, pandas, altair, numpy,
pydantic are installed), do not call real LLM APIs (you have no keys), and do NOT read data/ or the folder holding the client's original pack (confidential client scripts).

BUILD
1. app/streamlit_app.py — the demo the graders will open.
   - Sidebar: model pickers for the three stages (analyze / judge / writer) as "provider:model" text inputs,
     prefilled from config.yaml, plus presets: "Free tier (Gemini + Groq)" = config defaults; "Fast (Groq only)"
     = groq:openai/gpt-oss-120b for analyze and writer, groq:qwen/qwen3.8-27b for judge; "Claude" =
     anthropic:claude-opus-5-5 for all three; "Custom gateway" = custom:<model> (uses HOOKCUT_BASE_URL /
     HOOKCUT_API_KEY). Promo settings: final_top_n, judge_top_k. Cache toggle.
   - Input, three ways: (a) upload one or more episode .txt files; (b) "Use the synthetic sample show"
     (samples/*.txt); (c) "Open a saved run": pick outputs/*/result.json or upload a result.json. Browsing a saved
     run must need no API key.
   - Run: build cfg with hookcut.config.load_config(overrides=...) and call
     hookcut.pipeline.run_pipeline(inputs=[(filename, text), ...], cfg=cfg, out_dir=Path("outputs/app_runs/<ts>"),
     progress=callback) with st.progress + a status line. Catch exceptions and show them readably (e.g. missing
     key -> say which env var/secret is needed).
   - Tabs: "Promos" (one card per promo in rank order: rank, post wave badge (1 cold audiences / 2 escalation /
     3 big reveals), every segment as "E01 00:39.00–01:23.70", duration, hook time+text, cliffhanger time+text,
     open question, why it works, cold-viewer premise + verdict + confusions, flags, and an expander with the
     variants; filter by wave); "Episode report" (table + chart of opening-hook and ending-cliffhanger scores per
     episode, pickup status and delay; this is the script-editor view); "Timeline" (per selected episode: beat
     hook/cliff scores over time from result.analyses, with promo windows shaded); "All candidates" (table incl.
     rejected ones and cold-viewer reasons); "Run log & cost" (calls, tokens, latency, cached, est. cost per
     stage from run_log rows); "Downloads" (zip of the run folder, plus single files when present: promos.csv,
     promos.json, promos.md, episode_report.csv, editor/*).
   - Secrets on Streamlit Community Cloud: copy only these names from st.secrets into os.environ if present:
     GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, HOOKCUT_BASE_URL,
     HOOKCUT_API_KEY. Never display a key. Wrap st.secrets access in try/except (no secrets file locally).
   - A short "How it works" expander: parse (code) -> score beats per episode (LLM) -> build 20–60 s clips
     (code) -> cold-viewer test by a different model (LLM) -> rank with diversity (code) -> why-it-works lines
     (LLM); human editor reviews the shortlist. Keep copy short and plain.
   - Must work with partial data (e.g. a RunResult with no verdicts, or analyses missing for an episode).
2. samples/ — four ORIGINAL synthetic episodes (60–100 s each) of a made-up contemporary US romance-drama with
   clear hooks and cliffhangers, in exactly this file format (this example is ours, not the client's):

EPISODE NUMBER: Episode 1
EPISODE LOGLINE: Maya crashes her ex-fiancé's engagement party to return the ring and learns the bride is her sister.
CONTINUITY NOTES: First episode. Establishes Maya, her ex Daniel, and her sister Lena.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. ROOFTOP BAR – NIGHT
Scene start: 00:00.00
Scene end: 01:05.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:03.20)
Close-up on a diamond ring dropping into a champagne glass.

**ACTION:**
Start: (00:00.40) | End: (00:03.20)
Maya stands at the bar watching the ring sink, jaw tight.

**MAYA (Cold, Quiet):**
Start: (00:03.40) | End: (00:05.10)
"Keep it. You'll need it for her."

**SOUND FX:**
Start: (00:05.20) | End: (00:06.00)
A glass shatters somewhere behind them.

   Name files like samples/episode-001-sample.txt. Verify with
   .venv/bin/python -c "from hookcut.parse import parse_folder; [print(e.number, len(e.beats), e.warnings) for e in parse_folder('samples')]"
   (zero warnings). Each episode should end on a clear cliffhanger; episode 2+ should open by picking up the
   previous cliffhanger.
3. app/fixtures/demo_run.json — a hand-built RunResult (valid against hookcut.schema.RunResult) for the sample
   show: parsed episodes from samples/ plus plausible hand-written analyses, ~8 candidates, verdicts, ~5 promos
   with variants, episode_report rows and a few run_log rows. Generate it with a small script
   app/fixtures/make_demo_run.py so it stays valid. The app should offer it as "Demo run (no API calls)".
4. tests/test_app.py — streamlit.testing.v1.AppTest smoke tests: the app loads, opening the demo run renders
   every tab without exceptions, and the fixture validates against RunResult. Run:
   .venv/bin/python -m pytest -q tests/test_app.py

When done, end with a short report: what you built, how to run it (.venv/bin/streamlit run app/streamlit_app.py),
test results, and anything in other people's files you'd like the lead to change. Also append one dated line
under "Status" at the bottom of docs/process/team_board.md (that line is the only edit you may make outside your files).
