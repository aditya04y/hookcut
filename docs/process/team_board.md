# HookCut team board

Read this first. The lead owns this file and updates it; workers append one dated status line at the bottom
when they finish (or get blocked).

## What we are building
Kuku's AI Generalist assignment, **Problem 1: Find the Promos**. Input: a show's microdrama screenplays (one
.txt per episode, timestamps on every camera/action/dialogue/sound block). Output: ranked promo clips for
Instagram/TikTok/YouTube. The brief's definition of a good promo:
- **20–60 seconds** total; usually one continuous stretch, but may join a few moments from one or several episodes
- **opens on a hook**: something in the first 3 seconds that stops a scroller (slap, reveal, shocking line)
- **ends on a cliffhanger**: cut at an unresolved moment
- **makes sense on its own** to someone who never saw the show
- for each promo: episode + start/end of every segment in order, the hook, the cliffhanger, one line on why it
  works; then a ranking (what to post first, and why); then an honest "where does the tool pick badly".
- Only the moments and the cut points. **No editing, music or text overlays.**

Second output, same engine (our Problem 3 angle, kept small): a **per-episode report for script editors** — how
strong each episode's opening hook and ending cliffhanger are, and whether the next episode picks the
cliffhanger up (immediately / after a recap / later / not at all).

Graders look at: problem framing, approach & scale (what's AI vs code vs human, cost at volume), quality checking
(real runs, honest failures), a prototype that runs on new input (they will rerun it on a new show in the
interview, maybe with stronger models through their own API — so the model must be swappable), and short clear
communication. "A simple, honest submission that works beats a polished one that overpromises."

## Pitch (from docs/research/competitors.md)
Leaders run thousands of creatives per show (SocialPeta: ~4.1K for one title in a week; NetShort ~770K creatives in
June 2026 vs Kuku TV ~160K). ByteDance's HIVE paper cuts ads from finished video. Nobody publicly picks promos
**from the script before the video exists**. Kuku generates shows from screenplays, so HookCut can hand the promo
team a ranked, cold-viewer-checked shortlist plus variants the day a script locks.

## Architecture and stage contracts (do not change signatures without the lead)
`hookcut/pipeline.py` calls, in order:
| Stage | Function | Owner |
|---|---|---|
| parse | `parse.parse_folder(path) -> list[Episode]` (done) | lead |
| analyze | `analyze.analyze_show(episodes, client, *, log, concurrency=4, progress=None) -> list[EpisodeAnalysis]` | analyst |
| candidates | `candidates.generate_candidates(episodes, analyses, promo_cfg) -> list[Candidate]` | editor |
| judge | `judge.judge_candidates(candidates, episodes, client, *, top_k, log, concurrency=2, progress=None) -> list[ColdViewerVerdict]` | analyst |
| rank | `rank.rank_promos(candidates, verdicts, episodes, analyses, promo_cfg) -> list[Promo]` | editor |
| writer | `writer.write_why_lines(promos, episodes, client, *, log) -> list[Promo]` | analyst |
| report | `report.episode_report(episodes, analyses) -> list[dict]` | editor |
| validate | `validate.validate_promos(promos, episodes, promo_cfg) -> list[dict]` (empty list = clean) | editor |
| export | `export.write_outputs(result: RunResult, out_dir) -> dict[str, str]` | editor |

Data models: `hookcut/schema.py` (lead owns). LLM access: `hookcut/llm/` — every stage receives an `LLMClient`
and calls `client.complete_json(system, user, schema=..., validate=..., cache_tag=...)`; it returns `LLMResult`
(`.data`, tokens, latency, cached). Log each call with `log.record(stage, item, result)` (`hookcut/runlog.py`).
Never import a vendor SDK in a stage. Formatting helpers: `hookcut/textfmt.py` (`tc`, `ep_tc`, `render_beats`).
Config: `config.yaml` (`promo:` section is the `promo_cfg` dict).

## Ownership (edit only your files; propose other changes in your final report)
- **lead**: schema.py, parse.py, textfmt.py, config.py, config.yaml, pipeline.py, runlog.py, llm/*, docs/process/team_board.md, docs/ai_notes.md, README.md
- **analyst** (Claude worker): analyze.py, judge.py, writer.py, prompts/*.md, tests/test_analyze.py, tests/test_judge.py
- **editor** (Claude worker): candidates.py, rank.py, report.py, validate.py, export.py, tests/test_candidates.py, tests/test_rank.py, tests/test_export.py
- **app** (Codex worker, gpt-6.2-sol): app/*, samples/*, tests/test_app.py
- **evaluator** (Claude worker): eval/*, tests/test_eval.py, docs/failures.md

## Ground rules
1. **Confidential input.** The screenplays (`data/pack` → the show folder) are Kuku's. Never copy their text into
   samples/, tests/fixtures, README, or anything that could be published. Tests use synthetic text or read
   `data/pack` at runtime. `samples/` holds original synthetic episodes only.
2. **Secrets.** Never print, log, or copy API keys. To call real models run with
   `HOOKCUT_ENV_FILE=path/to/.env` (only provider key variables are
   read). Never read other variables from that file.
3. Python: `.venv/bin/python`, tests with `.venv/bin/python -m pytest -q`. Keep code small and explainable — the
   candidate must defend every line in an interview. No new heavy deps (no torch). Ask the lead first.
4. Free tiers are small (Groq: 8K tokens/min and 200K tokens/day per model). Use the disk cache (on by default),
   test on 2–3 episodes before any full run, and never start a full 50-episode run yourself — the lead does that.
5. AI mistakes are part of the deliverable: when a model (or you) gets something wrong in an instructive way,
   add one line to `docs/ai_notes.md` under "Worker notes" (what, where, how caught).

## Models (tested 2026-11-19)
- Groq `qwen/qwen3.8-27b` and `openai/gpt-oss-120b`: <1 s per short call, good JSON. Smaller models picked a
  whispered "[the groom's name]?" as Episode 1's best hook; Gemini 4 Hydrogen picked the stronger "[the rival sister's reveal]."
- Gemini (measured 15:00): `gemini-3.8-flash` native returned 503 "high demand"; via the compatible endpoint it
  answered in 66 s and picked the strong hook (E01.B027). `gemini-4-hydrogen` 115 s, also B027.
  `gemini-3.5-flash-lite` 1.7 s but picked a weaker hook. Analysis quality matters most, so analysis stays on
  `gemini-3.8-flash` (fallbacks: 4-hydrogen, then Groq gpt-oss-120b); expect 1–2 min per call and retries.
  The compatible endpoint may not count thinking tokens in `completion_tokens` — cost estimates are a floor.
- Paid prices are in `config.yaml` (`prices_usd_per_mtok`) for cost estimates.

## Decisions log
- 2026-11-19 Submit Problem 1; episode report is the Problem 3 angle (one engine, two users).
- 2026-11-19 Model-neutral client (OpenAI-compatible SDK for Gemini/Groq/OpenRouter/Ollama/custom gateway;
  official Anthropic SDK for Claude). Fallback chains per stage. Disk cache.
- 2026-11-19 Cold-viewer judge uses a different model family from the analyzer (independence).
- 2026-11-19 Editor handoff via OpenTimelineIO (EDL + FCP XML + .otio); embeddings via model2vec potion-base-8M.

## Status (workers append below: date, role, one line)
- 2026-11-19 app — Streamlit demo, four synthetic episodes, and offline fixture complete. Six app tests passed and all sample scripts parsed without warnings.
- 2026-11-19 editor — candidates/rank/report/validate/export done; OTIO installed in venv; 25 editor tests pass (80 suite-wide); smoke on data/pack with fake analyses: 95 candidates (68 single, 27 multi), 20 promos, 0 validation issues. Asks: add OTIO packages to requirements.txt; pipeline should pass analyses to validate_promos (4th arg) or Promo should carry hook/cliff beat ids.
- 2026-11-19 evaluator — harness built (eval/: pre-registration README, baselines, run_eval with 70-call guard, labeling packet + score_labels, cost_model, mine_failures, synthetic fixture); 26 eval tests pass (suite 83 at last full run), no network. Private labeling packet for the 10 hold-out episodes is in outputs/labeling/. Waiting for outputs/<run>/result.json to run the real eval. Ask: judge.py should store the model that answered (`res.provider:res.model`) in ColdViewerVerdict.model, not the fallback chain.
- 2026-11-19 analyst — analyze/judge/writer + prompts (analyze_system@33db640f, judge_system@f3d16702, writer_system@7ce95e31) done; 14 analyst tests pass (83 suite-wide), offline end-to-end smoke on samples/ clean; 15 real calls: Gemini 3.8 Flash meets the E01/E23 sanity targets (~172 s/call), Groq gpt-oss-120b is fast (~3-5 s) but misses E01.B027 in 4 of 5 runs; model fields record the model that answered, failed analyses are "failed:<chain>". Asks: treat Groq 413 TPM as retryable in openai_compat.py.

---------------------------------------------------------------------------------------------------------------
## v2 (2026-11-19 evening): HookCut as the promo layer on Kuku's own three-layer stack
The user changed direction: an LLM wrapper alone is not differentiated. Kuku's published stack (Kuku Stack I–III) is
(1) a recommendation engine on **user-activity matrices**, (2) the **Content Genome** — stories analysed beyond genre
(character types and relationships, plot structure, emotional moments, suspense/tension, what makes audiences keep
watching) and linked to engagement data, and (3) **AI Adda** — an orchestration layer that routes tasks to the best
model, exposes many tools through one interface, connects them to workflows, and lets teams reuse each other's tools.
HookCut v2 mirrors that: the promo engine (unchanged, `RunResult` is the contract) feeds three layers that read a
saved run and add no risk to it.

| Layer | Module(s) | Owner |
|---|---|---|
| Promo engine (done) | parse/analyze/candidates/judge/rank/writer/report/export/validate | lead |
| Content genome | `hookcut/story_graph.py` (new: show-level graph, open loops across episodes, `episode_features`), `hookcut/genome.py` (extend per-cut features), `app/genome_view.py`, prompts/story_graph_system.md | genome worker (Claude) |
| Activity matrices + recommendation | `hookcut/engagement.py` (new: viewer×episode and viewer×cut matrices, simulator, interpretable model, hybrid ranking), extend `hookcut/audience.py` / `hookcut/studio.py` / `app/audience_studio.py`, docs/audience_workflow.md | engagement worker (Claude) |
| AI Adda | `hookcut/adda/` (tool registry, router + quota ledger, YAML workflows, runner, MCP server, CLI), `app/toolkit_view.py` | adda worker (Codex gpt-6.2-sol) |
| App tab wiring, config, pipeline hooks, docs | app/streamlit_app.py, config.yaml, pipeline.py, solution doc | lead |

### v2 contracts
- `story_graph.build_story_graph(result, client=None, *, log=None) -> StoryGraph` (pydantic models live in story_graph.py),
  `story_graph.episode_features(result, graph=None) -> dict[int, dict[str, float]]` (works without a graph, from
  EpisodeAnalysis alone), `story_graph.write_story_graph(graph, out_dir)` -> `story_graph.json`.
- `engagement` consumes `episode_features` and `genome.build_genome`; persists `engagement.json` in the run folder.
  Simulated activity is labelled SIMULATED in every file, table and chart; it proves the loop works, never audience lift.
- `adda` wraps existing functions as tools; it never edits them. Router exposes
  `Router.choose(task, *, quality=..., exclude_families=...) -> list[str]` (provider:model specs) and a `QuotaLedger`
  that remembers exhausted daily quotas from 429 errors (e.g. Gemini "GenerateRequestsPerDayPerProjectPerModel",
  retryDelay) so runs stop wasting retries on dead models. The lead wires it into `config.build_client`.
- Each worker ships a `show_*(result, ...)` Streamlit function in its own view module; the lead adds the tabs.

### v2 ground rules (in addition to the ones above)
- **Disk is almost full (~250 MB free).** Do not install packages or download models; keep outputs small; delete temp
  files you create. Available in .venv: numpy, pandas, pydantic, streamlit, altair, model2vec, opentimelineio, mcp 2.x
  (`from mcp.server.mcpserver import MCPServer`; FastMCP was renamed), json-repair. No scikit-learn: write models in numpy.
- Real model calls only through HOOKCUT_ENV_FILE (see above). Today's quotas: Gemini 3.8 Flash and 4 Hydrogen are
  exhausted (20/day free); gemini-3.5-flash-lite is fine; Groq gpt-oss-120b is reserved for the evaluator's run today.

### v2 addendum: relationship layer follows the user's bidders model
Reference (read-only): the author's tender-intelligence bidder graph, a separate private project. Bipartite contractor↔tender projection → character↔scene projection; Jaccard co-bidding
index → co-presence index; win asymmetry → dominance asymmetry; additive syndicate risk → conflict intensity (core
conflict pairs); bidder profiles → character profiles with "promo pull"; forensic alerts → story alerts (reversals,
identity reveals) with beat evidence; name resolver → alias/identity resolution. Disk: ~1.6 GB free after cleanup.
- 2026-11-19 evaluator — real eval on run_full done (60 gpt-oss-120b calls; 2 hit 429 TPM and were refilled by a cached rerun). Tool top-20 pass 75% vs last-45 s 20% vs random 50%; judge agreement kappa -0.00 (< pre-registered 0.2, so the judge numbers are one model's opinion; the hold-out human labels must carry the claim). Results in eval/results/run_full/ (summary.md, cost.md, flagged.md); failure write-up in docs/private/failures.md. Asks: anonymise unspoken speaker names in judge.clip_text; treat sfx like speech when snapping cuts.
- 2026-11-19 adda — Tool registry, quota-aware router, four workflows, CLI, MCP, and toolkit view complete. 122 tests passed offline. Lead handoff: wire config.build_client with Router and with_quota, then pass the analyzer families when creating the judge client.
- 2026-11-19 engagement — hookcut/engagement.py: viewer×cut (audience.Exposure) and viewer×episode (Session CSV) matrices, PLANTED simulator, four L2-logistic models with cluster-bootstrap CIs, planted-recovery and episode-holdout checks, hybrid per-cohort promo ranking (also `studio.build_studio(prior="genome")` + Audience studio radio), retention, drop-off, paywall and A/B power; `app/engagement_view.show_engagement(result, report=None)`. 146 tests pass. Real run (SIMULATED): planted sign recovery 95–100%, holdout beats base rate on all 4 outcomes; outputs/run_full/engagement.json. Asks: add the Engagement tab; config `engagement:` keys; adda should pass `simulate`/CSVs to the engagement tool.
- 2026-11-19 genome — hookcut/story_graph.py (entity resolution, character↔scene projection with Jaccard/dominance/hostility and an additive conflict_intensity score, character profiles with promo pull, story alerts, open loops, 54 documented episode_features, one show-level LLM pass with code-checked ids), genome.build_genome(result, graph=None) story context per cut, app/genome_view.py, prompts/story_graph_system.md. 10 new tests (156 suite-wide); the only failure is tests/test_engagement.py::test_audience_studio_offers_the_story_feature_prior, which still opens app.tabs[0] as the studio after the tab reorder. Real run: 1 flash-lite call (13.6K in / 4.9K out, 19 s) -> outputs/run_full/story_graph.json: 52 characters, 58 edges (14 named, 7 core conflict pairs), 50 ending questions (48 answered / 2 open) + 6 threads (5 / 1), 47 alerts. Asks: config stage "genome", pipeline hook after export.
- 2026-11-20 graph→rank worker — (A) story_graph: titles/role words (Duchess, disgraced bride, wife, maid, veteran…) are no longer identity aliases but time-scoped `CharacterNode.roles` with an evidence beat (38 claims → 29 roles); script labels (THE HOODED STRANGER, LADY IN [COLOUR] GOWN) stay identities; rebuilt from the same cached reply. (B) `rank_promos(..., graph=None)`: reveal → wave 3 (early-season premise exempt), "feud: A vs B" flags + top-3 feud coverage swap, +0.2 lead-hook bonus; pipeline builds the graph after judging, before ranking (failure → graph=None). Suite green (165 tests; 5 new are mine). Rerun (baseline kept in outputs/run_full_v1, eval/results/run_full_v1): graph changed no membership, moved E42's reveal clip to wave 3 and reordered within waves; 1 of 20 changed (C0053 out, C0052 in) because the primary judge, live this time, failed C0053, which the fallback gpt-oss-20b had passed. Eval tool pass 75% → 80%, hook lands 85% → 90% (that swap, not the graph). Asks: cost.md/flagged.md in eval/results/run_full are from v1; config's genome `reasoning: low` differed from the cached call (made without it), so I filed the same reply under the current key.
- 2026-11-20 release — `tools/export_public.py` builds the public copy in ../hookcut-github: whitelist copy (sources for README, docs/how_it_works.md, docs/process/README.md, requirements split, .gitignore, .streamlit placeholder in tools/public/) → scrub (client names → roles in docs and invented names in code comments, script quotes → [placeholders], local and private-project paths) → scans (names, title, quotes, pack ids; word n-grams shared with the pack, the run outputs and quotes in docs/private; secrets; local paths) → pytest inside the export (164 passed, 1 skipped: no saved full run) → fresh git repo, one commit. README results are filled from eval/results/run_full/summary.md at export time. Re-run after the app lands: `.venv/bin/python tools/export_public.py` (~20 s; exits 1 and commits nothing on any scan hit or test failure). Asks: clean the client names out of the docstrings in story_graph.py/parse.py/schema.py/textfmt.py/rank.py and tests/test_story_graph.py at the source (the export rewrites them today).
