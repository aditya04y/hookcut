# HookCut

HookCut reads a microdrama's screenplays (one timestamped `.txt` per episode) and returns a ranked list of
20–60 second promo cuts: each opens on a hook in the first 3 seconds, ends on an unresolved cliffhanger, and is
checked by a second AI model for whether a stranger would follow it. It works from the script, before any video
exists, so a promo team can plan cuts the day a script locks instead of after an editor has watched every
episode. The same engine writes a per-episode report for script editors (opening hook strength, ending
cliffhanger strength, whether the next episode picks the cliffhanger up). Built for Kuku's AI Generalist
assignment (Problem 1, "Find the Promos"). HookCut only picks moments and cut points; it does no editing.

**Live demo:** [https://hookcut-5xnxxizkewdukazjds7appy.streamlit.app/](https://hookcut-5xnxxizkewdukazjds7appy.streamlit.app/)

## How it works

| # | Step | Done by | What happens |
|---|---|---|---|
| 1 | Parse | code | Each screenplay becomes timestamped beats (camera, action, dialogue, sound). |
| 2 | Score beats | AI, 1 call per episode | Hook and cliffhanger strength per beat, hook type, what a newcomer must know, spoiler level, where each open question is answered. |
| 3 | Build cuts | code | Every hook-to-cliffhanger window of 20–60 s that starts on a shot, never cuts mid-line and keeps the answer outside the clip; cross-episode joins skip the recap. |
| 4 | Cold-viewer test | AI, a different model family | Sees only the clip text and says what it thinks is happening and what confused it; a fixed rule turns its scores into pass / borderline / fail. |
| 5 | Rank and explain | code + 1 AI call | Combines scores and the cold-viewer result, drops near-duplicates, orders by posting wave (premise first, reveals last), adds variants, and writes one "why it works" line per promo. |
| 6 | Review | human editor | Scans the shortlist and opens the EDL / FCP XML / OTIO timeline in Premiere or Resolve to confirm on footage. |

On top of a finished run sit three optional layers: a content genome (story graph of characters, relationships
and open loops), viewer-activity matrices with an interpretable model (runs on **simulated** activity until real
data is supplied), and an orchestration layer ("AI Adda": model routing, quota tracking, reusable workflows, an
MCP server). Details: [docs/how_it_works.md](docs/how_it_works.md).

## Results

On a confidential 50-episode client show (not included in this repo), one cold-viewer judge from a third model
family scored three sets of 20 clips each:

| Set | Passes the cold-viewer test | Hook lands in 3 s |
|---|---|---|
| HookCut top 20 | **80%** (90% CI 65–95) | 90% |
| Last 45 s of 20 episodes (an editor's default) | 20% (5–35) | 30% |
| A random 25–45 s cut from 20 episodes | 50% (30–70) | 55% |

Structural checks (length, hook in 3 s, no line cut) pass on 100% of the top 20. The protocol was
written before the run: [eval/README.md](eval/README.md); numbers: [eval/results/](eval/results/).

## Run it locally

Python 3.11. Keys: `GEMINI_API_KEY` and `GROQ_API_KEY` (both have free tiers).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # then fill in GEMINI_API_KEY and GROQ_API_KEY
python -m hookcut.pipeline --input samples --out outputs/demo
streamlit run app/streamlit_app.py
```

`samples/` is a short original show written for this repo. The app can also open a bundled demo run with no
API key. Run folders hold `promos.md` (readable shortlist), `promos.csv/json`, `all_candidates.csv` (rejected
cuts and why), `episode_report.md/csv`, `editor/` timelines and `run_log.jsonl` (calls, tokens, cost).

Tests run offline, no keys needed: `pip install -r requirements-dev.txt && python -m pytest -q`.

## Swap the model

Every model is a `provider:model` string, set in `config.yaml`, on the command line, or in an env var:

```bash
python -m hookcut.pipeline --input samples --out outputs/claude \
  --analyze anthropic:claude-opus-5-5 --judge groq:openai/gpt-oss-120b --writer groq:qwen/qwen3.8-27b
HOOKCUT_ANALYZE_MODEL=custom:my-model HOOKCUT_BASE_URL=https://gateway.example/v1 HOOKCUT_API_KEY=<key> \
  python -m hookcut.pipeline --input samples --out outputs/gateway
```

Providers: `gemini`, `groq`, `openai`, `openrouter`, `ollama` (local), `anthropic`, and `custom` (any
OpenAI-compatible gateway). Each stage has a fallback chain in `config.yaml`, so a rate limit does not stop a run,
and calls are cached on disk, so a rerun is free. The cold-viewer judge should come from a different model family
than the analyzer.

## Project layout

```
hookcut/           engine: parse, analyze, candidates, judge, rank, writer, report, validate, export
hookcut/llm/       model-neutral clients (OpenAI-compatible and Anthropic), fallback chains, disk cache
hookcut/prompts/   system prompts, with a README on why each rule is there
hookcut/adda/      AI Adda: tool registry, model router and quota ledger, YAML workflows, MCP server
app/               Streamlit app (entry point app/streamlit_app.py); fixtures/ holds offline demo runs
eval/              pre-registered evaluation: baselines, judge agreement, cost model; results/ holds numbers only
samples/           four original synthetic episodes to try the tool on
tests/             offline test suite
docs/              how it works, AI notes, research, and the multi-agent build process (briefs, team board)
```

## Honest limits

- The judges are language models reading a transcript, not people watching video. The two judge models
  disagreed with each other (Cohen's kappa 0.07), so the pass rates are one model's opinion; hand labels on
  held-out episodes (the harness is in `eval/`) are the stronger test. Visual-only hooks read weakly on paper.
- Script timings stand in for the final cut, so an editor still confirms every pick on footage. Short-reply
  hooks ("No.") get over-scored, and judges read character names from speaker labels that a viewer never hears.
- The audience layer runs on simulated viewers. It shows the loop works (it recovers preferences planted in the
  simulator); it says nothing about real audience lift.

## Built with AI

Built with Claude Code (a lead agent writing briefs for parallel worker subagents, coordinated through a shared
team board) and OpenAI Codex (the first app and the AI Adda layer, plus an independent editor review of the top
picks). Inside the tool, Gemini and Groq-hosted open models (Qwen, gpt-oss) score beats, judge clips and write
the explanations. The briefs and team board are in [docs/process/](docs/process/); the places where AI got things
wrong, and how each was caught, are in [docs/ai_notes.md](docs/ai_notes.md).
