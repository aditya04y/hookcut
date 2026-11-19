# How HookCut works

HookCut has one engine and three optional layers on top of it. The engine turns screenplays into a ranked promo
shortlist and a script-editor report, saved as one `result.json` per run. Each layer reads that saved run and
adds a file next to it, so none of them can change or break the shortlist.

```
screenplays (.txt) ──► promo engine ──► result.json ──┬──► content genome      ──► story_graph.json
                                                      ├──► activity matrices   ──► engagement.json  (SIMULATED unless real data)
                                                      └──► AI Adda: routes models, runs the same tools as workflows, MCP
```

## 1. The promo engine

A promo, as the brief defines it: 20–60 seconds, one stretch or a few moments joined; opens on a hook in the first
3 seconds; ends on a cliffhanger (cut at an unresolved moment); makes sense to someone who never saw the show.

| Stage | Code, AI or human | What it does |
|---|---|---|
| Parse (`parse.py`) | code | Splits each episode into beats: every camera, action, dialogue and sound block with its start and end time. Unknown labels become "other" beats, so a new format degrades instead of failing. |
| Analyze (`analyze.py`) | AI, one call per episode | Scores every notable beat 0–5 as a hook ("would a scroller stop if a clip opened here?") and as a cut point ("how badly do they need the next second if it cuts to black here?"), with the hook type, what a newcomer must already know, spoiler level, and which later beat answers the open question. Also grades the episode's own opening and ending, and how it picks up the previous episode. Code checks every beat id and re-asks with the exact error. |
| Candidates (`candidates.py`) | code | Builds every hook-to-cliffhanger window of 20–60 s: starts on a shot, never cuts a line, ends before the answer. Joins moments across episodes and skips the recap. |
| Judge (`judge.py`) | AI, a different model family | The cold-viewer test. The judge sees only what a viewer would see and hear in the clip and reports the premise, the conflict, the open question and what confused it. A fixed rule turns its scores into a verdict: fail if "makes sense alone" or "want to keep watching" is 2 or less; pass if the hook lands and both are 4 or more. |
| Rank (`rank.py`) | code | Combines beat scores and the verdict, removes near-duplicates (small static embeddings), caps picks per episode, and orders by posting wave: premise first for new viewers, escalation next, big reveals last. Adds variants: other openings for the same ending, shortest and longest cuts. |
| Writer (`writer.py`) | AI, one call | One line per promo on why it works, grounded in the clip. Falls back to a template line if the call fails. |
| Report, validate, export | code | Per-episode report for script editors; structural checks on every promo; CSV / JSON / Markdown plus EDL, FCP XML and OTIO timelines. |
| Review | human | The editor confirms the pick on footage and renders it. |

Every model is a `provider:model` string with a fallback chain and an on-disk cache (`hookcut/llm/`). Stages never
import a vendor SDK; they call one `complete_json` interface, and every call is logged with tokens, latency and
whether it came from the cache, which is what the cost model reads.

## 2. Content genome (`story_graph.py`, `genome.py`)

A show-level story graph built from the saved analyses, mostly in code:

- **Who is who.** Entity resolution merges character names across episodes: honorifics stripped, aliases from the
  per-episode analyses, a gender guard against wrong alias claims, and titles treated as roles that can change
  hands rather than as identities.
- **How they are bound.** Characters are projected through the scenes they share (a character-to-scene graph
  becomes character-to-character edges) with co-presence, dominance (who delivers the strong lines), hostility
  and an additive conflict score that picks out the core conflict pairs. This borrows the method of my
  tender-intelligence bidder graph (co-bidding index, win asymmetry, syndicate risk), applied to a story.
- **What keeps viewers watching.** Open loops: each scored open question and episode-ending question, and the
  episode that answers it (or not). Story alerts flag relationship reversals, identity reveals and long-open loops.
- **One optional AI pass** over compact per-episode digests merges aliases, labels relationships and archetypes,
  links ending questions to their answers and tags tropes from fixed lists. Code checks every id it returns; if the
  call fails, the code-only graph stands.

`genome.build_genome` then gives every candidate cut a small, readable feature set (stakes, kinship, identity
questions, rising suspense, who is on screen, which loops it opens or closes), each pointing back to timed beats.

## 3. Activity matrices (`engagement.py`, `audience.py`, `studio.py`) — simulated

This layer shows how real viewing data would re-rank promos. It has not seen real viewers.

- **Two grains.** Viewer × cut (one row per promo impression: watch seconds, whether the viewer started the show)
  and viewer × episode (started, completed, continued to the next). A skip is an observed zero; a missing cell
  means "never shown".
- **An interpretable model.** Four small logistic regressions in numpy (3-second hold, story start, episode
  completion, next-episode start) on the genome and episode features, with bootstrap intervals that resample whole
  cuts or episodes. Extra story features are tested one at a time, because fitting all of them at once over-fitted.
- **Decisions.** Per-audience promo order (direct counts smoothed toward the model's prior, so a cut with no
  impressions still gets a score), retention and drop-off by episode, candidate paywall points, and the sample
  size for an A/B test of two openings.
- **Checks.** A simulator plants known preferences per cohort; the model must recover them (planted-sign
  recovery) and beat the base rate on held-out episodes. Every simulated file, table and chart says SIMULATED.

More detail and the demo walkthrough: [audience_workflow.md](audience_workflow.md).

## 4. AI Adda (`hookcut/adda/`)

An orchestration layer in the spirit of Kuku's own: one place to find, route and reuse tools.

- **Tool registry.** Every engine stage and layer is wrapped as a tool with typed inputs and outputs, a team
  (promo, script, audience, platform) and a cost hint. Nothing is rewritten; the tools call the same functions.
- **Router and quota ledger.** A per-task routing table (`routing.yaml`) ranks models by tier (quality, balanced,
  budget) with prices and free limits; the judge never shares a model family with the run's analyzer. The ledger
  remembers daily-quota errors (for example a free tier's requests-per-day limit) until the reported reset, so a
  run stops retrying a model that cannot answer today.
- **Workflows.** YAML pipelines composed from tools: `promo_finder` (the full engine), `script_review`
  (reuses the promo team's analysis for script editors), `audience_update` and `quick_check` (no model calls).
  Each run writes a manifest of which model did which step, why, and what it cost.
- **MCP server and CLI.** The same tools are available to any MCP-capable assistant and from the command line:
  [hookcut/adda/README.md](../hookcut/adda/README.md).

## What is code, what is AI, what is human

| Code | AI | Human |
|---|---|---|
| Parsing; every rule (20–60 s, hook in the first 3 s, never cut a line, the answer stays outside the clip); ranking and posting order; story-graph statistics; activity models; editor files | Scoring beats; the cold-viewer test by a model from another family; one "why it works" line per promo; an optional story-graph pass | The final pick, confirmed on footage |
