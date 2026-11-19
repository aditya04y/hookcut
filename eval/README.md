# HookCut evaluation: pre-registration

Written 2026-11-19, **before any prompt tuning or full run**. Numbers, splits and failure criteria below are
fixed; any later change gets a dated line under "Changes" with the reason.

## Splits
- **Tuning episodes: 1, 23, 44.** The analyst may look at outputs and tune prompts only on these.
- **Hold-out episodes: 5, 12, 19, 27, 33, 38, 41, 46, 48, 50** (spread across the 50-episode season). Nobody
  tunes on them. A human labels them (below) **before** looking at the tool's picks for those episodes.
- Constants live in `eval/__init__.py`.

## What we measure
1. **Cold-viewer test vs baselines** (`eval/run_eval.py`). Three sets of N=20 clips, all judged by the same
   cold-viewer function (`hookcut.judge.judge_candidates`) with an **eval judge from a different model family**
   than the run's own judge and analyzer (default `groq:openai/gpt-oss-120b`; the run uses Gemini + Qwen):
   - **tool**: the tool's top-20 ranked promos;
   - **last-window**: the final ~45 s of 20 evenly spaced episodes, snapped so no line is cut (a naive editor's
     default, and a strong one: episodes end on cliffhangers);
   - **random**: one random 25-45 s window from each of 20 evenly spaced episodes (seed 7), snapped the same way.

   Per set: pass rate, mean `standalone` (0-5), mean `keep_watching` (0-5), `hook_lands` rate, structural
   validity (`hookcut.validate.validate_promos` with zero issues, plus our own "no dialogue/action line cut"
   check). Every number is reported with its denominator and a bootstrap 90% CI (2,000 resamples). We also
   report tool minus each baseline with a CI, and the tool's numbers excluding clips from tuning episodes.
2. **Judge agreement.** The tool's top-20 is judged by a second model (default: the run's own stored verdicts,
   which costs no calls; or `--judge2 provider:model`). We report verdict agreement %, Cohen's kappa on
   pass vs not-pass, and Spearman on `keep_watching`. Caveat stated up front: the top-20 was *selected* with the
   run judge, so pass rates there are inflated and kappa is depressed by low variance.
3. **Human labels on hold-out** (`eval/labeling.py`, `eval/score_labels.py`). The candidate (the human on
   this team) fills `outputs/labeling/labels.csv` with the best promo window per hold-out episode (one or more
   rows per episode). Not done yet; the harness is ready. We report temporal IoU of the tool's best pick, and
   **hit@1 / hit@3** (hit = IoU >= 0.3, or hook within +-3 s of the labelled start and cliff within +-3 s of
   the labelled end), with denominators, for the tool and for the last-window baseline (no model calls).
4. **Cost at volume** (`eval/cost_model.py`): measured tokens per episode and per show from the run log,
   priced for 200 shows x 50 episodes/month under several model choices; free-tier capacity in shows/day.

## Budget
- At most **70 judge calls per `run_eval` invocation** (enforced twice: planned count before any call, and a
  hard counter on the client). Default plan: 3 x 20 = 60 calls. The disk cache stays on, so reruns are free.
- Rough size: ~1.5-3K tokens per judge call, so one invocation is ~100-180K tokens: most of a Groq free
  model's 200K tokens/day, ~15-25 min at 8K tokens/min. Do not run it on the same day/model as a full pipeline run.

## What counts as the tool failing (decided now)
- **Fails** if the tool's top-20 pass rate is **not above** the last-window baseline's, or its mean
  `keep_watching` is not above it.
- **Inconclusive, not a win** if the tool is ahead but the 90% CI of the difference includes 0 (likely at N=20;
  we will say so rather than claim a win).
- **Fails on usability** if more than 10% of the top-20 have structural issues (wrong length, cut line, hook
  outside the first 3 s), or if hit@3 on the hold-out labels is below 5/10.
- **Judge not trustworthy** if kappa between the two judges is below 0.2: then cold-viewer numbers are reported
  as one model's opinion only, and human labels carry the claim.

## Known limits
- The judge is an LLM reading a transcript, not a person watching video: visual-only hooks are under-rated.
- N=20 per set gives wide CIs (roughly +-18 points on a pass rate). This is a sanity check, not a benchmark.
- Future benchmark: ByteDance's DramaAD set (2,500+ episodes, 500 professional ad clips, HIVE, EMNLP 2025)
  would let us score against real editors' cuts, once scripts can be aligned to it.

## Commands
```bash
# cold-viewer eval (prints the plan first; --dry-run to stop there)
HOOKCUT_ENV_FILE=path/to/.env \
  .venv/bin/python -m eval.run_eval --run outputs/<run>/result.json --judge groq:openai/gpt-oss-120b --n 20
# labeling packet (private, git-ignored) + scoring once labels exist
.venv/bin/python -m eval.labeling --input data/pack
.venv/bin/python -m eval.score_labels --run outputs/<run>/result.json --labels outputs/labeling/labels.csv
# cost model
.venv/bin/python -m eval.cost_model --run outputs/<run>
# flag likely bad picks for docs/failures.md (ids, timecodes, flags; no script text)
.venv/bin/python -m eval.mine_failures --run outputs/<run>/result.json --eval eval/results/<run>/eval.json
```
Outputs land in `eval/results/<run>/` (numbers and ids only; per-clip judge text stays in `outputs/<run>/`).

## Changes
- 2026-11-19 (before any real run) Random windows also end within 1.5 s of their last line (hookcut.validate's
  `cliff_at_end` rule). Last-window clips keep the episode's final frames, as defined above.
- 2026-11-19 run_full: 2 of the 60 judge calls hit Groq's 8K tokens/min limit (429). The same command was rerun
  with `--concurrency 1`: 58 calls came from the cache and 2 were live, so there were 60 live calls in total and
  n=20 in every set. The protocol is unchanged. The first pass's summary is kept privately in
  `outputs/run_full/eval_summary_pass1.md` (tool 79% on n=19, same conclusions).
