# Cold-viewer eval: run_full

Created 2026-11-19T18:57:55. Eval judge `groq:openai/gpt-oss-120b`; run judge `groq:qwen/qwen3.8-27b -> groq:openai/gpt-oss-20b`, analyzer `gemini:gemini-3.5-flash-lite -> gemini:gemini-3.8-flash -> groq:openai/gpt-oss-120b`. Planned calls 60 (cap 70); live calls 2, cached 58, errors 0; tokens in 3,839 / out 1,153.

Means with bootstrap 90% CI and n (judged clips). Verdict = pass / borderline / fail.

| set | clips (judged) | pass rate | standalone 0-5 | keep_watching 0-5 | hook lands | verdicts p/b/f | validator clean | own checks clean | cuts a line |
|---|---|---|---|---|---|---|---|---|---|
| tool | 20 (20) | 75% [60%, 90%] (n=20) | 3.90 [3.80, 4.00] (n=20) | 3.75 [3.60, 3.90] (n=20) | 85% [70%, 95%] (n=20) | 15/5/0 | 100% [100%, 100%] (n=20) | 100% [100%, 100%] (n=20) | 0 |
| last_window | 20 (20) | 20% [5%, 35%] (n=20) | 3.60 [3.40, 3.80] (n=20) | 3.40 [3.20, 3.60] (n=20) | 30% [15%, 45%] (n=20) | 4/15/1 | 60% [40%, 75%] (n=20) | 100% [100%, 100%] (n=20) | 0 |
| random | 20 (20) | 50% [30%, 70%] (n=20) | 3.90 [3.80, 4.00] (n=20) | 3.45 [3.20, 3.70] (n=20) | 55% [35%, 75%] (n=20) | 10/8/2 | 100% [100%, 100%] (n=20) | 100% [100%, 100%] (n=20) | 0 |
| tool_excl_tuning | 18 (18) | 72% [56%, 89%] (n=18) | 3.89 [3.72, 4.00] (n=18) | 3.72 [3.56, 3.89] (n=18) | 83% [67%, 94%] (n=18) | 13/5/0 | 100% [100%, 100%] (n=18) | 100% [100%, 100%] (n=18) | 0 |

Validator issues by check (a clip can have several): tool: none; last_window: cliff_at_end 8; random: none. Last-window clips keep the episode's final frames, so a trailing reaction shot can trip `cliff_at_end`; 'own checks' (length, hook in 3 s, no line cut) is the like-for-like structural comparison.

## Tool minus baseline

| comparison | difference | 90% CI | n tool / n base |
|---|---|---|---|
| tool_minus_last_window.pass | +55 pts | [+30 pts, +75 pts] | 20 / 20 |
| tool_minus_last_window.keep_watching | +0.35 | [+0.05, +0.60] | 20 / 20 |
| tool_minus_last_window.standalone | +0.30 | [+0.10, +0.50] | 20 / 20 |
| tool_minus_random.pass | +25 pts | [+0 pts, +50 pts] | 20 / 20 |
| tool_minus_random.keep_watching | +0.30 | [+0.00, +0.60] | 20 / 20 |
| tool_minus_random.standalone | +0.00 | [-0.15, +0.15] | 20 / 20 |

## Judge agreement (tool clips)

Second judge: run's stored verdicts ['groq:openai/gpt-oss-20b', 'groq:qwen/qwen3.8-27b']. Pairs with both verdicts: n=20.
- verdict agreement (3-way): 45%
- Cohen's kappa, pass vs not-pass: -0.00
- Spearman on keep_watching: 0.10
- mean |standalone difference|: 0.35
- pass rate, eval judge vs second: 75% vs 40%
- Caveat: these clips were selected using the run's judge, so its pass rate is inflated and kappa is depressed by low variance.

## Diagnostics

- Spearman(analyzer prelim_score, eval keep_watching) on tool clips: 0.10

## Warnings

- eval judge family 'openai' also judged in the run (1 of 64 run judge calls; calls by family: {'qwen': 63, 'openai': 1}): not independent for those clips

Per-clip ids, timecodes and scores: eval.json. Judge free text (private): outputs/<run>/eval_verdicts.json.
