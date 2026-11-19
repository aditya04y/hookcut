# Cold-viewer eval: run_full

Created 2026-11-20T00:57:04. Eval judge `groq:openai/gpt-oss-120b`; run judge `groq:qwen/qwen3.8-27b -> groq:openai/gpt-oss-20b`, analyzer `gemini:gemini-3.5-flash-lite -> gemini:gemini-3.8-flash -> groq:openai/gpt-oss-120b`. Planned calls 60 (cap 70); live calls 1, cached 59, errors 0; tokens in 1,431 / out 503.

Means with bootstrap 90% CI and n (judged clips). Verdict = pass / borderline / fail.

| set | clips (judged) | pass rate | standalone 0-5 | keep_watching 0-5 | hook lands | verdicts p/b/f | validator clean | own checks clean | cuts a line |
|---|---|---|---|---|---|---|---|---|---|
| tool | 20 (20) | 80% [65%, 95%] (n=20) | 3.90 [3.80, 4.00] (n=20) | 3.80 [3.65, 3.95] (n=20) | 90% [80%, 100%] (n=20) | 16/4/0 | 100% [100%, 100%] (n=20) | 100% [100%, 100%] (n=20) | 0 |
| last_window | 20 (20) | 20% [5%, 35%] (n=20) | 3.60 [3.40, 3.80] (n=20) | 3.40 [3.20, 3.60] (n=20) | 30% [15%, 45%] (n=20) | 4/15/1 | 60% [40%, 75%] (n=20) | 100% [100%, 100%] (n=20) | 0 |
| random | 20 (20) | 50% [30%, 70%] (n=20) | 3.90 [3.80, 4.00] (n=20) | 3.45 [3.20, 3.70] (n=20) | 55% [35%, 75%] (n=20) | 10/8/2 | 100% [100%, 100%] (n=20) | 100% [100%, 100%] (n=20) | 0 |
| tool_excl_tuning | 18 (18) | 78% [61%, 94%] (n=18) | 3.89 [3.78, 4.00] (n=18) | 3.78 [3.61, 3.94] (n=18) | 89% [78%, 100%] (n=18) | 14/4/0 | 100% [100%, 100%] (n=18) | 100% [100%, 100%] (n=18) | 0 |

Validator issues by check (a clip can have several): tool: none; last_window: cliff_at_end 8; random: none. Last-window clips keep the episode's final frames, so a trailing reaction shot can trip `cliff_at_end`; 'own checks' (length, hook in 3 s, no line cut) is the like-for-like structural comparison.

## Tool minus baseline

| comparison | difference | 90% CI | n tool / n base |
|---|---|---|---|
| tool_minus_last_window.pass | +60 pts | [+40 pts, +80 pts] | 20 / 20 |
| tool_minus_last_window.keep_watching | +0.40 | [+0.15, +0.65] | 20 / 20 |
| tool_minus_last_window.standalone | +0.30 | [+0.10, +0.50] | 20 / 20 |
| tool_minus_random.pass | +30 pts | [+5 pts, +55 pts] | 20 / 20 |
| tool_minus_random.keep_watching | +0.35 | [+0.05, +0.65] | 20 / 20 |
| tool_minus_random.standalone | +0.00 | [-0.15, +0.15] | 20 / 20 |

## Judge agreement (tool clips)

Second judge: run's stored verdicts ['groq:qwen/qwen3.8-27b']. Pairs with both verdicts: n=20.
- verdict agreement (3-way): 45%
- Cohen's kappa, pass vs not-pass: 0.07
- Spearman on keep_watching: 0.13
- mean |standalone difference|: 0.40
- pass rate, eval judge vs second: 80% vs 35%
- Caveat: these clips were selected using the run's judge, so its pass rate is inflated and kappa is depressed by low variance.

## Diagnostics

- Spearman(analyzer prelim_score, eval keep_watching) on tool clips: -0.08

Per-clip ids, timecodes and scores: eval.json. Judge free text (private): outputs/<run>/eval_verdicts.json.
