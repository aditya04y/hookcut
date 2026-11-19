# Cost at volume: run_full

Measured on 50 episodes (run `run_full`) (wall time 4.3 min, mostly cache hits: see live-call latency below). Volume: 200 shows x 50 episodes = 10,000 episodes/month.

## Measured tokens

| stage | calls (cached) | errors | input tok | output tok | max per call | models |
|---|---|---|---|---|---|---|
| analyze | 50 (50) | 0 | 616,390 | 129,939 | 25,702 | gemini-3.5-flash-lite x50 |
| judge | 64 (40) | 0 | 90,689 | 11,244 | 2,611 | qwen/qwen3.8-27b x63, openai/gpt-oss-20b x1 |
| writer | 1 (0) | 0 | 3,139 | 930 | 4,069 | qwen/qwen3.8-27b x1 |

Cached rows carry the token counts of the original call, so a cached rerun still measures a fresh run.

Check against live (uncached) calls in other runs:

| stage | model | live calls | mean input | mean output | mean latency |
|---|---|---|---|---|---|
| analyze | gemini-3.5-flash-lite | 50 | 12,328 | 2,599 | 16.4 s |
| analyze | gemini-3.8-flash | 4 | 6,422 | 2,224 | 226.7 s |
| analyze | gemini-4-hydrogen | 6 | 6,828 | 2,266 | 152.0 s |
| analyze | openai/gpt-oss-120b | 1 | 11,599 | 2,572 | 153.0 s |
| judge | openai/gpt-oss-20b | 4 | 1,906 | 690 | 1.3 s |
| judge | qwen/qwen3.8-27b | 53 | 1,397 | 167 | 9.7 s |
| writer | openai/gpt-oss-120b | 2 | 2,978 | 4,098 | 9.2 s |
| writer | qwen/qwen3.8-27b | 1 | 3,045 | 699 | 1.8 s |

## Per show (50 episodes)

| stage | input tok | output tok | calls | basis |
|---|---|---|---|---|
| analyze | 616,390 | 129,939 | 50 | 50 calls for 50 episodes, x1.00 |
| judge | 90,689 | 11,244 | 64 | 64 calls, scaled to 64 finalists |
| writer | 3,139 | 930 | 1 | per show, as measured |

Total ~852,331 tokens per show (~17,047 per episode); ~170M tokens/month.

## Price per model choice (same token counts for every model)

| models | per episode | per show | per month |
|---|---|---|---|
| As configured, paid tiers (3.5 Flash-Lite analyze + Qwen3.8 27B judge + gpt-oss-120b writer) | $0.0126 | $0.63 | $126 |
| Quality preset (3.8 Flash analyze + Qwen3.8 27B judge + gpt-oss-120b writer) | $0.0214 | $1.07 | $214 |
| All Groq gpt-oss-120b (paid) | $0.0038 | $0.19 | $38 |
| All Groq gpt-oss-20b (paid) | $0.0019 | $0.10 | $19 |
| All Gemini 3.8 Flash (paid, 2026 price) | $0.0213 | $1.07 | $213 |
| All Gemini 3.8 Flash (paid, from 2027-01-01) | $0.0426 | $2.13 | $426 |
| Gemini 3.5 Flash-Lite analyze + gpt-oss-20b judge/writer | $0.0104 | $0.52 | $104 |
| All Claude Opus 5.5 | $0.1137 | $5.68 | $1,137 |
| All Claude Haiku 5.5 | $0.0028 | $0.14 | $28 |
| Claude Opus 5.5 analyze + Haiku 5.5 judge/writer | $0.1016 | $5.08 | $1,016 |

Caveats: token counts are the run's models'; other models tokenize differently and reasoning models bill thinking as output (Gemini output prices include thinking; the OpenAI-compatible endpoint may not report all thinking tokens, so Gemini rows are a floor). Groq and Gemini batch tiers are ~50% off; prompt caching of the fixed system prompts lowers input cost further. Free-tier Gemini data may be used to improve Google's products, so production scripts belong on a paid tier.

## Free tier: how many shows per day fit

| stage | model | shows/day | min per show at TPM cap | note |
|---|---|---|---|---|
| analyze | gemini-3.5-flash-lite | unknown |  | TODO: per-model free caps only in AI Studio |
| analyze | openai/gpt-oss-120b | 0.27 | 93 | if analysis falls back to Groq gpt-oss-120b (shares the writer's quota). a mean call (~14,927 tok) is above the 8,000 tokens/min cap: expect rate-limit retries and long waits |
| judge | qwen/qwen3.8-27b | 1.96 | 13 |  |
| writer | qwen/qwen3.8-27b | 49.15 | 1 |  |

The slowest stage sets the pace; Groq caps are per model, so the writer and an eval judge on the same model share one daily budget.

## Human time (ASSUMPTIONS, not measured)

- Today: watch 50 episodes x 2.1 min + 30 min picking = **~133 min per show** before a first promo is chosen; 200 shows = ~442 h/month.
- With HookCut: review a 20-item shortlist at 1.5 min each = **~30 min per show**; ~100 h/month.
- Not counted: rendering, posting, and the human's call on taste. The shortlist exists the day the script locks, before any video.
