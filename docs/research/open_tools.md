# HookCut: free LLMs and open tools (checked 2026-11-19)

## 1. Free-tier LLMs

| Provider | Free models to use | Free limits | JSON |
|---|---|---|---|
| **Gemini API** | `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-4-hydrogen` (stable, released 2026-11-12), `gemini-3.1-flash-lite`, Gemma 4 [1][2] | Per-model limits shown only in AI Studio; per project, not per key [3] | JSON-Schema constrained decoding. `minLength`/`uniqueItems` are stripped. Over-complex schemas return 400 [4] |
| **Groq** | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` | Each model: 30 RPM, 1K RPD, **8K TPM**, 200K TPD [5] | `strict: true` (constrained) on all three. All fields must be required, `additionalProperties:false`, no streaming [6] |
| **OpenRouter** | 14 `:free` models today (e.g. Gemma 4, Nemotron 3.5) [7] | 20 RPM, **50 RPD** (1,000 after $10 of credits) [8] | Varies by model. Gemma 4 free supports `json_object` only |
| **HF Inference Providers** | none | Free users get **no** monthly credits (PRO: $2) [9] | n/a |

Key facts:
- **Gemini OpenAI-compatible endpoint:** `https://generativelanguage.googleapis.com/v1beta/openai/` (beta); supports `beta.chat.completions.parse` with pydantic. Thinking can't be disabled on Gemini 3.x/4 [10].
- **Gemini caveats:** free-tier data is "used to improve our products" [1]: keep confidential scripts off the free tier in production. `gemini-2.5-flash`/`-pro` start rerouting on **2026-12-01** [11], so don't pin 2.5.
- **Qwen3.8-27B on Groq:** Preview; 131K context, 16,384 max output, `reasoning_effort` defaults to `none`, accepts images [12]. Llama 3.3 70B is now Enterprise-only [13].
- **Groq budget:** with 4k-token prompts, 8K TPM ≈ 1 call/min and 200K TPD ≈ 35 episodes/day, per model.
- **Reliable valid JSON at 3–4k tokens:** Groq strict (gpt-oss-120b, qwen3.8-27b) and Gemini structured output: constrained, so syntax is guaranteed; still validate semantics. OpenRouter free: fallback only.
- **Gemini Decision 1:** free tier is 200 RPM, 50K TPM, 10K RPD [3]. It returns calibrated yes/no, choice or rubric scores with zero output tokens [14]. It fits the cold-viewer check, but only on Gemini.
- **Scale (paid):** 10k episodes/month at about 5k input + 1k output tokens each costs roughly $7 on gpt-oss-20b, $14 on gpt-oss-120b, $40 on 3.5-flash-lite (plus thinking tokens) and $75 on 3.8-flash (doubles 2027-01-01) [1][13].

**Pick:** one `openai` client with provider/base_url/model set in config. Beat scoring runs on `gemini-3.8-flash`. The cold-viewer pass runs on Groq `qwen/qwen3.8-27b` (strict): a different family judges more independently, on separate quota. `gpt-oss-120b` is the fallback.

## 2. Building blocks (Streamlit Cloud: 690 MB–2.7 GB RAM [15])

| Tool | Verdict | Reason | License | Size / CPU |
|---|---|---|---|---|
| `openai` SDK 3.55 + pydantic + own 2-retry loop | **USE NOW** | Works with Gemini, Groq, OpenRouter and HF through `base_url`. `.parse()` emits strict schemas | Apache-2.0 | tiny |
| json-repair 0.64 | **USE NOW** | Salvages JSON from best-effort providers | MIT | tiny |
| model2vec `potion-base-8M` [16] | **USE NOW** | Dedup embeddings, numpy only, up to 500× faster than its teacher model [17] | MIT | 30 MB, CPU-trivial |
| `potion-multilingual-128M` | **USE NOW if scripts are Hindi/Hinglish** | 101 languages including `hi` | MIT | 512 MB, local only |
| OpenTimelineIO 0.18.1 + `otio-cmx3600-adapter` + `otio-fcp-adapter` [18] | **USE NOW** | Writes `.otio` (Resolve 18.5+ [19]), FCP7 XML (Premiere/Resolve), CMX3600 EDL (reel names ≤8 chars: use `E01`). Wheels cp39–cp313: pin Python ≤3.13 | Apache-2.0 | ~1 MB |
| instructor 1.17 | LATER | Good, but redundant once strict schemas work | MIT | small |
| fastembed (bge-small) | LATER | Better English embeddings without torch | Apache-2.0 | 134 MB |
| `SamLowe/roberta-base-go_emotions-onnx` int8 [20] | LATER | 28-label pre-filter at scale. F1 0.45. Trained on Reddit (domain gap) | MIT | 125 MB, onnxruntime |
| `j-hartmann/emotion-english-distilroberta-base` [21] | LATER | 7 Ekman labels, 66% eval accuracy, English only. Card has no license tag | unstated | 329 MB, torch |
| Groq `whisper-large-v3-turbo` | **LATER (first pick for video)** | Word timestamps via `verbose_json`. Free tier: 28.8K audio-seconds/day (about 240 two-minute episodes), 25 MB per file [5][22] | API | none |
| PySceneDetect 0.7.3 | LATER | Shot cuts, active (commit 2026-11-18) | BSD-3 | OpenCV, CPU-fine |
| TransNetV2 (`transnetv2-pytorch`) | LATER | More accurate shot boundaries. Repo idle since 2021 | MIT | 33 MB + torch |
| WhisperX 3.8.6 | LATER | Local aligned words plus pyannote diarization. Heavy, slow on CPU, gated HF model | BSD-2 | torch |
| Lighthouse (LINE) [23] | LATER (research) | Moment-DETR, CG-DETR etc. behind one API; CPU OK with `clip` features; tested on Py3.9/torch 2.1. Weights trained on QVHighlights (**CC BY-NC-SA** [24]) vlogs/news, not drama | Apache-2.0 code | torch |
| `all-MiniLM-L6-v2`, `gte-small`, `bge-small` via sentence-transformers | AVOID on Cloud | torch is too heavy for small gains over potion | Apache/MIT | 91/67/134 MB |
| LiteLLM 1.110 | AVOID | Heavy. PyPI 1.82.7/1.82.8 shipped a credential stealer in March 2026 [25] | MIT | large |
| pydantic-ai 2.71 | AVOID | Agent framework with high churn, needs Py ≥3.11 | MIT | medium |
| outlines 1.3.3 | AVOID | Constrained decoding for local weights, not hosted APIs | Apache-2.0 | needs weights |
| UniVTG / CG-DETR / Moment-DETR raw repos | AVOID | Stale (last commits 2024-05 / 2024-08). Use through Lighthouse if at all | MIT code | GPU-oriented |
| ClipsAI 0.2.1 | AVOID | Dead since 2024-01, built for podcasts, pulls in WhisperX + pyannote | MIT | heavy |

**CapCut desktop:** no official import docs; third-party posts claim XML and "EDL (Experimental)" import [26] (unverified). Also ship CSV + ffmpeg-cut MP4s.

Last commits (2026): model2vec 11-11, instructor 11-18, OTIO 11-10, WhisperX 11-12, Lighthouse 11-01.

## Sources
[1]: https://ai.google.dev/gemini-api/docs/pricing
[2]: https://ai.google.dev/gemini-api/docs/models/gemini-4-hydrogen
[3]: https://ai.google.dev/gemini-api/docs/rate-limits
[4]: https://ai.google.dev/gemini-api/docs/structured-output
[5]: https://console.groq.com/docs/rate-limits
[6]: https://console.groq.com/docs/structured-outputs
[7]: https://openrouter.ai/api/v1/models
[8]: https://openrouter.ai/docs/api/reference/limits
[9]: https://huggingface.co/docs/inference-providers/pricing
[10]: https://ai.google.dev/gemini-api/docs/openai
[11]: https://ai.google.dev/gemini-api/docs/deprecations
[12]: https://console.groq.com/docs/model/qwen/qwen3.8-27b
[13]: https://console.groq.com/docs/models
[14]: https://ai.google.dev/gemini-api/docs/decisions-api
[15]: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
[16]: https://huggingface.co/minishlab/potion-base-8M
[17]: https://github.com/MinishLab/model2vec
[18]: https://opentimelineio.readthedocs.io/en/latest/tutorials/adapters.html
[19]: https://www.steakunderwater.com/VFXPedia/__man/Resolve18-6/DaVinciResolve18_Manual_files/part1410.htm
[20]: https://huggingface.co/SamLowe/roberta-base-go_emotions
[21]: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
[22]: https://console.groq.com/docs/speech-to-text
[23]: https://github.com/line/lighthouse
[24]: https://github.com/jayleicn/moment_detr
[25]: https://www.hiddenlayer.com/research/litellm-supply-chain-attack
[26]: https://cutsio.com/blog/how-to-export-screenstudio-to-capcut-with-jump-cut

## Prices for cost model (checked 2026-11-19)

```yaml
# Paid Standard tier, USD per 1M tokens. Sources: ai.google.dev/gemini-api/docs/pricing,
# console.groq.com/docs/model/<id>, console.groq.com/docs/batch, console.groq.com/docs/prompt-caching
prices_usd_per_mtok:
  gemini-3.8-flash: {input: 0.75, output: 3.75, note: "Price through 2026-12-31; from 2027-01-01 input 1.50, output 7.50. Output price includes thinking tokens. Batch/Flex 0.375/1.875 (0.75/3.75 from 2027); Priority 1.35/6.75 (2.70/13.50 from 2027). Cached input 0.075 (0.15 from 2027). No long-context surcharge listed."}
  gemini-4-hydrogen: {input: 1.50, output: 7.50, note: "Output includes thinking tokens (thinking cannot be turned off on Gemini 4). Batch/Flex 0.75/3.75; Priority 2.70/13.50. Implicit cache 0.075. 1M context, no long-context surcharge listed, no 2027 change listed."}
  gemini-3.5-flash-lite: {input: 0.30, output: 2.50, note: "Same input price for text/image/video/audio. Output includes thinking tokens. Batch/Flex 0.15/1.25; Priority 0.54/4.50; cached 0.03. No long-context surcharge, no 2027 change listed."}
  openai/gpt-oss-120b: {input: 0.15, output: 0.60, note: "Cached input 0.075 (automatic prompt caching). Batch API 50% off, does not stack with caching. 131K context, no long-context surcharge listed. Reasoning tokens are emitted as output; billing them at output rate is assumed, not stated."}
  openai/gpt-oss-20b: {input: 0.075, output: 0.30, note: "Cached input 0.037. Batch API 50% off, does not stack with caching. 131K context, no long-context surcharge listed. Reasoning tokens assumed billed as output."}
  qwen/qwen3.8-27b: {input: 0.80, output: 4.00, note: "Preview model. No cached-input price and not on the Batch API model list. 131K context, 16,384 max output. reasoning_effort defaults to none."}
data_use_paid:
  gemini: "Terms (ai.google.dev/gemini-api/terms): on Paid Services Google does not use prompts (incl. system instructions, cached content, files) or responses to improve its products, and processes them under its Data Processing Addendum. Prompts and responses are logged for a limited period only for Prohibited Use Policy enforcement and legal disclosures. The API counts as paid only via a Cloud project with an active billing account. Free/unpaid quota IS used to improve products and human reviewers may read it. The invite-only discounted Contributor tier also uses data."
  groq: "Services Agreement s4.2 (console.groq.com/docs/legal/services-agreement): Groq is not permitted to use Inputs or Outputs for training or fine-tuning any models unless the customer explicitly permits or instructs it. Your Data page: inference data not retained by default; reliability/abuse logs kept up to 30 days; Zero Data Retention can be enabled by all customers in Data Controls; batch files kept 30 days."
```
