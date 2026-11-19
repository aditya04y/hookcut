# HookCut prompts

Three prompts, one per LLM stage. Each file's content hash is its version (`analyze_system@<8 hex chars>`), stored on
every analysis and in the cache key, so an output always says which prompt made it.

| File | Stage | Calls | Sees |
|---|---|---|---|
| `analyze_system.md` | analyze | 1 per episode | the whole episode + the previous episode's last 25 s |
| `judge_system.md` | cold-viewer test | 1 per finalist | only the clip's beats |
| `writer_system.md` | why-it-works lines | 1 per run | facts already chosen upstream |

## Design choices

**The model scores beats; code makes the clips.** The analyzer never proposes start/end times. It marks where a
clip could open (hook) and cut (cliff); candidates.py builds the 20–60 s windows. Clip rules (length, hook
inside the first 3 s) then hold by construction, and every number is traceable to a beat id.

**Calibrated 0–5 anchors with generic examples.** Each score level gets a concrete description ("a door opens on
someone we cannot see" = cliff 5). Examples are invented, never taken from the client's script, so the prompt
does not leak confidential text and does not overfit to one show.

**Legibility rule for hooks.** Small models pick the first emotional line (a whispered name, "Why?") as the hook.
The prompt says to score the line or action that delivers the blow, not the question that sets it up or the
reaction after it, and that a 4–5 must make sense to a stranger from those 3 seconds alone.

**Read the episode as exchanges.** A question or plea, then the answer or blow: the answer is usually the hook,
the question just before it the cut point (a clip that stops there withholds the answer). Silence counts as an
answer. This one generic rule gives hooks and cliffs a shared structure instead of two unrelated lists.

**Shortlist before scoring.** The reply starts with `top_hooks` and `top_cuts` (3 ids each, best first), so the
model compares the episode's strongest moments with each other before it gives absolute scores. Code checks
that every shortlisted beat is also scored.

**Cliff = "cut to black right after this beat".** `resolved_by` records the later beat in the same episode that
answers the question, because a clip that runs past the answer has spent its cliffhanger. Cut points must end
a thought (no mid-line or mid-rant cuts); music swells and cut-to-black are not cliffs.

**Visual hooks count.** CAMERA and ACTION lines are sent (camera lines truncated to 140 chars) because a slap or
a door opening is often only in those lines.

**Stable system prompt, episode in the user message.** The system prompt is byte-identical for every episode, so
providers that cache prompt prefixes can reuse it. The previous episode's tail comes from the script, not from
its analysis, so all episodes run in parallel.

**Code checks the model.** Every beat id, `resolved_by` ordering, the opening beat (first 3 s), the ending beat
(last 20 s) and the pickup fields are checked in code; a failure is re-asked with the exact error ("E01.B099
does not exist; use an id from E01.B001 to E01.B095"). Code fills `delay_s`, model and prompt version, keeps
only notable beats (hook or cliff >= 2, at most 30) and drops duplicates.

**Compact output.** Only notable beats, reasons of 12 words or fewer. Output tokens dominate latency and free
tier rate limits (Groq: 8K tokens/minute).

**Cold viewer sees only the clip.** No logline, summary or character list; times restart at 00:00.00; segments
are separated by `--- CUT ---`; speaker labels stay (a viewer sees faces) but the prompt says you only know a
name if someone says it. The judge runs on a different model family from the analyzer so it does not share its
blind spots. It returns understanding first (premise, conflict, question, confusions), then scores; the
pass/borderline/fail rule is computed in code (`judge.verdict_for`) so it is visible and tunable.

**Writer is grounded.** One call for the whole list, given only the hook line, cliffhanger line, open question and
what the cold viewer understood. Missing ids or lines over 25 words are re-asked; if the call still fails, a
deterministic template line is used, so a run never ends without "why it works" lines.
