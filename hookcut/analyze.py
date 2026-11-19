"""Stage 2: one LLM call per episode scores its notable beats for hook / cliffhanger power.

The system prompt (prompts/analyze_system.md) is identical for every episode so providers can cache it; the user
message is the episode itself. Each episode also carries the previous episode's last ~25 s (taken from the
script, not from the previous analysis), so all episodes run in parallel.
Code, not the model, checks every beat id and fills timings, model and prompt version.

Checking policy: the first reply that breaks a rule is sent back once with the exact problem (models usually fix
it). After that, fixable problems are corrected in code and recorded in `EpisodeAnalysis.warnings`, so a weaker
model or a free tier never loses a whole episode over one wrong beat id. Only unusable replies are re-asked again.
"""
from __future__ import annotations

import copy
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, get_args

from .llm.base import LLMClient
from .runlog import RunLog
from .schema import Episode, EpisodeAnalysis, EpisodeEdge, HookType, Pickup
from .textfmt import render_beats, tc

PROMPTS = Path(__file__).parent / "prompts"
PREV_TAIL_S = 25.0      # how much of the previous episode's ending we show for the pickup check
OPENING_S = 3.0         # the episode's opening hook must start in its first 3 s
ENDING_S = 20.0         # the episode's ending cliffhanger must start in its last 20 s
MAX_BEATS = 30
HOOK_TYPES = set(get_args(HookType))
PICKUP_STATUSES = {"immediate", "recap_then_advance", "delayed", "unrelated", "not_applicable"}

Progress = Callable[[str, float], None]


def load_prompt(name: str) -> tuple[str, str]:
    """Return (text, version); the version is a content hash, so any prompt edit shows up in outputs and cache."""
    text = (PROMPTS / f"{name}.md").read_text()
    return text, f"{name}@{hashlib.sha256(text.encode()).hexdigest()[:8]}"


# ------------------------------------------------------------------ prompt building
def prev_tail(prev: Optional[Episode], seconds: float = PREV_TAIL_S):
    if prev is None:
        return []
    return [b for b in prev.beats if b.end > prev.duration - seconds]


def episode_message(ep: Episode, prev: Optional[Episode]) -> str:
    lines = [f"EPISODE {ep.number} (duration {tc(ep.duration)}, {len(ep.beats)} beats, ids "
             f"{ep.beats[0].id}-{ep.beats[-1].id})" if ep.beats else f"EPISODE {ep.number} (no beats)",
             f"LOGLINE: {ep.logline or '(none)'}",
             f"CONTINUITY NOTES: {ep.continuity_notes or '(none)'}",
             "", "BEATS:", render_beats(ep.beats)]
    tail = prev_tail(prev)
    if tail:
        lines += ["", f"PREVIOUS EPISODE {prev.number}, LAST {PREV_TAIL_S:.0f} SECONDS (pickup check only; "
                      f"never score or cite these ids):", render_beats(tail)]
    else:
        lines += ["", "PREVIOUS EPISODE: not provided, so pickup.status must be not_applicable."]
    return "\n".join(lines)


# ------------------------------------------------------------------ checking the reply
def _fix_id(x, ep: Episode) -> Optional[str]:
    """Accept "B27" / "27" / "e01.b027" for the canonical "E01.B027"; leave anything else for the check."""
    if x is None or str(x).strip().lower() in ("", "null", "none"):
        return None
    s = str(x).strip()
    m = re.fullmatch(r"(?:E(\d+)\.)?B?(\d+)", s, re.I)
    if m and (m.group(1) is None or int(m.group(1)) == ep.number):
        return f"E{ep.number:02d}.B{int(m.group(2)):03d}"
    return s.upper()


def _hook_type(x) -> Optional[str]:
    t = str(x or "").strip().lower().replace(" ", "_")
    return None if t in ("", "none", "null") else (t if t in HOOK_TYPES else "other")


def _int(x, lo: int, hi: int) -> Optional[int]:
    try:
        return max(lo, min(hi, int(round(float(x)))))
    except (TypeError, ValueError):
        return None


def coerce_analysis(raw, ep: Episode, has_prev: bool, strict: bool = False) -> EpisodeAnalysis:
    """Turn the model's JSON into an EpisodeAnalysis.

    strict=True: raise ValueError listing every problem (the client re-asks with that message).
    strict=False: correct fixable problems in code and record them in .warnings; raise only if the reply is
    unusable (not an object, beats missing, or most beat ids invalid)."""
    if not isinstance(raw, dict):
        raise ValueError("reply must be one JSON object with summary, characters, beats, opening_hook, "
                         "ending_cliffhanger, ending_open_question, pickup")
    d = copy.deepcopy(raw)
    d.pop("episode", None)
    index = {b.id: i for i, b in enumerate(ep.beats)}
    first, last = (ep.beats[0].id, ep.beats[-1].id) if ep.beats else ("-", "-")
    problems: list[str] = []          # every rule broken (all of them are sent back in strict mode)
    warnings: list[str] = []          # what lenient mode corrected

    def bad_id(bid) -> Optional[str]:
        if bid in index:
            return None
        other = re.match(r"E(\d+)\.", str(bid))
        return (f"is from Episode {int(other.group(1))}, not this one" if other and int(other.group(1)) != ep.number
                else "does not exist")

    beats_in = d.get("beats")
    if not isinstance(beats_in, list):
        raise ValueError("beats must be a list of objects with beat_id, hook, cliff, ...")
    kept_beats, invalid = [], 0
    for k, b in enumerate(beats_in):
        if not isinstance(b, dict):
            invalid += 1
            problems.append(f"beats[{k}] must be an object")
            continue
        b["beat_id"] = _fix_id(b.get("beat_id"), ep)
        if b["beat_id"] is None or (why := bad_id(b["beat_id"])):
            invalid += 1
            problems.append(f"beats[{k}].beat_id {b.get('beat_id')!r} "
                            f"{'is missing' if b['beat_id'] is None else why}; use an id from {first} to {last}")
            continue
        b["hook_type"] = _hook_type(b.get("hook_type"))
        for key, hi in (("hook", 5), ("cliff", 5), ("spoiler", 3)):
            v = _int(b.get(key, 0), 0, hi)
            if v is None:
                problems.append(f"beats[{k}].{key} must be a number 0-{hi}")
                v = 0
            b[key] = v
        nc = b.get("needs_context")
        b["needs_context"] = ([nc] if isinstance(nc, str) and nc.strip() else
                              [str(x) for x in nc] if isinstance(nc, list) else [])
        rb = _fix_id(b.get("resolved_by"), ep)
        if rb is not None and (bad_id(rb) or index[rb] <= index[b["beat_id"]]):
            problems.append(f"beats[{k}] ({b['beat_id']}): resolved_by {rb} must be a LATER beat of this episode "
                            f"that answers the question, or null")
            warnings.append(f"dropped resolved_by {rb} on {b['beat_id']}")
            rb = None
        b["resolved_by"] = rb
        for key in ("open_question", "reason"):
            if b.get(key) is not None and not isinstance(b.get(key), str):
                b[key] = str(b[key])
        kept_beats.append(b)
    if beats_in and invalid > len(beats_in) / 2:
        raise ValueError("most beat ids are invalid: " + "; ".join(problems[:6]))
    if ep.beats and not kept_beats:
        raise ValueError("no valid beats were scored; score the episode's notable beats by their ids")
    if invalid:
        warnings.append(f"dropped {invalid} beat(s) with invalid ids")
    d["beats"] = kept_beats

    listed = {b["beat_id"] for b in kept_beats}
    for key in ("top_hooks", "top_cuts"):     # the shortlist makes the model compare before it scores
        ids = [_fix_id(x, ep) for x in (d.get(key) or []) if x is not None]
        missing = [bid for bid in ids if bid not in listed]
        if missing:
            problems.append(f"{key} has {missing} but beats has no entry for them; score them in beats")
        d.pop(key, None)

    scored = {b["beat_id"]: b for b in kept_beats}
    opening = [b for b in ep.beats if b.start < OPENING_S] or ep.beats[:1]
    ending = [b for b in ep.beats if b.start >= ep.duration - ENDING_S] or ep.beats[-5:]
    for key, window, field, where in (("opening_hook", opening, "hook", f"first {OPENING_S:.0f} s"),
                                      ("ending_cliffhanger", ending, "cliff", f"final {ENDING_S:.0f} s")):
        edge = d.get(key) if isinstance(d.get(key), dict) else {}
        edge["beat_id"] = _fix_id(edge.get("beat_id"), ep)
        edge["score"] = _int(edge.get("score", 0), 0, 5) or 0
        allowed = {b.id for b in window}
        if edge["beat_id"] is not None and edge["beat_id"] not in allowed:
            ids = [b.id for b in window]
            problems.append(f"{key}.beat_id {edge['beat_id']} is not in the {where}; choose one of "
                            f"{ids if len(ids) <= 6 else ids[0] + '..' + ids[-1]}")
            best = max((b for b in window if b.id in scored), key=lambda b: scored[b.id][field], default=None)
            if best is not None:
                edge["beat_id"], edge["score"] = best.id, scored[best.id][field]
            else:
                content = [b for b in window if b.kind != "camera"] or window
                edge["beat_id"] = (content[0] if key == "opening_hook" else content[-1]).id
            warnings.append(f"{key} moved into the {where} ({edge['beat_id']}, score {edge['score']})")
        d[key] = edge

    p = d.get("pickup") if isinstance(d.get("pickup"), dict) else {}
    p["first_new_beat_id"] = _fix_id(p.get("first_new_beat_id"), ep)
    p.pop("delay_s", None)
    if p.get("status") not in PICKUP_STATUSES:
        problems.append(f"pickup.status {p.get('status')!r} must be one of {sorted(PICKUP_STATUSES)}")
        p["status"] = "unrelated" if has_prev else "not_applicable"
        warnings.append(f"pickup.status set to {p['status']}")
    if not has_prev:
        p.update(status="not_applicable", first_new_beat_id=None)
    else:
        if p["status"] == "not_applicable":
            problems.append("pickup.status is not_applicable but the previous episode's ending was given; use "
                            "immediate, recap_then_advance, delayed or unrelated")
            p["status"] = "unrelated"
            warnings.append("pickup.status not_applicable -> unrelated")
        if p["first_new_beat_id"] is not None and bad_id(p["first_new_beat_id"]):
            problems.append(f"pickup.first_new_beat_id {p['first_new_beat_id']} {bad_id(p['first_new_beat_id'])}")
            warnings.append(f"dropped pickup.first_new_beat_id {p['first_new_beat_id']}")
            p["first_new_beat_id"] = None
        if p["status"] in ("immediate", "recap_then_advance", "delayed") and not p["first_new_beat_id"]:
            problems.append(f"pickup.first_new_beat_id is required when status is {p['status']}")
            warnings.append("pickup has no first_new_beat_id")
    d["pickup"] = p
    for key in ("summary", "ending_open_question"):
        d[key] = str(d.get(key) or "")
    if not isinstance(d.get("characters"), list):
        d["characters"] = []

    if strict and problems:
        raise ValueError("; ".join(problems))
    try:
        a = EpisodeAnalysis.model_validate({**d, "episode": ep.number})
    except Exception:
        d["characters"] = []                       # the least important field is the usual culprit
        a = EpisodeAnalysis.model_validate({**d, "episode": ep.number})
        warnings.append("dropped malformed characters list")

    strength = lambda s: max(s.hook, s.cliff)
    best = {}                             # notable beats only (hook or cliff >= 2), one entry per beat id
    for s in a.beats:
        if strength(s) >= 2 and (s.beat_id not in best or strength(s) > strength(best[s.beat_id])):
            best[s.beat_id] = s
    kept = sorted(best.values(), key=lambda s: (-strength(s), index[s.beat_id]))[:MAX_BEATS]
    a.beats = sorted(kept, key=lambda s: index[s.beat_id])
    if a.pickup.first_new_beat_id:
        a.pickup.delay_s = ep.beat(a.pickup.first_new_beat_id).start
    a.warnings = warnings
    return a


def _span(ids: list[str]) -> str:
    return f"{ids[0]}..{ids[-1]}" if len(ids) > 6 else ", ".join(ids)


def failed_analysis(ep: Episode, error: str, model: str = "", version: str = "") -> EpisodeAnalysis:
    """Placeholder that keeps the run going; report.py spots it by the "analysis failed:" summary and the
    "failed:<models tried>" model field."""
    return EpisodeAnalysis(episode=ep.number, summary=f"analysis failed: {error[:300]}", beats=[],
                           opening_hook=EpisodeEdge(), ending_cliffhanger=EpisodeEdge(),
                           pickup=Pickup(status="not_applicable", note="analysis failed"),
                           model=model, prompt_version=version)


# ------------------------------------------------------------------ the stage
def analyze_episode(ep: Episode, prev: Optional[Episode], client: LLMClient, *, log: RunLog) -> EpisodeAnalysis:
    system, version = load_prompt("analyze_system")
    has_prev = bool(prev_tail(prev))
    item = f"E{ep.number:02d}"
    seen = {"n": 0, "first": None}

    def validate(data) -> None:          # first reply: strict (send problems back once); later: fix in code
        seen["n"] += 1
        if seen["n"] == 1:
            seen["first"] = data
        coerce_analysis(data, ep, has_prev, strict=seen["n"] == 1)

    try:
        res = client.complete_json(system, episode_message(ep, prev), validate=validate, cache_tag=version)
        a = coerce_analysis(res.data, ep, has_prev)   # again here: cached replies skip the validate callback
    except Exception as e:
        # Re-asks can make things worse (a weak model may return only the field it was told to fix). If the
        # first reply is usable with code fixes, keep it rather than losing the episode.
        if seen["first"] is not None:
            try:
                a = coerce_analysis(seen["first"], ep, has_prev)
                first_spec = getattr(client, "clients", [client])[0].spec
                a.warnings.append("re-asks failed; kept the first reply with code fixes")
                a.model, a.prompt_version = first_spec, version
                log.record("analyze", item, error=f"recovered first reply after: {type(e).__name__}: {e}",
                           recovered=True, model=first_spec)
                return a
            except Exception:
                pass
        log.record("analyze", item, error=f"{type(e).__name__}: {e}")   # nothing usable: placeholder
        return failed_analysis(ep, f"{type(e).__name__}: {e}", f"failed:{client.spec}", version)
    log.record("analyze", item, res, warnings=a.warnings)
    a.model, a.prompt_version = f"{res.provider}:{res.model}", version   # the model that answered, not the chain
    return a


def analyze_show(episodes: list[Episode], client: LLMClient, *, log: RunLog, concurrency: int = 4,
                 progress: Optional[Progress] = None) -> list[EpisodeAnalysis]:
    """One call per episode, `concurrency` at a time; results come back in episode order."""
    by_number = {e.number: e for e in episodes}
    out: dict[int, EpisodeAnalysis] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(analyze_episode, e, by_number.get(e.number - 1), client, log=log): i
                   for i, e in enumerate(episodes)}
        for done, f in enumerate(as_completed(futures), 1):
            i = futures[f]
            out[i] = f.result()
            if progress:
                a = out[i]
                status = "failed" if a.summary.startswith("analysis failed") else f"{len(a.beats)} notable beats"
                progress(f"analyzed E{a.episode:02d} ({status}) {done}/{len(episodes)}", done / len(episodes))
    return [out[i] for i in range(len(episodes))]
