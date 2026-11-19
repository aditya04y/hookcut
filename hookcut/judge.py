"""Stage 4: the cold-viewer test. A model plays a US stranger scrolling TikTok and sees ONLY the clip.

No logline, continuity notes, summaries or character lists: just the beats that play, with times relative to
the clip's first frame and "--- CUT ---" between segments. Speaker tags stay (a viewer sees faces).
The pass / borderline / fail verdict is computed here from the model's scores, so the rule is visible.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .analyze import load_prompt
from .llm.base import LLMClient
from .runlog import RunLog
from .schema import Candidate, ColdViewerVerdict, Episode
from .textfmt import render_beats, tc

Progress = Callable[[str, float], None]
CUT = "--- CUT ---"


def verdict_for(hook_lands: bool, standalone: int, keep_watching: int) -> str:
    if standalone <= 2 or keep_watching <= 2:
        return "fail"
    if hook_lands and standalone >= 4 and keep_watching >= 4:
        return "pass"
    return "borderline"


def clip_text(cand: Candidate, episodes: dict[int, Episode]) -> str:
    """What plays, in order. Beats are clipped to the segment: a shot that started just before the cut-in is
    still on screen (shown from 00:00.00), and nothing runs past the cut."""
    parts, elapsed = [], 0.0
    for seg in cand.segments:
        ep = episodes[seg.episode]
        beats = [b.model_copy(update={"start": max(b.start, seg.start), "end": min(b.end, seg.end)})
                 for b in ep.beats if b.start < seg.end - 0.05 and b.end > seg.start + 0.3]
        offset = seg.start - elapsed          # clip time = episode time - offset
        if parts:
            parts.append(CUT)
        parts.append(render_beats(beats, offset=offset, with_id=False))
        elapsed += seg.end - seg.start
    return f"CLIP ({tc(elapsed)} long):\n" + "\n".join(parts)


def _coerce(raw, cand: Candidate, model: str) -> ColdViewerVerdict:
    if not isinstance(raw, dict):
        raise ValueError("reply must be one JSON object with premise, conflict, open_question, confusions, "
                         "hook_lands, standalone, keep_watching")
    d = {k: raw.get(k) for k in ("premise", "conflict", "open_question", "confusions", "hook_lands",
                                 "standalone", "keep_watching")}
    missing = [k for k in ("premise", "hook_lands", "standalone", "keep_watching") if d[k] is None]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    d["conflict"] = d["conflict"] or ""
    d["open_question"] = d["open_question"] or ""
    if isinstance(d["confusions"], str):
        d["confusions"] = [d["confusions"]] if d["confusions"].strip() else []
    d["confusions"] = d["confusions"] or []
    v = ColdViewerVerdict.model_validate({**d, "candidate_id": cand.id, "verdict": "fail", "model": model})
    v.verdict = verdict_for(v.hook_lands, v.standalone, v.keep_watching)
    return v


def judge_one(cand: Candidate, episodes: dict[int, Episode], client: LLMClient, *,
              log: RunLog) -> Optional[ColdViewerVerdict]:
    system, version = load_prompt("judge_system")
    try:
        res = client.complete_json(system, clip_text(cand, episodes),
                                   validate=lambda d: _coerce(d, cand, client.spec), cache_tag=version)
        v = _coerce(res.data, cand, f"{res.provider}:{res.model}")   # the model that answered, not the chain
    except Exception as e:
        log.record("judge", cand.id, error=f"{type(e).__name__}: {e}")
        return None
    log.record("judge", cand.id, res, verdict=v.verdict)
    return v


def select_finalists(candidates: list[Candidate], top_k: int) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda c: -c.prelim_score)
    best_per_episode, seen = [], set()
    for c in ranked:
        ep = c.segments[0].episode
        if ep not in seen:
            seen.add(ep)
            best_per_episode.append(c)
    picked = best_per_episode[:top_k]
    chosen = {c.id for c in picked}
    picked += [c for c in ranked if c.id not in chosen][: max(0, top_k - len(picked))]
    return sorted(picked, key=lambda c: -c.prelim_score)


def judge_candidates(candidates: list[Candidate], episodes: list[Episode], client: LLMClient, *, top_k: int,
                     log: RunLog, concurrency: int = 2, progress: Optional[Progress] = None) -> list[ColdViewerVerdict]:
    """Judge up to top_k finalists: first the best candidate of every episode (so no episode goes untested just
    because another episode's scores run high), then the rest by prelim_score. Failed calls are skipped."""
    finalists = select_finalists(candidates, top_k)
    by_number = {e.number: e for e in episodes}
    out: dict[int, Optional[ColdViewerVerdict]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(judge_one, c, by_number, client, log=log): i for i, c in enumerate(finalists)}
        for done, f in enumerate(as_completed(futures), 1):
            out[futures[f]] = f.result()
            if progress:
                progress(f"cold-viewer test {done}/{len(finalists)}", done / len(finalists))
    return [out[i] for i in range(len(finalists)) if out[i] is not None]
