"""Build promo candidate clips from scored beats. Plain code, no model.

A candidate opens on a hook beat and closes right after a cliffhanger beat. The cut rules come straight
from the brief plus what an editor needs:
  * the hook beat starts within the first (hook_window_s - 1) s, so it plays inside the first 3 s;
  * the clip starts on the camera shot that contains the hook, but never more than LEAD_MAX_S before it;
  * the clip ends on the cliff beat (optionally run to the end of its shot, never into the answer);
  * no cut lands inside a spoken line (dialogue / V.O.) at either end: snap to the line or drop the clip;
  * the cliffhanger's answer (resolved_by) must not play inside the clip;
  * total length within [min_s, max_s].
Multi-segment candidates join episode N's ending cliffhanger to episode N+1 after its recap.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .schema import Beat, BeatScore, Candidate, Episode, EpisodeAnalysis, Segment

TOL = 0.05                # timing tolerance (s); screenplay times have 0.1 s resolution
LEAD_MAX_S = 1.0          # the clip may open at most this long before the hook beat (start of its shot)
SHOT_EXTEND_MAX_S = 1.5   # the clip may run past the cliff beat to the end of its shot by at most this
MULTI_MIN_SEGMENT_S = 3.0  # each joined segment must be long enough to register
SPEECH = ("dialogue", "vo")

# Preliminary score = sum(weight * feature). Hook and cliff carry the brief's two hard requirements on the
# same 0-5 scale; length fit is worth 1.5 hook points; penalties are small and only break near-ties.
PRELIM_WEIGHTS = {
    "hook": 1.0,             # per hook point (0-5) of the opening beat
    "cliff": 1.0,            # per cliff point (0-5) of the closing beat
    "duration_fit": 1.5,     # 1.0 inside sweet_spot, linear down to 0 at min_s / max_s
    "early_hook": 0.5,       # the hook beat starts within the clip's first second
    "needs_context": -0.4,   # per thing a newcomer must already know (capped at 3 items)
    "spoiler": -0.25,        # per spoiler point (0-3) of the worst beat inside the clip
    "multi": -0.5,           # joined segments: continuity risk, harder to follow cold
}
NEEDS_CONTEXT_CAP = 3


# ------------------------------------------------------------------ small geometry helpers
def score_index(analyses: Iterable[EpisodeAnalysis]) -> dict[str, BeatScore]:
    return {s.beat_id: s for a in analyses if a is not None for s in a.beats}


def beat_index(episodes: Iterable[Episode]) -> dict[str, Beat]:
    return {b.id: b for e in episodes for b in e.beats}


def hook_of(idx: dict[str, BeatScore], beat_id: str) -> int:
    s = idx.get(beat_id)
    return s.hook if s else 0


def cliff_of(idx: dict[str, BeatScore], beat_id: str) -> int:
    s = idx.get(beat_id)
    return s.cliff if s else 0


def speech_straddling(ep: Episode, t: float) -> Optional[Beat]:
    """The spoken line a cut at time t would chop, if any."""
    for b in ep.beats:
        if b.kind in SPEECH and b.start < t - TOL and b.end > t + TOL:
            return b
    return None


def shot_at(ep: Episode, t: float) -> Optional[Beat]:
    """The camera shot playing at time t (the latest-starting one if shots overlap)."""
    hits = [b for b in ep.beats if b.kind == "camera" and b.start <= t + TOL and b.end > t + TOL]
    return max(hits, key=lambda b: b.start) if hits else None


def first_beat_from(ep: Episode, t: float) -> Beat:
    """The first beat that starts at or after a cut-in point."""
    for b in ep.beats:
        if b.start >= t - TOL:
            return b
    return ep.beats[-1]


def open_at(ep: Episode, beat: Beat, max_lead: float) -> Optional[tuple[float, str, list[str]]]:
    """Cut-in point for a clip that should open on `beat`: (start, first_beat_id, notes) or None.
    Start of the beat's camera shot if that is at most LEAD_MAX_S earlier, else the beat itself; then
    never inside a spoken line (move back to the line's start if the beat stays within max_lead, or
    forward to its end if it finishes before the beat; otherwise give up)."""
    notes: list[str] = []
    shot = shot_at(ep, beat.start)
    start = shot.start if shot and beat.start - shot.start <= LEAD_MAX_S + TOL else beat.start
    line = speech_straddling(ep, start)
    if line:
        if beat.start - line.start <= max_lead + TOL:
            start = line.start
            notes.append(f"opening moved back {beat.start - start:.1f} s to the start of a line")
        elif line.end <= beat.start + TOL:
            start = line.end
        else:
            return None
        if speech_straddling(ep, start):
            return None
    return round(start, 3), first_beat_from(ep, start).id, notes


def close_at(ep: Episode, beat: Beat, resolver: Optional[Beat], extend: bool = True
             ) -> Optional[tuple[float, str, list[str]]]:
    """Cut-out point right after `beat`: (end, last_beat_id, notes) or None. Snaps to the end of a line
    the beat overlaps (≤ SHOT_EXTEND_MAX_S), and if `extend`, runs to the end of the camera shot when
    that adds ≤ SHOT_EXTEND_MAX_S, starts no new line and stops short of the resolving beat."""
    limit = resolver.start + TOL if resolver and resolver.episode == ep.number else float("inf")
    end, last, notes = beat.end, beat.id, []
    line = speech_straddling(ep, end)
    if line:
        if line.end - beat.end > SHOT_EXTEND_MAX_S or line.end > limit:
            return None
        end, last = line.end, line.id
        notes.append(f"end snapped to the end of a line (+{line.end - beat.end:.1f} s)")
    if extend:
        shot = next((b for b in ep.beats if b.kind == "camera" and b.start < end - TOL and b.end > end + TOL), None)
        if (shot and shot.end - beat.end <= SHOT_EXTEND_MAX_S and shot.end <= limit
                and not any(b.kind in SPEECH and end - TOL < b.start < shot.end - TOL for b in ep.beats)
                and not speech_straddling(ep, shot.end)):
            notes.append(f"end runs to the end of the shot (+{shot.end - end:.1f} s)")
            end, last = shot.end, shot.id
    return round(end, 3), last, notes


def answer_inside(resolver: Optional[Beat], segments: list[Segment]) -> bool:
    return bool(resolver) and any(
        s.episode == resolver.episode and s.start - TOL <= resolver.start < s.end - TOL for s in segments)


def still_frame_time(ep: Episode, hook: Beat) -> float:
    """A still for static-image ads: hook start + 0.5 s, kept inside the camera shot covering the hook."""
    t = hook.start + 0.5
    shot = shot_at(ep, hook.start)
    lo, hi = (shot.start, shot.end) if shot else (hook.start, hook.end)
    return round(max(lo, min(t, hi - 1 / 30)), 3)


def duration_fit(d: float, cfg: dict) -> float:
    mn, mx = cfg.get("min_s", 20), cfg.get("max_s", 60)
    lo, hi = cfg.get("sweet_spot", [25, 45])
    if lo <= d <= hi:
        return 1.0
    if d < lo:
        return max(0.0, min(1.0, (d - mn) / (lo - mn))) if lo > mn else 0.0
    return max(0.0, min(1.0, (mx - d) / (mx - hi))) if mx > hi else 0.0


def seg_length(segments: list[Segment]) -> float:
    return round(sum(s.end - s.start for s in segments), 3)


def time_overlap(a: list[Segment], b: list[Segment]) -> float:
    return sum(max(0.0, min(x.end, y.end) - max(x.start, y.start))
               for x in a for y in b if x.episode == y.episode)


def time_iou(a: list[Segment], b: list[Segment]) -> float:
    inter = time_overlap(a, b)
    union = seg_length(a) + seg_length(b) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(items: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


# ------------------------------------------------------------------------- building
def _candidate(kind: str, segments: list[Segment], hook: Beat, cliff: Beat, idx: dict[str, BeatScore],
               episodes: dict[int, Episode], cfg: dict, notes: list[str]) -> Candidate:
    dur = seg_length(segments)
    hs, cs = idx.get(hook.id), idx.get(cliff.id)
    inside = [b for s in segments for b in episodes[s.episode].beats if s.start - TOL <= b.start < s.end - TOL]
    spoiler = max([idx[b.id].spoiler for b in inside if b.id in idx] + [0])
    needs = _dedupe((hs.needs_context if hs else []) + (cs.needs_context if cs else []))
    lead = hook.start - segments[0].start
    w = PRELIM_WEIGHTS
    prelim = (w["hook"] * hook_of(idx, hook.id) + w["cliff"] * cliff_of(idx, cliff.id)
              + w["duration_fit"] * duration_fit(dur, cfg) + (w["early_hook"] if lead <= 1.0 + TOL else 0.0)
              + w["needs_context"] * min(len(needs), NEEDS_CONTEXT_CAP) + w["spoiler"] * spoiler
              + (w["multi"] if kind == "multi" else 0.0))
    return Candidate(
        id="", kind=kind, segments=segments, duration=dur, hook_beat_id=hook.id, cliff_beat_id=cliff.id,
        hook_score=float(hook_of(idx, hook.id)), cliff_score=float(cliff_of(idx, cliff.id)),
        hook_type=hs.hook_type if hs else None, open_question=(cs.open_question or "") if cs else "",
        spoiler=spoiler, needs_context=needs, prelim_score=round(prelim, 3), notes=notes)


def _resolver(idx: dict[str, BeatScore], beats: dict[str, Beat], beat_id: str) -> Optional[Beat]:
    s = idx.get(beat_id)
    return beats.get(s.resolved_by) if s and s.resolved_by else None


def _in_bounds(d: float, cfg: dict) -> bool:
    return cfg.get("min_s", 20) <= d <= cfg.get("max_s", 60)


def single_candidates(ep: Episode, idx: dict[str, BeatScore], beats: dict[str, Beat], cfg: dict) -> list[Candidate]:
    max_lead = cfg.get("hook_window_s", 3.0) - 1.0
    hooks = [b for b in ep.beats if hook_of(idx, b.id) >= cfg.get("hook_min", 3)]
    cliffs = [b for b in ep.beats if cliff_of(idx, b.id) >= cfg.get("cliff_min", 3)]
    ends = {k.id: close_at(ep, k, _resolver(idx, beats, k.id)) for k in cliffs}
    out = []
    for h in hooks:
        opening = open_at(ep, h, max_lead)
        if not opening:
            continue
        start, first_id, open_notes = opening
        for k in cliffs:
            if k.start <= h.start + TOL or not ends[k.id]:
                continue
            end, last_id, end_notes = ends[k.id]
            seg = [Segment(episode=ep.number, start=start, end=end, first_beat_id=first_id, last_beat_id=last_id)]
            if not _in_bounds(seg_length(seg), cfg) or answer_inside(_resolver(idx, beats, k.id), seg):
                continue
            out.append(_candidate("single", seg, h, k, idx, {ep.number: ep}, cfg, open_notes + end_notes))
    return out


def multi_candidates(ep_a: Episode, ep_b: Episode, an_a: EpisodeAnalysis, an_b: EpisodeAnalysis,
                     idx: dict[str, BeatScore], beats: dict[str, Beat], cfg: dict) -> list[Candidate]:
    """Episode N's late hook -> N's ending cliffhanger, then N+1 from its first new beat (recap skipped)
    -> a cliffhanger in N+1."""
    if an_a.ending_cliffhanger.score < 3 or an_b.pickup.status in ("unrelated", "not_applicable"):
        return []
    first_new = beats.get(an_b.pickup.first_new_beat_id or "")
    if first_new is None or first_new.episode != ep_b.number:
        if an_b.pickup.status != "immediate" or not ep_b.beats:
            return []
        first_new = ep_b.beats[0]
    end_beat = beats.get(an_a.ending_cliffhanger.beat_id or "")
    if end_beat is None or end_beat.episode != ep_a.number:
        spoken = [b for b in ep_a.beats if b.kind != "camera"]
        if not spoken:
            return []
        end_beat = max(spoken, key=lambda b: b.end)
    a_end = close_at(ep_a, end_beat, None)
    b_open = open_at(ep_b, first_new, LEAD_MAX_S)
    if not a_end or not b_open:
        return []
    max_lead = cfg.get("hook_window_s", 3.0) - 1.0
    b_start, b_first_id, _ = b_open
    skipped = f"skips {b_start:.1f} s of E{ep_b.number:02d} recap" if b_start > TOL else "no recap to skip"
    hooks = [b for b in ep_a.beats if hook_of(idx, b.id) >= cfg.get("hook_min", 3) and b.start < end_beat.start]
    cliffs = [b for b in ep_b.beats if cliff_of(idx, b.id) >= cfg.get("cliff_min", 3) and b.start >= first_new.start]
    out = []
    for k in cliffs:
        resolver = _resolver(idx, beats, k.id)
        b_end = close_at(ep_b, k, resolver)
        if not b_end:
            continue
        seg_b = Segment(episode=ep_b.number, start=b_start, end=b_end[0], first_beat_id=b_first_id,
                        last_beat_id=b_end[1])
        if seg_b.end - seg_b.start < MULTI_MIN_SEGMENT_S or answer_inside(resolver, [seg_b]):
            continue
        for h in hooks:
            opening = open_at(ep_a, h, max_lead)
            if not opening:
                continue
            seg_a = Segment(episode=ep_a.number, start=opening[0], end=a_end[0], first_beat_id=opening[1],
                            last_beat_id=a_end[1])
            segs = [seg_a, seg_b]
            if seg_a.end - seg_a.start < MULTI_MIN_SEGMENT_S or not _in_bounds(seg_length(segs), cfg):
                continue
            notes = opening[2] + [f"joins E{ep_a.number:02d} ending to E{ep_b.number:02d}; {skipped}"] + b_end[2]
            out.append(_candidate("multi", segs, h, k, idx, {ep_a.number: ep_a, ep_b.number: ep_b}, cfg, notes))
    return out


def _keep_best(cands: list[Candidate], keep: int, cfg: dict) -> list[Candidate]:
    """Greedy: best prelim first, skip anything with temporal IoU >= 0.5 against a kept one."""
    mid = sum(cfg.get("sweet_spot", [25, 45])) / 2
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (-c.prelim_score, abs(c.duration - mid), c.segments[0].start)):
        if len(kept) >= keep:
            break
        if all(time_iou(c.segments, k.segments) < 0.5 for k in kept):
            kept.append(c)
    return kept


def generate_candidates(episodes: list[Episode], analyses: list[EpisodeAnalysis], promo_cfg: dict) -> list[Candidate]:
    cfg = promo_cfg or {}
    idx = score_index(analyses)
    beats = beat_index(episodes)
    by_num = {a.episode: a for a in analyses if a is not None}
    eps = sorted(episodes, key=lambda e: e.number)
    keep = cfg.get("per_episode_keep", 4)
    kept: list[Candidate] = []
    for i, ep in enumerate(eps):
        pool = single_candidates(ep, idx, beats, cfg)
        nxt = eps[i + 1] if i + 1 < len(eps) else None
        if nxt and nxt.number == ep.number + 1 and ep.number in by_num and nxt.number in by_num:
            pool += multi_candidates(ep, nxt, by_num[ep.number], by_num[nxt.number], idx, beats, cfg)
        kept += _keep_best(pool, keep, cfg)
    kept.sort(key=lambda c: (-c.prelim_score, c.segments[0].episode, c.segments[0].start))
    for n, c in enumerate(kept, start=1):
        c.id = f"C{n:04d}"
    return kept
