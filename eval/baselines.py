"""Two cheap baselines the tool must beat. Both return valid schema Candidates, snapped to beat boundaries so no
line is cut, with prelim_score set so `judge_candidates(..., top_k=len(c))` judges every one.

- last_window_baseline: the final ~45 s of each episode. Microdrama episodes end on a cliffhanger, so this is
  what a hurried editor would post. It is a strong baseline on purpose.
- random_window_baseline: a random 25-45 s stretch per episode (seeded), cut cleanly.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

from hookcut.schema import Candidate, Episode, Segment

from .common import EPS, beats_in, is_safe_cut

MIN_S, MAX_S, HOOK_WINDOW_S = 20.0, 60.0, 3.0   # the brief's 20-60 s, hook in the first 3 s
CLIFF_END_SLACK_S = 1.5   # random windows end within 1.5 s of their last line (hookcut.validate's rule).
                          # last_window keeps the episode's final frames: that is the naive editor's default.


def safe_cuts(ep: Episode) -> list[float]:
    """Every beat boundary (plus 0 and the episode end) where a cut splits no dialogue/action/sfx beat."""
    times = {0.0, round(ep.duration, 3)} | {b.start for b in ep.beats} | {b.end for b in ep.beats}
    return [t for t in sorted(times) if 0.0 <= t <= ep.duration + EPS and is_safe_cut(ep, t)]


def _opens_on_beat(ep: Episode, t: float, window: float) -> bool:
    """Does a non-camera beat (a line, an action) start within `window` s of t? Then it can serve as the hook."""
    return any(b.kind != "camera" and t - EPS <= b.start <= t + window for b in ep.beats)


def _closes_on_beat(ep: Episode, t: float, slack: float) -> bool:
    """Does a non-camera beat end within `slack` s before t? Then the clip ends on it (the editor's rule)."""
    return any(b.kind != "camera" and t - slack - EPS <= b.end <= t + EPS for b in ep.beats)


def snap_window(ep: Episode, start: float, end: float, min_s: float = MIN_S, max_s: float = MAX_S,
                hook_window: float = HOOK_WINDOW_S, end_slack: Optional[float] = None) -> Optional[tuple[float, float]]:
    """Nearest clean window to [start, end]: start on a safe cut that opens on a line within `hook_window` s
    (if any such cut exists), end on a safe cut, length kept inside [min_s, max_s] when the episode allows.
    With end_slack, prefer ends that fall within end_slack s after a line/action ends (no trailing shots)."""
    cuts = safe_cuts(ep)
    if len(cuts) < 2:
        return None
    starts = [t for t in cuts if t < ep.duration - 1.0]
    good = [t for t in starts if _opens_on_beat(ep, t, hook_window)]
    t0 = min(good or starts, key=lambda t: (abs(t - start), t))
    ends = [t for t in cuts if t > t0 and min_s - EPS <= t - t0 <= max_s + EPS]
    if not ends:  # episode too short for min_s after t0: take the longest clean window allowed
        ends = [t for t in cuts if t > t0 and t - t0 <= max_s + EPS]
    if not ends:
        return None
    if end_slack is not None:
        ends = [t for t in ends if _closes_on_beat(ep, t, end_slack)] or ends
    t1 = min(ends, key=lambda t: (abs(t - end), -t))
    return round(t0, 3), round(t1, 3)


def window_candidate(ep: Episode, t0: float, t1: float, cid: str, note: str) -> Optional[Candidate]:
    inside = beats_in(ep, t0, t1)
    story = [b for b in inside if b.kind != "camera"]
    if not story:
        return None
    seg = Segment(episode=ep.number, start=t0, end=t1, first_beat_id=inside[0].id, last_beat_id=inside[-1].id)
    return Candidate(id=cid, kind="single", segments=[seg], duration=round(t1 - t0, 2),
                     hook_beat_id=story[0].id, cliff_beat_id=story[-1].id, hook_score=0.0, cliff_score=0.0,
                     prelim_score=1.0, notes=[note])


def last_window_baseline(episodes: Sequence[Episode], seconds: float = 45) -> list[Candidate]:
    out = []
    for ep in episodes:
        w = snap_window(ep, ep.duration - seconds, ep.duration)
        c = w and window_candidate(ep, *w, cid=f"LW-E{ep.number:02d}", note=f"baseline:last_window_{seconds:g}s")
        if c:
            out.append(c)
    return out


def random_window_baseline(episodes: Sequence[Episode], seed: int = 7, per_episode: int = 1,
                           length: tuple[float, float] = (25, 45)) -> list[Candidate]:
    rng = random.Random(seed)
    out = []
    for ep in episodes:
        for k in range(1, per_episode + 1):
            L = rng.uniform(*length)
            s = rng.uniform(0.0, max(0.0, ep.duration - L))
            w = snap_window(ep, s, s + L, end_slack=CLIFF_END_SLACK_S)
            c = w and window_candidate(ep, *w, cid=f"RW-E{ep.number:02d}-{k}", note=f"baseline:random_seed{seed}")
            if c:
                out.append(c)
    return out


def evenly_spaced(items: Sequence, n: int) -> list:
    """n items spread evenly over the sequence (first and last included), without repeats."""
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    if n == 1:
        return [items[len(items) // 2]]
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]
