"""Structural checks on the final promo list (and each promo's variants). Plain code; an empty list = clean.

Each issue: {"candidate_id", "check", "detail"}. These are the brief's hard rules plus the editor's:
length, inside the episode, segment order, no cut inside a spoken line, hook in the first seconds,
cliffhanger at the end, answer not inside, beat ids that exist.
"""
from __future__ import annotations

import re
from typing import Optional

from .schema import Beat, Episode, EpisodeAnalysis, Promo, Segment

TOL = 0.05                # timing tolerance (s)
CLIFF_END_SLACK_S = 1.5   # the cliff beat must end within this of the clip end (shot extension allowance)
LONG_CUT_S = (90.0, 120.0)
SPEECH = ("dialogue", "vo")
RE_EPTC = re.compile(r"E(\d+)\s+(\d+):(\d+(?:\.\d+)?)")


def _find_beat(eps: dict[int, Episode], when: str, text: str) -> Optional[Beat]:
    """Resolve 'E01 00:39.10' + its text back to a beat (Promo stores times and text, not beat ids)."""
    m = RE_EPTC.match(when or "")
    ep = eps.get(int(m.group(1))) if m else None
    if ep is None:
        return None
    t = int(m.group(2)) * 60 + float(m.group(3))
    hits = [b for b in ep.beats if abs(b.start - t) <= 0.01]
    return next((b for b in hits if b.text and b.text[:40] in (text or "")), hits[0] if hits else None)


def _check_clip(segs: list[Segment], hook: Optional[Beat], cliff: Optional[Beat], resolver: Optional[Beat],
                eps: dict[int, Episode], lo: float, hi: float, window: float) -> list[tuple[str, str]]:
    """All rules for one cut. Returns (check, detail) pairs."""
    out: list[tuple[str, str]] = []
    if not segs:
        return [("segments", "no segments")]
    total = sum(s.end - s.start for s in segs)
    if not (lo - TOL <= total <= hi + TOL):
        out.append(("duration", f"total {total:.2f} s outside [{lo}, {hi}]"))
    for i, s in enumerate(segs, start=1):
        ep = eps.get(s.episode)
        if ep is None:
            out.append(("episode", f"segment {i}: episode {s.episode} not in input"))
            continue
        if s.start < -TOL or s.end > ep.duration + TOL or s.end <= s.start:
            out.append(("bounds", f"segment {i}: {s.start:.2f}-{s.end:.2f} outside episode 0-{ep.duration:.2f}"))
        ids = {b.id for b in ep.beats}
        for bid in (s.first_beat_id, s.last_beat_id):
            if bid not in ids:
                out.append(("beat_ids", f"segment {i}: beat {bid} not in E{s.episode:02d}"))
        for edge, t in (("start", s.start), ("end", s.end)):
            line = next((b for b in ep.beats if b.kind in SPEECH and b.start < t - TOL and b.end > t + TOL), None)
            if line:
                out.append(("mid_line_cut", f"segment {i} {edge} {t:.2f} cuts line {line.id} "
                                            f"({line.start:.2f}-{line.end:.2f})"))
        if i > 1 and (s.episode, s.start) < (segs[i - 2].episode, segs[i - 2].end - TOL):
            out.append(("order", f"segment {i} starts before segment {i - 1} ends (or in an earlier episode)"))
    first, last = segs[0], segs[-1]
    if hook is None:
        out.append(("hook_beat", "hook beat not found"))
    elif hook.episode != first.episode or not (first.start - TOL <= hook.start <= first.start + window + TOL):
        out.append(("hook_window", f"hook {hook.id} starts {hook.start - first.start:.2f} s into the clip "
                                   f"(allowed 0-{window})"))
    if cliff is None:
        out.append(("cliff_beat", "cliffhanger beat not found"))
    elif cliff.episode != last.episode or not (-TOL <= last.end - cliff.end <= CLIFF_END_SLACK_S + TOL):
        out.append(("cliff_at_end", f"cliff {cliff.id} ends {last.end - cliff.end:.2f} s before the clip end "
                                    f"(allowed 0-{CLIFF_END_SLACK_S})"))
    if resolver and any(s.episode == resolver.episode and s.start - TOL <= resolver.start < s.end - TOL
                        for s in segs):
        out.append(("answer_inside", f"resolving beat {resolver.id} plays inside the clip"))
    return out


def validate_promos(promos: list[Promo], episodes: list[Episode], promo_cfg: dict,
                    analyses: Optional[list[EpisodeAnalysis]] = None) -> list[dict]:
    """`analyses` (optional) enables the 'answer inside the clip' check (Promo carries no resolved_by)."""
    cfg = promo_cfg or {}
    lo, hi, window = cfg.get("min_s", 20), cfg.get("max_s", 60), cfg.get("hook_window_s", 3.0)
    eps = {e.number: e for e in episodes}
    beats = {b.id: b for e in episodes for b in e.beats}
    resolved_by = {s.beat_id: s.resolved_by for a in (analyses or []) if a for s in a.beats if s.resolved_by}
    issues: list[dict] = []

    for p in promos:
        hook = _find_beat(eps, p.hook_time, p.hook_text)
        cliff = _find_beat(eps, p.cliffhanger_time, p.cliffhanger_text)
        resolver = beats.get(resolved_by.get(cliff.id, "")) if cliff else None
        found = _check_clip(p.segments, hook, cliff, resolver, eps, lo, hi, window)
        if abs(p.duration - sum(s.end - s.start for s in p.segments)) > TOL:
            found.append(("duration_field", f"duration {p.duration:.2f} != sum of segments"))
        for n, var in enumerate(p.variants, start=1):
            label = str(var.get("label", ""))
            vlo, vhi = LONG_CUT_S if label.startswith("long cut") else (lo, hi)
            try:
                vsegs = [Segment(**s) for s in var.get("segments", [])]
            except Exception as exc:  # malformed variant dict
                found.append((f"variant {n}: segments", f"unreadable: {exc}"))
                continue
            for check, detail in _check_clip(vsegs, beats.get(var.get("hook_beat_id", "")), cliff, resolver,
                                             eps, vlo, vhi, window):
                found.append((f"variant {n} ({label}): {check}", detail))
        issues += [{"candidate_id": p.candidate_id, "check": c, "detail": d} for c, d in found]
    return issues
