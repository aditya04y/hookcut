"""Candidate generation on synthetic episodes (never the client's scripts).

Layout of make_episode(): a camera shot every 4 s ([0,4], [4,8], ...) and one spoken line per shot at
[4i+0.5, 4i+3.5], unless `lines` overrides them. Extra beats (action, long lines) are added explicitly.
"""
from __future__ import annotations

from hookcut.candidates import generate_candidates, time_iou
from hookcut.schema import Beat, BeatScore, Episode, EpisodeAnalysis, EpisodeEdge, Pickup, Scene

CFG = {"min_s": 20, "max_s": 60, "sweet_spot": [25, 45], "hook_window_s": 3.0, "hook_min": 3, "cliff_min": 3,
       "per_episode_keep": 50, "judge_top_k": 40, "final_top_n": 20, "max_per_episode_in_top": 2,
       "long_cut": False, "embeddings": "jaccard"}


def make_episode(number: int, duration: float = 120.0, lines: list[tuple[float, float]] | None = None,
                 extra: list[tuple[str, float, float]] | None = None) -> Episode:
    raw = []
    t = 0.0
    while t < duration - 1e-9:
        raw.append(("camera", t, min(t + 4.0, duration), f"Shot at {t:.0f}"))
        t += 4.0
    if lines is None:
        lines = [(s + 0.5, s + 3.5) for s in range(0, int(duration), 4) if s + 3.5 <= duration]
    raw += [("dialogue", a, b, f"Line at {a:.1f}") for a, b in lines]
    raw += [(k, a, b, f"{k.title()} at {a:.1f}") for k, a, b in (extra or [])]
    order = {"camera": 0, "action": 1, "sfx": 1, "dialogue": 2, "vo": 2}
    raw.sort(key=lambda r: (r[1], order[r[0]]))
    beats = [Beat(id=f"E{number:02d}.B{i:03d}", episode=number, scene=1, kind=k,
                  speaker="ANA" if k in ("dialogue", "vo") else None, text=txt, start=a, end=b)
             for i, (k, a, b, txt) in enumerate(raw, start=1)]
    return Episode(number=number, source_file=f"ep{number}.txt", scenes=[Scene(number=1, start=0, end=duration)],
                   beats=beats, duration=duration)


def bid(ep: Episode, kind: str, start: float) -> str:
    return next(b.id for b in ep.beats if b.kind == kind and abs(b.start - start) < 1e-6)


def make_analysis(ep: Episode, scores: dict[str, dict], ending: tuple[str | None, int] = (None, 0),
                  pickup: Pickup | None = None) -> EpisodeAnalysis:
    return EpisodeAnalysis(
        episode=ep.number, summary="synthetic", beats=[BeatScore(beat_id=k, **v) for k, v in scores.items()],
        opening_hook=EpisodeEdge(beat_id=None, score=3), ending_cliffhanger=EpisodeEdge(beat_id=ending[0], score=ending[1]),
        ending_open_question="what next?", pickup=pickup or Pickup(status="not_applicable"))


def segs_of(c):
    return [(s.episode, s.start, s.end) for s in c.segments]


# ----------------------------------------------------------------------------- basics
def test_single_clip_opens_on_shot_and_ends_after_cliff():
    ep = make_episode(1)
    h, k = bid(ep, "dialogue", 40.5), bid(ep, "dialogue", 72.5)
    a = make_analysis(ep, {h: {"hook": 5, "hook_type": "reveal", "needs_context": ["who Ana is"]},
                           k: {"cliff": 4, "open_question": "will she stay?", "needs_context": ["Who Ana is"]}})
    cands = generate_candidates([ep], [a], CFG)
    assert len(cands) == 1
    c = cands[0]
    # shot 40-44 starts 0.5 s before the hook line; cliff line ends 75.5, its shot runs to 76.0 with no new line
    assert segs_of(c) == [(1, 40.0, 76.0)]
    assert c.id == "C0001" and c.kind == "single" and c.duration == 36.0
    assert (c.hook_beat_id, c.cliff_beat_id, c.hook_type, c.open_question) == (h, k, "reveal", "will she stay?")
    assert c.needs_context == ["who Ana is"]           # union, case-insensitive dedupe
    assert any("end of the shot" in n for n in c.notes)


def test_unscored_beats_make_no_candidates():
    ep = make_episode(1)
    assert generate_candidates([ep], [make_analysis(ep, {})], CFG) == []
    assert generate_candidates([ep], [], CFG) == []


def test_lead_never_more_than_one_second_before_hook():
    # the hook line starts 1.5 s into its shot -> the clip opens on the line itself, not the shot
    lines = [(s + 0.5, s + 3.5) for s in range(0, 120, 4) if s != 40] + [(41.5, 43.5)]
    ep = make_episode(1, lines=lines)
    h, k = bid(ep, "dialogue", 41.5), bid(ep, "dialogue", 72.5)
    c = generate_candidates([ep], [make_analysis(ep, {h: {"hook": 4}, k: {"cliff": 4}})], CFG)[0]
    assert c.segments[0].start == 41.5


# --------------------------------------------------------------------------- edge cases
def test_hook_too_late_is_dropped():
    # visual hook (action at 50.0) while a line runs 47.0-51.0: starting on the line puts the hook 3 s in
    lines = [(s + 0.5, s + 3.5) for s in range(0, 120, 4) if s not in (44, 48)] + [(47.0, 51.0)]
    ep = make_episode(1, lines=lines, extra=[("action", 50.0, 52.0)])
    h, k = bid(ep, "action", 50.0), bid(ep, "dialogue", 80.5)
    assert generate_candidates([ep], [make_analysis(ep, {h: {"hook": 5}, k: {"cliff": 5}})], CFG) == []


def test_visual_hook_inside_a_line_moves_back_to_line_start_when_close():
    # action hook at 50.0 while a line runs 49.0-51.0: opening moves back to 49.0 (hook 1 s in)
    lines = [(s + 0.5, s + 3.5) for s in range(0, 120, 4) if s != 48] + [(49.0, 51.0)]
    ep = make_episode(1, lines=lines, extra=[("action", 50.0, 52.0)])
    h, k = bid(ep, "action", 50.0), bid(ep, "dialogue", 80.5)
    c = generate_candidates([ep], [make_analysis(ep, {h: {"hook": 5}, k: {"cliff": 5}})], CFG)[0]
    assert c.segments[0].start == 49.0


def test_end_never_cuts_inside_a_line():
    # cliff = action ending 70.0 while a line runs 68.5-73.0 (3 s past): cannot snap -> dropped
    lines = [(s + 0.5, s + 3.5) for s in range(0, 120, 4) if s not in (68, 72)] + [(68.5, 73.0)]
    ep = make_episode(1, lines=lines, extra=[("action", 66.0, 70.0)])
    h, k = bid(ep, "dialogue", 40.5), bid(ep, "action", 66.0)
    assert generate_candidates([ep], [make_analysis(ep, {h: {"hook": 5}, k: {"cliff": 5}})], CFG) == []
    # same, but the line ends 71.0 (1 s past): end snaps to the line's end
    lines = [(s + 0.5, s + 3.5) for s in range(0, 120, 4) if s not in (68,)] + [(68.5, 71.0)]
    ep = make_episode(1, lines=lines, extra=[("action", 66.0, 70.0)])
    h, k = bid(ep, "dialogue", 40.5), bid(ep, "action", 66.0)
    c = generate_candidates([ep], [make_analysis(ep, {h: {"hook": 5}, k: {"cliff": 5}})], CFG)[0]
    assert c.segments[0].end == 71.0


def test_answer_inside_clip_is_rejected_and_extension_stops_short_of_it():
    ep = make_episode(1, extra=[("action", 62.0, 63.0), ("action", 63.8, 64.0)])
    h, k = bid(ep, "dialogue", 40.5), bid(ep, "dialogue", 60.5)
    inside, after = bid(ep, "action", 62.0), bid(ep, "action", 63.8)
    a = make_analysis(ep, {h: {"hook": 5}, k: {"cliff": 5, "resolved_by": inside}})
    assert generate_candidates([ep], [a], CFG) == []
    a = make_analysis(ep, {h: {"hook": 5}, k: {"cliff": 5, "resolved_by": after}})
    c = generate_candidates([ep], [a], CFG)[0]
    assert c.segments[0].end == 63.5            # shot runs to 64.0 but the answer starts at 63.8


def test_duration_bounds():
    ep = make_episode(1)
    h = bid(ep, "dialogue", 40.5)
    short, ok_lo, ok_hi, long_ = (bid(ep, "dialogue", t) for t in (52.5, 56.5, 96.5, 100.5))
    scores = {h: {"hook": 5}, short: {"cliff": 4}, ok_lo: {"cliff": 4}, ok_hi: {"cliff": 4}, long_: {"cliff": 4}}
    cands = generate_candidates([ep], [make_analysis(ep, scores)], CFG)
    durs = sorted(c.duration for c in cands)
    assert durs == [20.0, 60.0]                 # 40.0 -> 60.0 and 40.0 -> 100.0; 16 s and 64 s dropped
    assert all(CFG["min_s"] <= d <= CFG["max_s"] for d in durs)


# ----------------------------------------------------------------------------- scoring
def test_prelim_weights_spoiler_and_ordering():
    ep = make_episode(1)
    h1, h2 = bid(ep, "dialogue", 8.5), bid(ep, "dialogue", 60.5)
    k1, k2 = bid(ep, "dialogue", 36.5), bid(ep, "dialogue", 92.5)
    mid = bid(ep, "dialogue", 76.5)
    a = make_analysis(ep, {h1: {"hook": 4}, k1: {"cliff": 4}, h2: {"hook": 4}, k2: {"cliff": 4},
                           mid: {"spoiler": 3}})
    cands = {segs_of(c)[0][1]: c for c in generate_candidates([ep], [a], CFG)}
    clean, spoiled = cands[8.0], cands[60.0]
    assert clean.spoiler == 0 and spoiled.spoiler == 3
    assert clean.prelim_score - spoiled.prelim_score == 0.75     # 3 spoiler points x 0.25
    assert clean.id < spoiled.id                                # ids follow descending prelim


def test_per_episode_keep_and_iou():
    ep = make_episode(1)
    scores = {bid(ep, "dialogue", t): {"hook": 4} for t in (4.5, 8.5, 12.5, 40.5, 44.5)}
    scores.update({bid(ep, "dialogue", t): {"cliff": 4} for t in (36.5, 40.5, 72.5, 84.5)})
    cands = generate_candidates([ep], [make_analysis(ep, scores)], dict(CFG, per_episode_keep=3))
    assert 1 <= len(cands) <= 3
    assert all(time_iou(a.segments, b.segments) < 0.5 for a in cands for b in cands if a is not b)
    assert [c.id for c in cands] == [f"C{i:04d}" for i in range(1, len(cands) + 1)]
    assert [c.prelim_score for c in cands] == sorted((c.prelim_score for c in cands), reverse=True)


# ---------------------------------------------------------------------------- multi
def _two_episodes(ending_score: int = 4, status: str = "recap_then_advance"):
    e1, e2 = make_episode(1, duration=60.0), make_episode(2, duration=60.0)
    h, end = bid(e1, "dialogue", 40.5), bid(e1, "dialogue", 52.5)
    first_new, k = bid(e2, "dialogue", 8.5), bid(e2, "dialogue", 20.5)
    a1 = make_analysis(e1, {h: {"hook": 5}}, ending=(end, ending_score))
    a2 = make_analysis(e2, {k: {"cliff": 5}}, pickup=Pickup(status=status, first_new_beat_id=first_new))
    return e1, e2, a1, a2


def test_multi_segment_skips_the_recap():
    e1, e2, a1, a2 = _two_episodes()
    multi = [c for c in generate_candidates([e1, e2], [a1, a2], CFG) if c.kind == "multi"]
    assert len(multi) == 1
    c = multi[0]
    # E01 40.0 -> 56.0 (end of the cliff line's shot), then E02 from the shot of the first new beat (8.0)
    assert segs_of(c) == [(1, 40.0, 56.0), (2, 8.0, 24.0)]
    assert c.duration == 32.0 and c.cliff_beat_id == bid(e2, "dialogue", 20.5)
    assert any("recap" in n for n in c.notes)
    single_twin = c.prelim_score + 0.5          # the multi penalty is the only difference from a single
    assert abs(single_twin - (5 + 5 + 1.5 + 0.5)) < 1e-9


def test_no_multi_when_episode_ending_is_weak_or_unrelated():
    for kwargs in ({"ending_score": 2}, {"status": "unrelated"}):
        e1, e2, a1, a2 = _two_episodes(**kwargs)
        assert not [c for c in generate_candidates([e1, e2], [a1, a2], CFG) if c.kind == "multi"]
