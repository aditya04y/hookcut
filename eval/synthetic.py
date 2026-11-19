"""A tiny synthetic RunResult for tests and dry runs. All text is invented here (no client script).

Each episode has a contiguous camera track (3 s shots), dialogue lines that overlap the shot cuts (like the
real scripts), one long action beat and an sfx beat near the end.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from hookcut.schema import (Beat, BeatScore, Candidate, ColdViewerVerdict, Episode, EpisodeAnalysis, EpisodeEdge,
                            Pickup, Promo, RunResult, Scene, Segment)
from hookcut.textfmt import ep_tc

SPEAKERS = ("NOVA", "REED", "AUNT CORA")


def synthetic_episode(n: int, duration: float = 90.0) -> Episode:
    beats: list[tuple] = []
    t = 0.0
    while t < duration - 1e-6:                                   # camera track, 3 s shots
        beats.append(("camera", None, f"Shot {int(t // 3) + 1}: the greenhouse at night.", t, min(duration, t + 3)))
        t += 3
    t = 0.4
    i = 0
    while t + 2.4 < duration - 4:                                # lines that cross shot cuts
        beats.append(("dialogue", SPEAKERS[i % 3], f"Synthetic line {i + 1} of episode {n}.", t, t + 2.4))
        t += 2.6 + (i % 3) * 0.5
        i += 1
    beats.append(("action", None, "Reed drops a brass key into the fountain.", 31.0, 36.5))
    beats.append(("sfx", None, "A glass pane cracks.", duration - 3.0, duration - 0.5))
    beats.sort(key=lambda b: (b[3], b[0] != "camera"))
    bs = [Beat(id=f"E{n:02d}.B{k:03d}", episode=n, scene=1, kind=kind, speaker=who, text=text,
               start=round(s, 2), end=round(e, 2)) for k, (kind, who, text, s, e) in enumerate(beats, 1)]
    return Episode(number=n, source_file=f"synthetic-{n:02d}.txt", logline="Invented test episode.",
                   scenes=[Scene(number=1, heading="INT. GREENHOUSE - NIGHT", start=0.0, end=duration)],
                   beats=bs, duration=duration)


def _first(ep: Episode, kind: str, after: float) -> Beat:
    return next(b for b in ep.beats if b.kind == kind and b.start >= after)


def _last(ep: Episode, kind: str, before: float) -> Beat:
    return [b for b in ep.beats if b.kind == kind and b.end <= before][-1]


def _cand(cid: str, ep: Episode, start: float, end: float, prelim: float, hook_score: float = 4,
          cliff_score: float = 4) -> Candidate:
    """A single-segment candidate from [start, end]; hook/cliff = first/last spoken line inside."""
    inside = [b for b in ep.beats if b.start >= start - 0.05 and b.end <= end + 0.05]
    hook, cliff = _first(ep, "dialogue", start - 0.05), _last(ep, "dialogue", end + 0.05)
    return Candidate(id=cid, kind="single", segments=[Segment(episode=ep.number, start=start, end=end,
                     first_beat_id=inside[0].id, last_beat_id=inside[-1].id)], duration=round(end - start, 2),
                     hook_beat_id=hook.id, cliff_beat_id=cliff.id, hook_score=hook_score, cliff_score=cliff_score,
                     hook_type="reveal", open_question="Who sent the key?", prelim_score=prelim)


def _promo(rank: int, c: Candidate, eps: dict[int, Episode], v: Optional[ColdViewerVerdict]) -> Promo:
    hook = eps[int(c.hook_beat_id[1:3])].beat(c.hook_beat_id)
    cliff = eps[int(c.cliff_beat_id[1:3])].beat(c.cliff_beat_id)
    return Promo(rank=rank, candidate_id=c.id, kind=c.kind, segments=c.segments, duration=c.duration,
                 hook_time=ep_tc(hook.episode, hook.start), hook_text=hook.text,
                 cliffhanger_time=ep_tc(cliff.episode, cliff.start), cliffhanger_text=cliff.text,
                 open_question=c.open_question, why_it_works="Synthetic.", post_wave=1, cold_viewer=v,
                 scores={"prelim": c.prelim_score})


def synthetic_run(n_episodes: int = 4) -> RunResult:
    episodes = [synthetic_episode(n, 90.0 + 6 * (n % 3)) for n in range(1, n_episodes + 1)]
    eps = {e.number: e for e in episodes}
    analyses = []
    for e in episodes:
        lines = [b for b in e.beats if b.kind == "dialogue"]
        analyses.append(EpisodeAnalysis(
            episode=e.number, summary="Invented.", beats=[
                BeatScore(beat_id=lines[2].id, hook=4, hook_type="reveal", reason="synthetic"),
                BeatScore(beat_id=lines[-1].id, cliff=5, open_question="Who sent the key?", reason="synthetic")],
            opening_hook=EpisodeEdge(beat_id=lines[0].id, score=3), ending_cliffhanger=EpisodeEdge(
                beat_id=lines[-1].id, score=4), pickup=Pickup(status="immediate" if e.number > 1 else "not_applicable"),
            model="gemini-3.8-flash", prompt_version="analyze_system@synthetic"))

    cands: list[Candidate] = []
    for e in episodes:
        h = _first(e, "dialogue", 6.0)
        c1 = _first(e, "dialogue", h.start + 30)                  # clean ~33 s clip
        cands.append(_cand(f"C{len(cands) + 1:04d}", e, h.start, c1.end, prelim=9.0 - e.number * 0.5))
        h2 = _first(e, "dialogue", 45.0)
        c2 = _last(e, "dialogue", e.duration - 2)
        cands.append(_cand(f"C{len(cands) + 1:04d}", e, h2.start, c2.end, prelim=7.0 - e.number * 0.3))
    # a deliberately bad pick: the end cuts a spoken line in half (the track-overlap failure)
    e1 = episodes[0]
    line = _first(e1, "dialogue", 40.0)
    bad = _cand("C0099", e1, _first(e1, "dialogue", 12.0).start, round(line.start + 1.0, 2), prelim=8.5)
    bad.notes.append("synthetic: cuts a line")
    cands.append(bad)
    # one multi-segment candidate: episode 1 ending + episode 2 opening
    e2 = episodes[1]
    s1h = _first(e1, "dialogue", e1.duration - 20)
    s2c = _first(e2, "dialogue", 8.0)
    segs = [Segment(episode=1, start=s1h.start, end=e1.duration, first_beat_id=s1h.id, last_beat_id=e1.beats[-1].id),
            Segment(episode=2, start=_first(e2, "dialogue", 0).start, end=s2c.end,
                    first_beat_id=_first(e2, "dialogue", 0).id, last_beat_id=s2c.id)]
    cands.append(Candidate(id="C0100", kind="multi", segments=segs,
                           duration=round(sum(s.end - s.start for s in segs), 2), hook_beat_id=s1h.id,
                           cliff_beat_id=s2c.id, hook_score=3, cliff_score=4, prelim_score=6.0))

    def verdict(c: Candidate, i: int) -> ColdViewerVerdict:
        sa, kw = (4, 4) if i % 3 else (3, 2)
        v = "pass" if sa >= 4 and kw >= 4 else ("fail" if kw <= 2 else "borderline")
        return ColdViewerVerdict(candidate_id=c.id, premise="Invented premise.", conflict="Invented.",
                                 open_question="Invented?", hook_lands=bool(i % 3), standalone=sa, keep_watching=kw,
                                 verdict=v, model="groq:qwen/qwen3.8-27b")

    ranked = sorted(cands, key=lambda c: -c.prelim_score)
    verdicts = [verdict(c, i) for i, c in enumerate(ranked)]
    by_id = {v.candidate_id: v for v in verdicts}
    promos = [_promo(r, c, eps, by_id.get(c.id)) for r, c in enumerate(ranked, 1)]

    run_log = [{"stage": "config", "item": "note", "message": "models"}]
    for e in episodes:
        run_log.append({"stage": "analyze", "item": f"E{e.number:02d}", "provider": "gemini",
                        "model": "gemini-3.8-flash", "input_tokens": 6000, "output_tokens": 3000,
                        "latency_s": 20.0, "cached": e.number == 1, "attempts": 1})
    for c in ranked:
        run_log.append({"stage": "judge", "item": c.id, "provider": "groq", "model": "qwen/qwen3.8-27b",
                        "input_tokens": 1200, "output_tokens": 300, "latency_s": 1.0, "cached": False, "attempts": 1})
    run_log.append({"stage": "writer", "item": "why_lines", "provider": "groq", "model": "openai/gpt-oss-120b",
                    "input_tokens": 4000, "output_tokens": 1500, "latency_s": 3.0, "cached": False, "attempts": 1})
    return RunResult(
        show="Synthetic Greenhouse", created_at="2026-11-19T12:00:00",
        config={"models": {"analyze": "gemini:gemini-3.8-flash", "judge": "groq:qwen/qwen3.8-27b",
                           "writer": "groq:openai/gpt-oss-120b"},
                "promo": {"min_s": 20, "max_s": 60, "sweet_spot": [25, 45], "hook_window_s": 3.0, "judge_top_k": 40}},
        episodes=episodes, analyses=analyses, candidates=cands, verdicts=verdicts, promos=promos, run_log=run_log)


def write_synthetic_run(out_dir: str | Path, n_episodes: int = 4) -> Path:
    """Write result.json + run_log.jsonl + run_summary.json like a real run folder."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = synthetic_run(n_episodes)
    (out / "result.json").write_text(r.model_dump_json(indent=1))
    with open(out / "run_log.jsonl", "w") as f:
        for row in r.run_log:
            f.write(json.dumps(row) + "\n")
    (out / "run_summary.json").write_text(json.dumps({"show": r.show, "episodes": len(r.episodes),
                                                      "candidates": len(r.candidates), "promos": len(r.promos)}))
    return out / "result.json"


if __name__ == "__main__":
    import sys
    print(write_synthetic_run(sys.argv[1] if len(sys.argv) > 1 else "outputs/synthetic_eval"))
