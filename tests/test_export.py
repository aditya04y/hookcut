"""Exports, editor timelines and the episode report on synthetic data."""
from __future__ import annotations

import csv
import re

from hookcut import export as ex
from hookcut import rank as rk
from hookcut.candidates import generate_candidates, shot_at, still_frame_time
from hookcut.report import episode_report, season_summary
from hookcut.schema import EpisodeEdge, Pickup, RunResult

try:
    from test_candidates import CFG, bid, make_analysis, make_episode
except ImportError:  # tests/ became a package
    from tests.test_candidates import CFG, bid, make_analysis, make_episode


def small_run() -> RunResult:
    e1, e2 = make_episode(1, duration=60.0), make_episode(2, duration=60.0)
    h, end = bid(e1, "dialogue", 40.5), bid(e1, "dialogue", 52.5)
    h0, k0 = bid(e1, "dialogue", 4.5), bid(e1, "dialogue", 28.5)
    first_new, k = bid(e2, "dialogue", 8.5), bid(e2, "dialogue", 20.5)
    a1 = make_analysis(e1, {h: {"hook": 5}, h0: {"hook": 4}, k0: {"cliff": 4}}, ending=(end, 4))
    a2 = make_analysis(e2, {k: {"cliff": 5}}, pickup=Pickup(status="recap_then_advance", first_new_beat_id=first_new))
    a1.opening_hook = EpisodeEdge(beat_id=h0, score=2)
    cands = generate_candidates([e1, e2], [a1, a2], CFG)
    promos = rk.rank_promos(cands, [], [e1, e2], [a1, a2], CFG)
    assert any(p.kind == "multi" for p in promos)
    return RunResult(show="Synthetic Show", created_at="2026-11-19T12:00:00", episodes=[e1, e2], analyses=[a1, a2],
                     candidates=cands, promos=promos, episode_report=episode_report([e1, e2], [a1, a2]))


def test_write_outputs_writes_every_file(tmp_path):
    result = small_run()
    paths = ex.write_outputs(result, tmp_path)
    expected = {"result.json", "promos.json", "promos.csv", "promos.md", "all_candidates.csv",
                "episode_report.csv", "episode_report.md", "editor/editor_readme"}
    for p in result.promos:
        expected |= {f"editor/promo_{p.rank:02d}.{x}" for x in ("otio", "edl", "xml")}
    assert expected <= set(paths)
    for path in paths.values():
        assert open(path, encoding="utf-8").read().strip(), path
    rows = list(csv.DictReader(open(paths["promos.csv"], encoding="utf-8")))
    assert len(rows) == len(result.promos) and rows[0]["still_frame_time"].startswith("E0")
    multi = next(r for r in rows if r["kind"] == "multi")
    assert multi["segments"] == "E01 00:40.00–00:56.00; E02 00:08.00–00:24.00"
    cand_rows = list(csv.DictReader(open(paths["all_candidates.csv"], encoding="utf-8")))
    assert len(cand_rows) == len(result.candidates) and cand_rows[0]["status"].startswith("promo #")
    md = open(paths["promos.md"], encoding="utf-8").read()
    assert "## Wave" in md and "Still frame" in md
    readme = open(paths["editor/editor_readme"], encoding="utf-8").read()
    assert "E01.mp4" in readme and f"{ex.FPS} fps" in readme


def _edl_events(text: str) -> list[tuple[str, str, str, str, str]]:
    tc = r"(\d\d:\d\d:\d\d:\d\d)"
    return re.findall(rf"^\d{{3}}\s+(\S+)\s+V\s+C\s+{tc} {tc} {tc} {tc}", text, re.M)


def test_edl_timecodes_match_segments(tmp_path):
    result = small_run()
    paths = ex.write_outputs(result, tmp_path)
    for p in result.promos:
        edl = open(paths[f"editor/promo_{p.rank:02d}.edl"]).read()
        events = _edl_events(edl)
        assert len(events) == len(p.segments)
        rec = 0
        for (reel, src_in, src_out, rec_in, rec_out), s in zip(events, p.segments):
            a, b = ex.frames(s.start), ex.frames(s.end)
            assert reel == f"E{s.episode:02d}" and len(reel) <= 8
            assert (src_in, src_out) == (ex.timecode(a), ex.timecode(b))
            assert (rec_in, rec_out) == (ex.timecode(rec), ex.timecode(rec + b - a))
            rec += b - a
        # the hand-written fallback produces the same events
        assert _edl_events(ex.edl_text("x", p.segments)) == events


def test_timecode_format():
    assert ex.timecode(ex.frames(39.1)) == "00:00:39:03"
    assert ex.timecode(ex.frames(3725.5)) == "01:02:05:15"


def test_still_frame_is_inside_the_hook_shot():
    ep = make_episode(1)
    hook = ep.beat(bid(ep, "dialogue", 40.5))
    t = still_frame_time(ep, hook)
    shot = shot_at(ep, hook.start)
    assert t == 41.0 and shot.start <= t < shot.end
    # a line starting 0.3 s before its shot ends: the still stays on that shot (last frame), not the next
    ep = make_episode(1, lines=[(43.7, 45.5)])
    late = ep.beat(bid(ep, "dialogue", 43.7))
    assert 43.9 < still_frame_time(ep, late) < 44.0


def test_episode_report_flags_and_season_summary(tmp_path):
    result = small_run()
    rows = result.episode_report
    r1, r2 = rows
    assert r1["pickup_status"] == "not_applicable" and "weak opening (<3)" in r1["flags"]
    assert r1["opening_hook_time"] == "00:04.50" and r1["ending_cliff_score"] == 4
    assert r2["pickup_status"] == "recap_then_advance" and r2["pickup_delay_s"] == 8.5
    assert "recap longer than 5 s" in r2["flags"] and "weak ending (<3)" in r2["flags"]
    missing = episode_report(result.episodes, result.analyses[:1])
    assert missing[1]["flags"] == ["analysis failed"]
    s = season_summary(rows)
    assert s["avg_opening_hook"] == 2.5 and s["weakest_endings"][0] == {"episode": 2, "score": 0}
    assert s["unlock_candidates"] == []           # synthetic season has no episodes 5-15
    md = ex.episode_report_md(result)
    assert md.index("Season summary") < md.index("| Ep |") and "unlock" in md


def test_season_summary_unlock_window():
    rows = [{"episode": n, "opening_hook_score": 3, "ending_cliff_score": s, "ending_open_question": f"q{n}",
             "flags": []} for n, s in [(2, 5), (5, 3), (8, 5), (12, 4), (15, 5), (16, 5)]]
    s = season_summary(rows)
    assert [u["episode"] for u in s["unlock_candidates"]] == [8, 15, 12]
    assert "free episodes" in s["unlock_assumption"]
