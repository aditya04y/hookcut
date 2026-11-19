"""Ranking, posting waves, variants and validation on synthetic data (no network: token-Jaccard similarity)."""
from __future__ import annotations

import numpy as np

import pytest

from hookcut import rank as rk
from hookcut.candidates import generate_candidates
from hookcut.schema import Candidate, ColdViewerVerdict, Segment
from hookcut.story_graph import AliasRow, CharacterNode, RelationChange, RelationEdge, StoryAlert, StoryGraph
from hookcut.validate import validate_promos

try:
    from test_candidates import CFG, bid, make_analysis, make_episode
except ImportError:  # tests/ became a package
    from tests.test_candidates import CFG, bid, make_analysis, make_episode


def verdict(cid: str, v: str = "pass", standalone: int = 4, keep: int = 4, lands: bool = True, q: str = ""):
    return ColdViewerVerdict(candidate_id=cid, premise="a woman is betrayed", conflict="x", open_question=q,
                             hook_lands=lands, standalone=standalone, keep_watching=keep, verdict=v)


def season(n: int = 10):
    """n episodes, each with one strong hook at 8.5 s and cliffs at 36.5 s and 92.5 s."""
    eps, ans = [], []
    for i in range(1, n + 1):
        ep = make_episode(i)
        h, h2 = bid(ep, "dialogue", 8.5), bid(ep, "dialogue", 60.5)
        k1, k2 = bid(ep, "dialogue", 36.5), bid(ep, "dialogue", 92.5)
        ans.append(make_analysis(ep, {h: {"hook": 5, "hook_type": "reveal"}, h2: {"hook": 4},
                                      k1: {"cliff": 4, "open_question": f"what does ep {i} hide?"},
                                      k2: {"cliff": 5, "open_question": f"who is at the door in ep {i}?"}}))
        eps.append(ep)
    return eps, ans


def test_fail_never_enters_and_borderline_is_flagged():
    eps, ans = season(3)
    cands = generate_candidates(eps, ans, CFG)
    best, second = cands[0], cands[1]
    vs = [verdict(best.id, "fail", 5, 5), verdict(second.id, "borderline")]
    promos = rk.rank_promos(cands, vs, eps, ans, CFG)
    ids = [p.candidate_id for p in promos]
    assert best.id not in ids and second.id in ids
    p = next(p for p in promos if p.candidate_id == second.id)
    assert "borderline cold-viewer" in p.flags and p.cold_viewer.verdict == "borderline"
    assert all("not cold-viewer tested" in q.flags for q in promos if q.candidate_id != second.id)
    assert p.why_it_works == "" and p.hook_time.startswith("E0") and p.scores["final"] > 0


def test_diversity_caps_per_episode_and_time_overlap():
    eps, ans = season(4)
    cands = generate_candidates(eps, ans, CFG)
    promos = rk.rank_promos(cands, [verdict(c.id) for c in cands], eps, ans, dict(CFG, max_per_episode_in_top=1))
    assert len(promos) == 4 and len({p.segments[0].episode for p in promos}) == 4
    promos = rk.rank_promos(cands, [verdict(c.id) for c in cands], eps, ans, CFG)
    by_ep: dict[int, list] = {}
    for p in promos:
        by_ep.setdefault(p.segments[0].episode, []).append(p)
    assert max(len(v) for v in by_ep.values()) <= 2
    for ps in by_ep.values():
        if len(ps) == 2:
            a, b = ps
            inter = max(0.0, min(a.segments[0].end, b.segments[0].end) - max(a.segments[0].start, b.segments[0].start))
            assert inter / min(a.duration, b.duration) <= rk.MAX_TIME_OVERLAP


def test_mmr_prefers_a_different_story_over_a_near_duplicate():
    seg = [Segment(episode=1, start=0, end=30, first_beat_id="x", last_beat_id="y")]
    items = [Candidate(id=f"C{i}", kind="single", segments=[s.model_copy(update={"episode": i}) for s in seg],
                       duration=30, hook_beat_id="h", cliff_beat_id="k", hook_score=4, cliff_score=4)
             for i in (1, 2, 3)]
    sim, backend = rk.similarity_matrix(["who is the stranger at the wedding", "who is the stranger at the wedding",
                                         "will the money save her father"], "jaccard")
    assert backend == "jaccard" and sim[0, 1] == 1.0 and sim[0, 2] == 0.0
    picked, _ = rk.mmr_select(items, {"C1": 1.0, "C2": 0.95, "C3": 0.8}, sim, 2, 2)
    assert picked == [0, 2]
    picked, _ = rk.mmr_select(items, {"C1": 1.0, "C2": 0.95, "C3": 0.8}, np.eye(3), 2, 2)
    assert picked == [0, 1]


def test_posting_waves():
    eps, ans = season(10)
    # E01 cliff at 92.5 carries a big spoiler; early season = E01-E03
    ans[0].beats.append(ans[0].beats[0].model_copy(update={"beat_id": bid(eps[0], "dialogue", 76.5), "hook": 0,
                                                           "spoiler": 3}))
    cands = generate_candidates(eps, ans, CFG)
    promos = rk.rank_promos(cands, [verdict(c.id, standalone=5 if c.segments[0].episode < 9 else 3) for c in cands],
                            eps, ans, dict(CFG, final_top_n=20))
    assert [p.rank for p in promos] == list(range(1, len(promos) + 1))
    assert [p.post_wave for p in promos] == sorted(p.post_wave for p in promos)
    for p in promos:
        ep, sp = p.segments[0].episode, p.scores["spoiler"]
        expected = 3 if sp >= 2 else (1 if ep <= 3 and sp <= 1 and p.scores["standalone"] >= 4 else 2)
        assert p.post_wave == expected, (p.candidate_id, ep, sp)
    assert {1, 2, 3} <= {p.post_wave for p in promos}


def test_variants_keep_the_cliffhanger_and_obey_cut_rules():
    eps, ans = season(2)
    cands = generate_candidates(eps, ans, CFG)
    promos = rk.rank_promos(cands, [], eps, ans, CFG)
    p = next(p for p in promos if p.variants)
    for v in p.variants:
        assert set(v) == {"label", "segments", "duration", "hook_beat_id", "note"}
        assert CFG["min_s"] <= v["duration"] <= CFG["max_s"]
        assert abs(v["segments"][-1]["end"] - p.segments[-1].end) <= 1.5
    assert not any(v["label"].startswith("long cut") for v in p.variants)
    assert validate_promos(promos, eps, CFG, ans) == []
    # a long cut appears only when asked for: needs a hook >= 90 s before a cliff
    ep = make_episode(1, duration=140.0)
    h, k = bid(ep, "dialogue", 8.5), bid(ep, "dialogue", 112.5)
    h2 = bid(ep, "dialogue", 76.5)
    a = make_analysis(ep, {h: {"hook": 4}, h2: {"hook": 5}, k: {"cliff": 5}})
    cfg = dict(CFG, long_cut=True)
    promos = rk.rank_promos(generate_candidates([ep], [a], cfg), [], [ep], [a], cfg)
    longs = [v for v in promos[0].variants if v["label"].startswith("long cut")]
    assert len(longs) == 1 and 90 <= longs[0]["duration"] <= 120
    assert validate_promos(promos, [ep], cfg, [a]) == []


def test_flags_for_visual_hook_multi_and_context():
    ep = make_episode(1, extra=[("action", 40.0, 41.0)])
    h, k = bid(ep, "action", 40.0), bid(ep, "dialogue", 72.5)
    a = make_analysis(ep, {h: {"hook": 5, "needs_context": ["the twins swapped"]}, k: {"cliff": 5}})
    p = rk.rank_promos(generate_candidates([ep], [a], CFG), [], [ep], [a], CFG)[0]
    assert "visual-only hook: check footage" in p.flags
    assert "needs context: the twins swapped" in p.flags


# ----------------------------------------------------------------------------- story-graph rules
PULL = {"ANA": 0.4, "BEA": 0.3, "CAL": 0.2, "DAN": 0.05}     # leads (top 3 promo pull): Ana, Bea, Cal


def speak(ep, start: float, who: str) -> None:
    """season() gives every line to ANA; hand one line to someone else."""
    next(b for b in ep.beats if b.kind == "dialogue" and abs(b.start - start) < 1e-6).speaker = who


def story(eps, relationships=(), alerts=()) -> StoryGraph:
    labels = sorted({b.speaker for e in eps for b in e.beats if b.speaker})
    nums = [e.number for e in eps]
    return StoryGraph(
        show="synthetic", relationships=list(relationships), alerts=list(alerts),
        characters=[CharacterNode(id=x.lower(), name=x.title(), promo_pull=PULL.get(x, 0.0)) for x in labels],
        aliases=[AliasRow(alias=x, canonical=x.lower(), first_ep=nums[0], last_ep=nums[-1], episodes=nums,
                          sources=["speaker label"]) for x in labels])


def alert(aid, kind, ep, beat_id, who):
    return StoryAlert(id=aid, type=kind, severity="high", confidence=0.8, ep=ep, beat_ids=[beat_id], characters=who)


def test_rank_with_a_story_graph_protects_twists_tags_feuds_and_favours_leads():
    eps, ans = season(10)                                     # early season = E01-E03
    speak(eps[0], 8.5, "DAN")                                 # E01's first clip (8-40 s): Dan's hook, Bea and Ana
    speak(eps[0], 12.5, "BEA")
    speak(eps[8], 16.5, "CAL")                                # E09's first clip: Ana's hook, Cal on screen
    graph = story(eps, relationships=[
        RelationEdge(a="ana", b="bea", kind="rivalry", label="enemy", core=True, conflict_intensity=80.0),
        RelationEdge(a="cal", b="ana", kind="family", label="sister", core=True, conflict_intensity=70.0,
                     changes=[RelationChange(ep=9, to="betrayer")]),
        RelationEdge(a="bea", b="dan", kind="alliance", label="conspirator", core=True, conflict_intensity=90.0)],
        alerts=[alert("A001", "relationship_reversal", 1, bid(eps[0], "dialogue", 20.5), ["ana", "bea"]),
                alert("A002", "identity_reveal", 9, bid(eps[8], "dialogue", 24.5), ["cal"])])
    cands = generate_candidates(eps, ans, CFG)
    vs = [verdict(c.id) for c in cands]
    plain = rk.rank_promos(cands, vs, eps, ans, CFG)
    assert ([p.model_dump() for p in rk.rank_promos(cands, vs, eps, ans, CFG, graph=None)]
            == [p.model_dump() for p in plain])                            # no graph: exactly as before
    ranked = rk.rank_promos(cands, vs, eps, ans, CFG, graph=graph.model_dump())   # the saved JSON works too
    by, before = {p.candidate_id: p for p in ranked}, {p.candidate_id: p for p in plain}
    first = {ep: by[next(c.id for c in cands if c.segments[0].episode == ep and c.segments[0].start < 10)]
             for ep in (1, 9)}
    # a: a late twist waits for warm audiences; an early-season reversal is the premise and keeps wave 1
    assert first[9].post_wave == 3 and rk.REVEAL_FLAG in first[9].flags
    assert before[first[9].candidate_id].post_wave != 3
    assert first[1].post_wave == 1 and rk.PREMISE_FLAG in first[1].flags and rk.REVEAL_FLAG not in first[1].flags
    # b: feuds on screen are tagged, bigger draw first; the allied pair (Bea + Dan, intensity 90) is no feud
    assert "feud: Ana vs Bea" in first[1].flags and "feud: Ana vs Cal" in first[9].flags
    assert not any("Dan" in f for p in ranked for f in p.flags)
    # c: a lead's hook line earns the flat bonus, Dan's does not
    assert (first[9].scores["lead_hook"], first[1].scores["lead_hook"]) == (1.0, 0.0)
    assert first[9].scores["final"] == pytest.approx(before[first[9].candidate_id].scores["final"]
                                                     + rk.FINAL_WEIGHTS["lead_hook"], abs=1e-3)
    assert first[1].scores["final"] == before[first[1].candidate_id].scores["final"]
    assert validate_promos(ranked, eps, CFG, ans) == []


def test_feud_coverage_swaps_in_a_passing_clip_but_is_not_a_quota():
    eps, ans = season(4)
    speak(eps[3], 64.5, "BEA")              # the Ana-Bea feud is on screen only in E04's second clip (60-96 s)
    graph = story(eps, relationships=[RelationEdge(a="ana", b="bea", kind="rivalry", label="rival", core=True,
                                                   conflict_intensity=80.0)])
    cands = generate_candidates(eps, ans, CFG)
    feud = next(c for c in cands if c.segments[0].episode == 4 and c.segments[0].start > 50)
    cfg = dict(CFG, final_top_n=3)

    def ids(verdict_of_feud: str, graph_=graph):
        vs = [verdict(c.id, verdict_of_feud, standalone=2, keep=1) if c is feud else verdict(c.id) for c in cands]
        return [p.candidate_id for p in rk.rank_promos(cands, vs, eps, ans, cfg, graph=graph_)]

    assert feud.id not in ids("pass", None)                  # too weak for the plain top 3
    picked = ids("pass")
    assert len(picked) == 3 and feud.id in picked            # swapped in for the weakest pick
    assert feud.id not in ids("borderline")                  # only a cold-viewer pass can cover a feud


# ----------------------------------------------------------------------------- validation
def _clean():
    eps, ans = season(1)
    promos = rk.rank_promos(generate_candidates(eps, ans, CFG), [], eps, ans, CFG)
    for p in promos:
        p.variants = []
    assert validate_promos(promos, eps, CFG, ans) == []
    return eps, ans, promos[0]


def _checks(p, eps, ans):
    return {i["check"] for i in validate_promos([p], eps, CFG, ans)}


def test_validation_catches_each_broken_rule():
    eps, ans, good = _clean()
    s = good.segments[0]

    def broken(**seg):
        p = good.model_copy(deep=True)
        p.segments = [s.model_copy(update=seg)]
        p.duration = sum(x.end - x.start for x in p.segments)
        return p

    assert "duration" in _checks(broken(end=s.start + 10), eps, ans)
    assert "bounds" in _checks(broken(end=eps[0].duration + 5), eps, ans)
    assert "mid_line_cut" in _checks(broken(start=s.start + 1.5), eps, ans)     # inside the hook line
    assert "hook_window" in _checks(broken(start=s.start - 4.0), eps, ans)
    assert "cliff_at_end" in _checks(broken(end=s.end + 4.0), eps, ans)
    assert "beat_ids" in _checks(broken(first_beat_id="E01.B999"), eps, ans)
    p = good.model_copy(deep=True)
    p.duration += 3
    assert "duration_field" in _checks(p, eps, ans)
    p = good.model_copy(deep=True)
    half = (s.start + s.end) / 2 - 0.5        # a cut at a shot boundary between lines
    half = round(half / 4) * 4
    p.segments = [s.model_copy(update={"end": half}), s.model_copy(update={"start": half - 8.0})]
    assert "order" in _checks(p, eps, ans)
    # answer inside: point the cliff's resolver at a beat in the middle of the clip
    cliff_id = next(b.beat_id for b in ans[0].beats if b.cliff and abs(eps[0].beat(b.beat_id).start
                                                                       - _start_of(good.cliffhanger_time)) < 0.01)
    mid = next(b.id for b in eps[0].beats if s.start + 5 < b.start < s.end - 5)
    ans2 = [ans[0].model_copy(deep=True)]
    next(b for b in ans2[0].beats if b.beat_id == cliff_id).resolved_by = mid
    assert "answer_inside" in _checks(good, eps, ans2)
    # a broken variant is reported with its label
    p = good.model_copy(deep=True)
    p.variants = [{"label": "shortest cut", "segments": [s.model_copy(update={"start": s.start + 1.5}).model_dump()],
                   "duration": 0, "hook_beat_id": "E01.B999", "note": ""}]
    assert any(c.startswith("variant 1 (shortest cut)") for c in _checks(p, eps, ans))


def _start_of(ep_time: str) -> float:
    m, s = ep_time.split()[1].split(":")
    return int(m) * 60 + float(s)
