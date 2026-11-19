"""judge.py and writer.py without network, on small synthetic episodes."""
import json
import threading

import pytest

from hookcut import judge, writer
from hookcut.llm.base import LLMClient
from hookcut.runlog import RunLog
from hookcut.schema import Beat, Candidate, ColdViewerVerdict, Episode, Promo, Scene, Segment


def make_episode(n: int, lines: list[tuple[str, str, str | None]], step: float = 2.0) -> Episode:
    beats = [Beat(id=f"E{n:02d}.B{i:03d}", episode=n, scene=1, kind=k, speaker=sp, text=t,
                  start=(i - 1) * step, end=i * step) for i, (k, t, sp) in enumerate(lines, 1)]
    dur = len(lines) * step
    return Episode(number=n, source_file=f"ep{n}.txt", logline="SECRET LOGLINE", continuity_notes="SECRET NOTES",
                   scenes=[Scene(number=1, start=0, end=dur)], beats=beats, duration=dur)


EP1 = make_episode(1, [("dialogue", f"line one-{i}", "NOVA") for i in range(1, 21)])          # 40 s
EP3 = make_episode(3, [("action", f"action three-{i}", None) for i in range(1, 21)])          # 40 s


def cand(cid, segs, prelim=1.0):
    segments = [Segment(episode=e, start=s, end=t, first_beat_id=f"E{e:02d}.B{int(s // 2) + 1:03d}",
                        last_beat_id=f"E{e:02d}.B{int(t // 2):03d}") for e, s, t in segs]
    return Candidate(id=cid, kind="multi" if len(segs) > 1 else "single", segments=segments,
                     duration=sum(t - s for _, s, t in segs), hook_beat_id=segments[0].first_beat_id,
                     cliff_beat_id=segments[-1].last_beat_id, hook_score=4, cliff_score=4, prelim_score=prelim)


VERDICT = {"premise": "A woman is betrayed.", "conflict": "Nova vs her ex", "open_question": "Will she leave?",
           "confusions": "who is the man?", "hook_lands": True, "standalone": 4, "keep_watching": 5}


class FakeClient(LLMClient):
    provider = "fake"

    def __init__(self, replies: list, fail_if: str = ""):
        super().__init__("canned")
        self.replies, self.fail_if = list(replies), fail_if
        self.seen: list[str] = []
        self._lock = threading.Lock()

    def _call(self, system, user, json_mode):
        with self._lock:
            self.seen.append(user)
            r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if self.fail_if and self.fail_if in user:
            raise RuntimeError("rate limited")
        return (r if isinstance(r, str) else json.dumps(r)), 80, 40


@pytest.mark.parametrize("hook,standalone,keep,expected", [
    (True, 4, 4, "pass"), (False, 5, 5, "borderline"), (True, 3, 5, "borderline"),
    (True, 2, 5, "fail"), (True, 5, 2, "fail"), (False, 1, 1, "fail")])
def test_verdict_rule(hook, standalone, keep, expected):
    assert judge.verdict_for(hook, standalone, keep) == expected


def test_clip_shows_only_the_clip_with_relative_times_and_cuts():
    c = cand("C0001", [(1, 10.0, 20.0), (3, 30.0, 36.0)])
    text = judge.clip_text(c, {1: EP1, 3: EP3})
    lines = text.splitlines()
    assert lines[0] == "CLIP (00:16.00 long):"
    assert lines[1] == "[00:00.00-00:02.00] NOVA: line one-6"           # episode time 10.0 -> clip time 0
    assert "line one-5" not in text and "line one-11" not in text       # nothing outside the segment
    assert judge.CUT in lines
    after_cut = lines[lines.index(judge.CUT) + 1]
    assert after_cut == "[00:10.00-00:12.00] ACTION: action three-16"    # second segment continues the clock
    assert "SECRET" not in text and "E01.B" not in text                 # no logline/notes/ids


def test_judge_top_k_order_and_failures():
    cands = [cand("C0001", [(1, 0.0, 30.0)], prelim=0.2), cand("C0002", [(1, 10.0, 40.0)], prelim=0.9),
             cand("C0003", [(3, 0.0, 30.0)], prelim=0.5)]
    # C0003 (episode 3 text) hits a provider error on every attempt; C0001 is outside top_k
    client = FakeClient(["{oops", VERDICT], fail_if="action three-1")
    log = RunLog()
    out = judge.judge_candidates(cands, [EP1, EP3], client, top_k=2, log=log, concurrency=1)
    assert [v.candidate_id for v in out] == ["C0002"]
    v = out[0]
    assert v.verdict == "pass" and v.confusions == ["who is the man?"] and v.model == "fake:canned"
    assert any("REJECTED" in u for u in client.seen)                     # bad JSON was re-asked
    assert [r["item"] for r in log.rows if "error" in r] == ["C0003"]
    assert not any("C0001" in str(r) for r in log.rows)


def promo(cid, rank, verdict=None):
    seg = Segment(episode=1, start=0, end=30, first_beat_id="E01.B001", last_beat_id="E01.B015")
    return Promo(rank=rank, candidate_id=cid, kind="single", segments=[seg], duration=30, hook_time="E01 00:00.00",
                 hook_text="You will never see your daughter again.", cliffhanger_time="E01 00:28.00",
                 cliffhanger_text="The car door opens.", open_question="Who is in the car?", why_it_works="",
                 post_wave=1, cold_viewer=verdict)


def test_writer_validates_ids_then_falls_back_to_template():
    v = ColdViewerVerdict(candidate_id="C0001", premise="A kidnapping threat.", conflict="", open_question="",
                          hook_lands=True, standalone=4, keep_watching=4, verdict="pass")
    promos = [promo("C0001", 1, v), promo("C0002", 2)]
    partial = {"lines": [{"id": "C0001", "why": "A threat in the first line, then a door opens on someone unseen."}]}
    full = {"lines": partial["lines"] + [{"id": "C0002", "why": "Same threat; the open door is the question."}]}
    client = FakeClient([partial, full])
    log = RunLog()
    out = writer.write_why_lines(promos, [EP1], client, log=log)
    assert [p.why_it_works for p in out] == [full["lines"][0]["why"], full["lines"][1]["why"]]
    assert "missing or empty lines for ids: C0002" in client.seen[1]
    assert '"cold_viewer_premise": "A kidnapping threat."' in client.seen[0]

    log2 = RunLog()
    out2 = writer.write_why_lines(promos, [EP1], FakeClient(["no json"]), log=log2)
    assert all(p.why_it_works.startswith('Opens on "You will never') for p in out2)
    assert all(len(p.why_it_works.split()) <= writer.MAX_WORDS for p in out2)
    assert "template" in log2.rows[0]["error"]
    assert writer.write_why_lines([], [EP1], FakeClient(["{}"]), log=RunLog()) == []


def test_select_finalists_covers_every_episode_first():
    from hookcut.judge import select_finalists
    from hookcut.schema import Candidate, Segment

    def cand(i, ep, score):
        seg = Segment(episode=ep, start=0, end=30, first_beat_id=f"E{ep:02d}.B001", last_beat_id=f"E{ep:02d}.B009")
        return Candidate(id=f"C{i:04d}", kind="single", segments=[seg], duration=30, hook_beat_id=seg.first_beat_id,
                         cliff_beat_id=seg.last_beat_id, hook_score=4, cliff_score=4, prelim_score=score)
    cands = [cand(1, 1, 9.9), cand(2, 1, 9.8), cand(3, 1, 9.7), cand(4, 2, 5.0), cand(5, 3, 4.0)]
    picked = select_finalists(cands, 4)
    assert {c.segments[0].episode for c in picked} == {1, 2, 3}      # every episode gets a test
    assert [c.id for c in picked] == ["C0001", "C0002", "C0004", "C0005"]
