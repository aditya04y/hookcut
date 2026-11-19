"""analyze.py without network: a fake LLMClient returns canned JSON for small synthetic episodes."""
import json
import re
import threading

import pytest

from hookcut import analyze
from hookcut.llm.base import LLMClient
from hookcut.runlog import RunLog
from hookcut.schema import Beat, Episode, Scene


def make_episode(n: int, lines: list[tuple[str, str, str | None]], step: float = 3.0) -> Episode:
    """lines: (kind, text, speaker); each beat lasts `step` seconds."""
    beats = [Beat(id=f"E{n:02d}.B{i:03d}", episode=n, scene=1, kind=k, speaker=sp, text=t,
                  start=(i - 1) * step, end=i * step) for i, (k, t, sp) in enumerate(lines, 1)]
    dur = len(lines) * step
    return Episode(number=n, source_file=f"ep{n}.txt", logline=f"Synthetic logline {n}.",
                   continuity_notes=f"Synthetic notes {n}.", scenes=[Scene(number=1, start=0, end=dur)],
                   beats=beats, duration=dur)


EP1 = make_episode(1, [
    ("camera", "Wide shot of a rooftop bar at night, neon signs flickering over the skyline.", None),
    ("dialogue", "You said you'd be alone.", "MAYA"),
    ("action", "Derek throws his drink in Maya's face in front of the whole bar.", None),
    ("dialogue", "I never loved you.", "DEREK"),
    ("dialogue", "Then who is she?", "MAYA"),
    ("camera", "Close-up on a stranger in a red coat stepping out of the elevator.", None),
    ("sfx", "Elevator chime.", None),
    ("camera", "Cut to black.", None),
], step=5.0)   # 40 s, so the 25 s tail shown to episode 2 starts at E01.B004
EP2 = make_episode(2, [
    ("camera", "Same rooftop. The stranger in the red coat walks toward Maya.", None),
    ("dialogue", "I'm his wife.", "WOMAN IN RED"),
    ("action", "Maya steps back, stunned.", None),
    ("dialogue", "Wife?", "MAYA"),
    ("camera", "Close-up on Derek, cornered.", None),
    ("dialogue", "This isn't over.", "DEREK"),
])


def reply(ep: Episode, **over) -> dict:
    n = ep.number
    base = {
        "summary": f"Synthetic summary {n}.",
        "characters": [{"name": "Maya", "aliases": [], "role": "lead"}],
        "beats": [
            {"beat_id": f"E{n:02d}.B003", "hook": 5, "hook_type": "violence", "cliff": 1, "spoiler": 0,
             "reason": "drink thrown in public"},
            {"beat_id": f"E{n:02d}.B005", "hook": 3, "hook_type": "Shocking Line", "cliff": 4,
             "open_question": "Who is the other woman?", "resolved_by": None, "needs_context": "they are dating",
             "reason": "question at peak"},
            {"beat_id": f"E{n:02d}.B002", "hook": 1, "cliff": 1, "reason": "setup"},   # not notable: dropped
        ],
        "opening_hook": {"beat_id": f"E{n:02d}.B001", "score": 2, "note": "establishing"},
        "ending_cliffhanger": {"beat_id": f"E{n:02d}.B006", "score": 4, "note": "stranger arrives"},
        "ending_open_question": "Who is she?",
        "pickup": {"status": "not_applicable", "first_new_beat_id": None, "note": ""},
    }
    base.update(over)
    return base


class FakeClient(LLMClient):
    """Answers from a per-episode queue of replies (str or dict); records every prompt it saw."""
    provider = "fake"

    def __init__(self, replies: dict[int, list]):
        super().__init__("canned")
        self.replies = {k: list(v) for k, v in replies.items()}
        self.seen: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def _call(self, system, user, json_mode):
        n = int(re.match(r"EPISODE (\d+)", user).group(1))
        with self._lock:
            self.seen.append((system, user))
            r = self.replies[n].pop(0) if len(self.replies[n]) > 1 else self.replies[n][0]
        return (r if isinstance(r, str) else json.dumps(r)), 100, 50


def ep2_reply(**over):
    fields = dict(beats=[{"beat_id": "E02.B002", "hook": 5, "hook_type": "reveal", "cliff": 3,
                          "open_question": "Did Derek know?", "resolved_by": "E02.B006", "reason": "bombshell"}],
                  ending_cliffhanger={"beat_id": "E02.B006", "score": 3, "note": ""},
                  pickup={"status": "immediate", "first_new_beat_id": "B2", "note": "continues the rooftop"})
    return reply(EP2, **{**fields, **over})


def test_order_prompts_and_code_filled_fields():
    client = FakeClient({1: [reply(EP1)], 2: [ep2_reply()]})
    log = RunLog()
    seen_progress = []
    out = analyze.analyze_show([EP1, EP2], client, log=log, concurrency=2,
                               progress=lambda m, f: seen_progress.append(f))
    assert [a.episode for a in out] == [1, 2]
    a1, a2 = out
    # notable beats only, in time order; hook_type and needs_context normalized
    assert [b.beat_id for b in a1.beats] == ["E01.B003", "E01.B005"]
    assert a1.beats[1].hook_type == "shocking_line" and a1.beats[1].needs_context == ["they are dating"]
    # pickup: episode 1 is not_applicable; episode 2 id "B2" normalized, delay filled from the script
    assert a1.pickup.status == "not_applicable" and a1.pickup.delay_s is None
    assert a2.pickup.first_new_beat_id == "E02.B002" and a2.pickup.delay_s == 3.0
    assert a1.model == "fake:canned" and a1.prompt_version.startswith("analyze_system@")
    # same system prompt for every episode (provider-side caching)
    assert len({s for s, _ in client.seen}) == 1
    users = {int(re.match(r"EPISODE (\d+)", u).group(1)): u for _, u in client.seen}
    assert "CAMERA: Close-up on a stranger" in users[1]               # camera lines are included
    assert "PREVIOUS EPISODE: not provided" in users[1]
    assert "PREVIOUS EPISODE 1, LAST 25 SECONDS" in users[2]
    assert "E01.B006" in users[2] and "E01.B001" not in users[2]      # only the tail of episode 1
    assert sorted(seen_progress) == [0.5, 1.0]
    assert {r["item"] for r in log.rows} == {"E01", "E02"} and all("error" not in r for r in log.rows)


def test_invalid_reply_is_re_asked_with_a_precise_error():
    bad = reply(EP1, beats=[{"beat_id": "E01.B099", "hook": 4, "cliff": 0},
                            {"beat_id": "E01.B005", "hook": 2, "cliff": 4, "resolved_by": "E01.B004"}])
    client = FakeClient({1: [bad, reply(EP1)]})
    log = RunLog()
    [a] = analyze.analyze_show([EP1], client, log=log)
    assert len(client.seen) == 2
    retry = client.seen[1][1]
    assert "REJECTED" in retry and "'E01.B099' does not exist" in retry
    assert "resolved_by E01.B004 must be a LATER beat" in retry
    assert [b.beat_id for b in a.beats] == ["E01.B003", "E01.B005"]
    assert log.rows[0]["attempts"] == 2


def test_strict_check_names_every_problem():
    with pytest.raises(ValueError, match="from Episode 1, not this one"):
        analyze.coerce_analysis(ep2_reply(pickup={"status": "immediate", "first_new_beat_id": "E01.B006"}),
                                EP2, has_prev=True, strict=True)
    with pytest.raises(ValueError, match="is not in the first 3 s"):
        analyze.coerce_analysis(reply(EP1, opening_hook={"beat_id": "E01.B004", "score": 4}), EP1,
                                has_prev=False, strict=True)
    long_ep = make_episode(1, [("dialogue", f"line {i}", "A") for i in range(1, 21)])   # 60 s
    with pytest.raises(ValueError, match="is not in the final"):
        analyze.coerce_analysis(reply(long_ep, ending_cliffhanger={"beat_id": "E01.B002", "score": 3}), long_ep,
                                has_prev=False, strict=True)
    with pytest.raises(ValueError, match="use immediate"):
        analyze.coerce_analysis(ep2_reply(pickup={"status": "not_applicable"}), EP2, has_prev=True, strict=True)
    with pytest.raises(ValueError, match="first_new_beat_id is required"):
        analyze.coerce_analysis(ep2_reply(pickup={"status": "delayed"}), EP2, has_prev=True, strict=True)


def test_lenient_check_fixes_and_records_warnings():
    a = analyze.coerce_analysis(ep2_reply(pickup={"status": "immediate", "first_new_beat_id": "E01.B006"}),
                                EP2, has_prev=True)
    assert a.pickup.first_new_beat_id is None and any("first_new_beat_id" in w for w in a.warnings)
    a = analyze.coerce_analysis(reply(EP1, opening_hook={"beat_id": "E01.B004", "score": 4}), EP1, has_prev=False)
    assert a.opening_hook.beat_id in {b.id for b in EP1.beats if b.start < 3.0}
    assert any("opening_hook moved" in w for w in a.warnings)
    a = analyze.coerce_analysis(ep2_reply(pickup={"status": "not_applicable"}), EP2, has_prev=True)
    assert a.pickup.status == "unrelated"
    bad = reply(EP1, beats=[{"beat_id": "E01.B099", "hook": 4, "cliff": 0},
                            {"beat_id": "E01.B003", "hook": 9, "cliff": "x"}])
    a = analyze.coerce_analysis(bad, EP1, has_prev=False)
    assert [(b.beat_id, b.hook, b.cliff) for b in a.beats] == [("E01.B003", 5, 0)]   # clamped, bad id dropped
    with pytest.raises(ValueError, match="most beat ids are invalid"):
        analyze.coerce_analysis(reply(EP1, beats=[{"beat_id": "E07.B001", "hook": 4}]), EP1, has_prev=False)


def test_failed_episode_becomes_placeholder_and_run_continues():
    client = FakeClient({1: ["not json at all"], 2: [ep2_reply()]})
    log = RunLog()
    a1, a2 = analyze.analyze_show([EP1, EP2], client, log=log, concurrency=1)
    assert a1.summary.startswith("analysis failed:") and a1.beats == [] and a1.opening_hook.score == 0
    assert a1.model == "failed:fake:canned" and a2.model == "fake:canned"
    assert a2.beats and a2.pickup.status == "immediate"
    errors = [r for r in log.rows if "error" in r]
    assert [r["item"] for r in errors] == ["E01"]
    assert len([s for s in client.seen if "EPISODE 1 " in s[1]]) == 3   # 1 try + 2 re-asks


def test_previous_episode_missing_from_input_means_not_applicable():
    client = FakeClient({2: [ep2_reply()]})
    [a2] = analyze.analyze_show([EP2], client, log=RunLog())
    assert a2.pickup.status == "not_applicable" and a2.pickup.first_new_beat_id is None
    assert "PREVIOUS EPISODE: not provided" in client.seen[0][1]


def test_partial_re_ask_reply_falls_back_to_first_reply():
    # first reply breaks a fixable rule (strict re-ask); the "fixed" replies only contain the corrected field
    first = reply(EP1, opening_hook={"beat_id": "E01.B004", "score": 4})
    partial = '{"opening_hook": {"beat_id": "E01.B001", "score": 4}}'
    client = FakeClient({1: [first, partial, partial]})
    log = RunLog()
    [a] = analyze.analyze_show([EP1], client, log=log)
    assert a.beats and not a.summary.startswith("analysis failed")
    assert any("kept the first reply" in w for w in a.warnings)
    assert log.rows[0].get("recovered") is True
