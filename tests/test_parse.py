"""Parser tests on synthetic screenplay text (never the client's scripts)."""
from hookcut.parse import parse_text
from hookcut.textfmt import beat_line, tc

SAMPLE = """EPISODE NUMBER: Episode 3
EPISODE LOGLINE: Nora finds a stranger's ring in her husband's coat
and confronts him at dinner.
CONTINUITY NOTES: Follows the gala.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. KITCHEN – NIGHT
Scene start: 00:00.00
Scene end: 00:20.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Close-up on a ring in an open palm.

**NORA (Cold, Quiet):**
Start: (00:01.00) | End: (00:02.50)
"Whose is this?"

**HUSBAND (O.S.) (Calm):**
Start: (00:03.00) | End: (00:04.00)
"Put that down."

**NARRATOR (V.O.):**
Start: (00:04.00) | End: (00:06.00)
"She already knew."

**SOUND FX:**
Start: (00:06.00) | End: (00:07.00)
A plate shatters.

SCENE 02
SCENE HEADING: EXT. PORCH – NIGHT
SCENE START: 00:20.00
SCENE END: 00:40.00

ACTIONS AND DIALOGUE:

**ACTION:**
Start: (00:21.00) | End: (00:24.00)
Nora walks out,
slamming the door.
"""


def test_header_scenes_and_kinds():
    ep = parse_text(SAMPLE, "episode-003-Episode 1234567.txt")
    assert ep.number == 3 and ep.source_id == "1234567"
    assert "confronts him at dinner." in ep.logline          # wrapped header line joined
    assert [s.number for s in ep.scenes] == [1, 2]           # upper-case scene labels parsed
    assert ep.duration == 40.0
    kinds = [b.kind for b in ep.beats]
    assert kinds == ["camera", "dialogue", "dialogue", "vo", "sfx", "action"]
    assert ep.warnings == []


def test_speaker_and_delivery_parsing():
    ep = parse_text(SAMPLE, "x.txt")
    husband = ep.beats[2]
    assert husband.speaker == "HUSBAND" and husband.delivery == "O.S.; Calm"   # O.S. stays dialogue
    narrator = ep.beats[3]
    assert narrator.kind == "vo" and narrator.speaker == "NARRATOR"
    assert ep.beats[5].text == "Nora walks out, slamming the door."           # multi-line text joined


def test_ids_sorted_by_time_and_formatting():
    ep = parse_text(SAMPLE, "x.txt")
    assert [b.id for b in ep.beats][:2] == ["E03.B001", "E03.B002"]
    assert all(a.start <= b.start for a, b in zip(ep.beats, ep.beats[1:]))
    assert tc(64.5) == "01:04.50"
    line = beat_line(ep.beats[1])
    assert line.startswith("E03.B002 [00:01.00-00:02.50] NORA (Cold, Quiet):")


def test_oddities_become_warnings_not_crashes():
    text = SAMPLE.replace("Start: (00:06.00) | End: (00:07.00)\nA plate", "A plate")  # sfx without time
    text = text.replace("Start: (00:21.00) | End: (00:24.00)", "Start: (00:24.00) | End: (00:21.00)")
    ep = parse_text(text, "x.txt")
    assert any("no timestamp" in w for w in ep.warnings)
    assert any("swapped" in w for w in ep.warnings)
    assert len(ep.beats) == 5


def test_episode_number_from_filename():
    ep = parse_text(SAMPLE.replace("EPISODE NUMBER: Episode 3\n", ""), "episode-017-whatever.txt")
    assert ep.number == 17 and any("filename" in w for w in ep.warnings)
