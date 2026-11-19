"""Turn a microdrama screenplay (.txt) into an Episode of timestamped beats.

Pure code, no model. Tolerant of label case (episode 6 upper-cases its scene
labels), multi-line text, and unknown block labels; every oddity becomes a
warning instead of a crash.
"""
from __future__ import annotations

import re
from pathlib import Path

from .schema import Beat, Episode, Scene

_TS = r"(\d{1,2}):(\d{1,2}(?:\.\d+)?)"
RE_TIME = re.compile(r"^\s*start:\s*\(\s*" + _TS + r"\s*\)\s*\|\s*end:\s*\(\s*" + _TS + r"\s*\)\s*$", re.I)
RE_LABEL = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
RE_FIELD = re.compile(
    r"^\s*(episode number|episode logline|continuity notes|scene heading|scene start|scene end)\s*:\s*(.*)$", re.I
)
RE_SCENE = re.compile(r"^\s*scene\s+(\d+)\s*$", re.I)
RE_CLOCK = re.compile(r"^\s*" + _TS + r"\s*$")
RE_SKIP = re.compile(r"^\s*(---\s*script start\s*---|actions and dialogue\s*:?)\s*$", re.I)

KIND_BY_LABEL = {
    "CAMERA": "camera", "SHOT": "camera", "INSERT": "camera",
    "ACTION": "action",
    "SOUND FX": "sfx", "SOUND": "sfx", "SFX": "sfx", "MUSIC": "sfx",
    "TEXT ON SCREEN": "other", "TITLE": "other", "SUPER": "other",
}
VO_MARKERS = ("V.O", "VOICE OVER", "VOICE-OVER", "NARRATOR", "INNER VOICE", "VOICEOVER")  # O.S. stays dialogue


def _secs(m: str, s: str) -> float:
    return round(int(m) * 60 + float(s), 3)


def _clock(text: str) -> float | None:
    m = RE_CLOCK.match(text)
    return _secs(m.group(1), m.group(2)) if m else None


def _label(inner: str) -> tuple[str, str | None, str | None]:
    """'LADY MARA (Confused, Breathless):' -> ('dialogue', 'LADY MARA', 'Confused, Breathless').
    'MATRIARCH (O.S.) (Firm):' -> ('dialogue', 'MATRIARCH', 'O.S.; Firm'). V.O./narrator -> 'vo'."""
    inner = inner.strip().rstrip(":").strip()
    name = re.split(r"\s*\(", inner, maxsplit=1)[0].strip()
    groups = [g.strip() for g in re.findall(r"\(([^()]*)\)", inner) if g.strip()]
    delivery = "; ".join(groups) or None
    kind = KIND_BY_LABEL.get(name.upper())
    if kind:
        return kind, None, None
    tag = f"{name} {delivery or ''}".upper()
    return ("vo" if any(v in tag for v in VO_MARKERS) else "dialogue"), name, delivery


def parse_text(text: str, source_file: str = "upload.txt") -> Episode:
    warnings: list[str] = []
    header: dict[str, str] = {}
    scenes: list[dict] = []
    blocks: list[dict] = []
    current: dict | None = None
    last_field: str | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or RE_SKIP.match(line):
            last_field = None if not line.strip() else last_field
            continue
        if m := RE_SCENE.match(line):
            scenes.append({"number": int(m.group(1)), "heading": "", "start": None, "end": None})
            current, last_field = None, None
            continue
        if m := RE_FIELD.match(line):
            key, val = m.group(1).lower(), m.group(2).strip()
            if key.startswith("scene"):
                if not scenes:
                    scenes.append({"number": 1, "heading": "", "start": None, "end": None})
                    warnings.append("scene field before any SCENE line; assumed scene 1")
                sc = scenes[-1]
                if key == "scene heading":
                    sc["heading"] = val
                else:
                    t = _clock(val)
                    if t is None:
                        warnings.append(f"unreadable {key} '{val}' in scene {sc['number']}")
                    sc["start" if key == "scene start" else "end"] = t
            else:
                header[key] = val
                last_field = key
            current = None
            continue
        if m := RE_LABEL.match(line):
            kind, speaker, delivery = _label(m.group(1))
            current = {"kind": kind, "speaker": speaker, "delivery": delivery, "start": None, "end": None,
                       "text": [], "scene": scenes[-1]["number"] if scenes else 1, "label": m.group(1)}
            blocks.append(current)
            last_field = None
            continue
        if m := RE_TIME.match(line):
            if current is None:
                warnings.append(f"timestamp with no block label: '{line.strip()}'")
                continue
            current["start"], current["end"] = _secs(m.group(1), m.group(2)), _secs(m.group(3), m.group(4))
            continue
        if current is not None:
            current["text"].append(line.strip())
        elif last_field in ("episode logline", "continuity notes"):
            header[last_field] += " " + line.strip()   # wrapped header text
        else:
            warnings.append(f"ignored line: '{line.strip()[:60]}'")

    # episode number: header first, filename second
    number = None
    if m := re.search(r"(\d+)", header.get("episode number", "")):
        number = int(m.group(1))
    name = Path(source_file).name
    if number is None and (m := re.search(r"episode[-_ ]?0*(\d+)", name, re.I)):
        number = int(m.group(1))
        warnings.append("episode number taken from filename")
    if number is None:
        number = 0
        warnings.append("no episode number found")
    source_id = m.group(1) if (m := re.search(r"Episode (\d{4,})", name)) else None

    scene_models = []
    for sc in scenes:
        if sc["start"] is None or sc["end"] is None:
            warnings.append(f"scene {sc['number']} missing start/end")
            continue
        scene_models.append(Scene(number=sc["number"], heading=sc["heading"], start=sc["start"], end=sc["end"]))

    raw_beats = []
    for i, b in enumerate(blocks):
        text_ = " ".join(b["text"]).strip()
        if b["start"] is None:
            warnings.append(f"block '{b['label']}' has no timestamp; dropped")
            continue
        if b["end"] < b["start"]:
            warnings.append(f"block '{b['label']}' ends before it starts ({b['start']}>{b['end']}); swapped")
            b["start"], b["end"] = b["end"], b["start"]
        if b["kind"] in ("dialogue", "vo"):
            text_ = text_.strip()
        sc = next((s for s in scene_models if s.number == b["scene"]), None)
        if sc and (b["start"] < sc.start - 0.05 or b["end"] > sc.end + 0.05):
            warnings.append(
                f"{b['kind']} block at {b['start']:.2f}-{b['end']:.2f} runs outside scene {sc.number} "
                f"({sc.start:.2f}-{sc.end:.2f}); kept as written"
            )
        raw_beats.append((b["start"], i, b, text_))

    raw_beats.sort(key=lambda x: (x[0], x[1]))
    beats = [
        Beat(id=f"E{number:02d}.B{j:03d}", episode=number, scene=b["scene"], kind=b["kind"],
             speaker=b["speaker"], delivery=b["delivery"], text=t, start=s, end=b["end"])
        for j, (s, _, b, t) in enumerate(raw_beats, start=1)
    ]
    duration = max([s.end for s in scene_models] + [b.end for b in beats] + [0.0])
    if not beats:
        warnings.append("no timestamped beats found")
    return Episode(number=number, source_file=name, source_id=source_id,
                   logline=header.get("episode logline", ""), continuity_notes=header.get("continuity notes", ""),
                   scenes=scene_models, beats=beats, duration=round(duration, 3), warnings=warnings)


def parse_file(path: str | Path) -> Episode:
    p = Path(path)
    return parse_text(p.read_text(encoding="utf-8", errors="replace"), p.name)


def parse_folder(folder: str | Path) -> list[Episode]:
    eps = [parse_file(p) for p in sorted(Path(folder).glob("*.txt"))]
    eps.sort(key=lambda e: e.number)
    seen: set[int] = set()
    for e in eps:
        if e.number in seen:
            e.warnings.append(f"duplicate episode number {e.number}")
        seen.add(e.number)
    return eps
