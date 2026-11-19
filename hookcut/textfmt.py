"""Shared formatting helpers: timecodes and the beat listing every prompt uses."""
from __future__ import annotations

from typing import Iterable

from .schema import Beat


def tc(seconds: float) -> str:
    """64.5 -> '01:04.50' (the screenplay's own format)."""
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def ep_tc(episode: int, seconds: float) -> str:
    return f"E{episode:02d} {tc(seconds)}"


def beat_line(b: Beat, offset: float = 0.0, with_id: bool = True, camera_chars: int = 140) -> str:
    """One line per beat, e.g.
    E01.B041 [00:39.10-00:41.20] LADY IN GREEN (Gloating, Smug): "He chose me, sister."
    offset shifts times (use clip start to show clip-relative times)."""
    t = f"[{tc(b.start - offset)}-{tc(b.end - offset)}]"
    head = f"{b.id} {t}" if with_id else t
    if b.kind in ("dialogue", "vo"):
        who = b.speaker or "VOICE"
        how = f" ({b.delivery})" if b.delivery else ""
        vo = " [V.O.]" if b.kind == "vo" and "V.O" not in who.upper() else ""
        return f"{head} {who}{how}{vo}: {b.text}"
    text = b.text if b.kind != "camera" or len(b.text) <= camera_chars else b.text[: camera_chars - 1] + "…"
    return f"{head} {b.kind.upper()}: {text}"


def render_beats(beats: Iterable[Beat], offset: float = 0.0, with_id: bool = True,
                 include_camera: bool = True, camera_chars: int = 140) -> str:
    return "\n".join(
        beat_line(b, offset, with_id, camera_chars)
        for b in beats
        if include_camera or b.kind != "camera"
    )
