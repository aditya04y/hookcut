"""Audience evidence at promo-cut grain, usable with a saved RunResult and no model calls.

Each row is one *settled impression*, including immediate skips. watch_seconds is
continuous first-play time from the opening. A continuation is an attributed open
of the show from the end card after reaching the end of the promo. An uploader
must use the same attribution window for every impression, including the skips.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields

from .schema import Candidate, RunResult


@dataclass(frozen=True)
class Exposure:
    impression_id: str
    viewer_id: str             # A pseudonymous ID, stable across this export.
    cohort: str                # Audience at the time of the impression.
    clip_key: str              # Identifies the cut across reruns.
    watch_seconds: float
    continued: bool
    source: str = "reported"


@dataclass
class ClipStats:
    impressions: int = 0
    held_3s: int = 0
    completed: int = 0
    continued: int = 0

    @property
    def hold_rate(self) -> float | None:
        return self.held_3s / self.impressions if self.impressions else None

    @property
    def completion_rate(self) -> float | None:
        return self.completed / self.impressions if self.impressions else None

    @property
    def continuation_rate(self) -> float | None:
        return self.continued / self.completed if self.completed else None

    def observe(self, event: Exposure, duration: float) -> None:
        self.impressions += 1
        self.held_3s += event.watch_seconds >= 3
        self.completed += event.watch_seconds >= duration
        self.continued += event.continued


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def clip_keys(result: RunResult) -> dict[str, str]:
    """Bind telemetry to the show, ordered cuts and script played in those cuts.

    Candidate IDs are reassigned on every ranked run. A title change deliberately
    creates a new namespace because the current input format has no immutable show ID.
    Changes to model scores, candidate order or unrelated episodes preserve the key.
    """
    beats = {episode.number: [
        (b.start, b.end, _digest([b.kind, b.speaker, b.delivery, b.text, b.start, b.end]))
        for b in episode.beats] for episode in result.episodes}
    keys = {}
    for candidate in result.candidates:
        cuts = []
        for seg in candidate.segments:
            played = [digest for start, end, digest in beats.get(seg.episode, [])
                      if start < seg.end and end > seg.start]
            cuts.append([seg.episode, seg.start, seg.end, played])
        keys[candidate.id] = "cut-" + _digest([result.show, cuts])
    return keys


_COLUMNS = tuple(field.name for field in fields(Exposure))


def events_csv(events: Iterable[Exposure]) -> str:
    """Write the upload format, preserving simulated provenance on round trip."""
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(_COLUMNS)
    for e in events:
        writer.writerow([e.impression_id, e.viewer_id, e.cohort, e.clip_key,
                         e.watch_seconds, int(e.continued), e.source])
    return out.getvalue()


def _csv_rows(reader: csv.DictReader) -> Iterator[Exposure]:
    for row in reader:
        try:
            if None in row:
                raise ValueError("too many columns")
            cells = {name: (value or "").strip() for name, value in row.items()}
            flag = cells["continued"].lower()
            if flag not in ("0", "1", "false", "true"):
                raise ValueError("continued must be 0 or 1")
            yield Exposure(cells["impression_id"], cells["viewer_id"], cells["cohort"],
                           cells["clip_key"], float(cells["watch_seconds"]),
                           flag in ("1", "true"), cells.get("source", "reported").lower())
        except ValueError as exc:
            raise ValueError(f"CSV row {reader.line_num}: {exc}") from exc


def _checked(events: Iterable[Exposure], durations: dict[str, float]) -> Iterator[Exposure]:
    seen: dict[str, Exposure] = {}
    source = None
    for row, event in enumerate(events, start=2):
        message = ""
        if not all((event.impression_id, event.viewer_id, event.cohort)):
            message = "each impression needs an impression_id, viewer_id and cohort"
        elif event.clip_key not in durations:
            message = "unknown clip_key for this show's cuts"
        elif not math.isfinite(event.watch_seconds) or not 0 <= event.watch_seconds <= durations[event.clip_key]:
            message = "watch_seconds must be finite and between zero and the clip duration"
        elif event.continued and event.watch_seconds < durations[event.clip_key]:
            message = "an end-card continuation requires the complete promo"
        elif event.source not in ("synthetic", "reported") or (source and event.source != source):
            message = "use one source per file (synthetic or reported)"
        elif event.impression_id in seen and event != seen[event.impression_id]:
            message = "conflicting rows for one impression_id"
        if message:
            raise ValueError(f"Impression row {row}: {message}")
        if event.impression_id not in seen:
            source = event.source
            seen[event.impression_id] = event
            yield event


def _durations(result: RunResult, keys: dict[str, str]) -> dict[str, float]:
    return {keys[c.id]: c.duration for c in result.candidates}


def read_events_csv(text: str, result: RunResult, *, keys: dict[str, str] | None = None) -> list[Exposure]:
    """Import settled impressions. Repeated delivery of an identical row counts once."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff"), newline=""), strict=True)
    try:
        columns = reader.fieldnames or []
        missing = set(_COLUMNS[:-1]) - set(columns)
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate CSV column")
        if keys is None:
            keys = clip_keys(result)
        return list(_checked(_csv_rows(reader), _durations(result, keys)))
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV near row {reader.line_num}: {exc}") from exc


@dataclass
class AudienceActivity:
    representative: dict[str, str]
    counts: dict[str, ClipStats]
    cohorts_by_cut: dict[str, dict[str, ClipStats]]
    viewer_cells: dict[str, dict[str, ClipStats]]


def aggregate(result: RunResult, events: Iterable[Exposure], cohort: str | None = None, *,
              keys: dict[str, str] | None = None) -> AudienceActivity:
    """Count impressions once per physical edit, even when it has multiple annotations.

    The first candidate for an edit represents its aliases. Cohort totals include
    every cohort. Counts and sparse viewer cells reflect the selected cohort.
    """
    if keys is None:
        keys = clip_keys(result)
    by_key: dict[str, str] = {}
    representative = {c.id: by_key.setdefault(keys[c.id], c.id) for c in result.candidates}
    durations = _durations(result, keys)
    counts = {cid: ClipStats() for cid in by_key.values()}
    cohorts_by_cut: dict[str, dict[str, ClipStats]] = {}
    viewer_cells: dict[str, dict[str, ClipStats]] = {}
    for event in _checked(events, durations):
        cid = by_key[event.clip_key]
        duration = durations[event.clip_key]
        cohort_counts = cohorts_by_cut.setdefault(event.cohort, {})
        cohort_counts.setdefault(cid, ClipStats()).observe(event, duration)
        if cohort is None or cohort == event.cohort:
            counts[cid].observe(event, duration)
            cells = viewer_cells.setdefault(event.viewer_id, {})
            cells.setdefault(cid, ClipStats()).observe(event, duration)
    return AudienceActivity(representative, counts, cohorts_by_cut, viewer_cells)


def activity_matrix(result: RunResult, events: Iterable[Exposure], cohort: str | None = None
                    ) -> dict[str, dict[str, ClipStats]]:
    """Sparse viewer × candidate matrix; missing cells represent no exposure."""
    activity = aggregate(result, events, cohort)
    aliases: dict[str, list[str]] = {}
    for cid, rep in activity.representative.items():
        aliases.setdefault(rep, []).append(cid)
    return {viewer: {cid: stats for rep, stats in cells.items() for cid in aliases[rep]}
            for viewer, cells in activity.viewer_cells.items()}


def summarize(result: RunResult, events: Iterable[Exposure], cohort: str | None = None) -> dict[str, ClipStats]:
    """Counts and rates by candidate, including the zero-support cuts."""
    activity = aggregate(result, events, cohort)
    return {cid: activity.counts[rep] for cid, rep in activity.representative.items()}


def _playback(candidate: Candidate, closing_kind: str, returning: bool,
              rng: random.Random) -> tuple[float, bool]:
    """Explicit response assumptions for the demo, independent of editorial ranks and scores."""
    favorite_hooks = {"mystery", "romance"} if returning else {"reveal", "shocking_line", "humiliation"}
    hold = 0.83 if candidate.hook_type in favorite_hooks else 0.49
    finish = 0.84 - 0.004 * max(0, candidate.duration - 25)
    if not returning and candidate.needs_context:
        finish -= 0.15
    # Returning viewers in this scenario want a spoken/visual mystery payoff.
    # New viewers more often open the show after a cut requiring little backstory.
    if returning:
        open_show = 0.70 if closing_kind in {"dialogue", "vo", "action"} else 0.42
        if candidate.hook_type == "mystery":
            open_show += 0.14
    else:
        open_show = 0.58 if not candidate.needs_context else 0.31
    if rng.random() >= hold:
        return round(rng.uniform(0, min(2.95, candidate.duration * 0.8)), 3), False
    if candidate.duration > 3.05 and rng.random() >= finish:
        return round(rng.uniform(3, candidate.duration - 0.05), 3), False
    return candidate.duration, rng.random() < open_show


def demo_events(result: RunResult, seed: int = 17, *, keys: dict[str, str] | None = None) -> list[Exposure]:
    """Reproducible simulation with uniform random exposure to launched cuts.

    Two cohorts have deliberately different response assumptions. Reserve at
    least one shortlisted cut with no history whenever there are two or more.
    This exercises cold start and preferential outcomes without claiming lift.
    """
    if keys is None:
        keys = clip_keys(result)
    catalog = {keys[c.id]: c for c in result.candidates}
    if not catalog:
        return []
    rng = random.Random(seed)
    launched = sorted(catalog)
    if len(launched) >= 2:
        shortlist = sorted(result.promos, key=lambda promo: promo.rank)
        eligible = sorted({keys[p.candidate_id] for p in shortlist[1:] if p.candidate_id in keys})
        if not eligible:
            eligible = sorted({keys[p.candidate_id] for p in shortlist if p.candidate_id in keys})
        reserved = rng.choice(eligible or launched)
        launched.remove(reserved)
    closing_kinds = {b.id: b.kind for ep in result.episodes for b in ep.beats}
    events = []
    for n in range(max(160, len(launched) * 12)):
        viewer = f"demo-v{n + 1:05d}"
        returning = bool(n % 2)
        cohort = "returning" if returning else "new"
        # Give each viewer up to four distinct cuts. Assignment ignores all scores.
        for key in rng.sample(launched, k=min(4, len(launched))):
            candidate = catalog[key]
            watched, continued = _playback(candidate, closing_kinds.get(candidate.cliff_beat_id, ""),
                                          returning, rng)
            events.append(Exposure(f"demo-{seed}-{n + 1}-{key}", viewer, cohort, key,
                                   watched, continued, "synthetic"))
    return events
