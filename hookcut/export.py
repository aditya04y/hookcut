"""Write a run to disk: JSON/CSV/Markdown for people, and editor timelines (OTIO, CMX3600 EDL, FCP7 XML).

Timelines assume one video file per episode named E01.mp4, E02.mp4, ... at FPS frames per second
(see editor/README.txt). If OpenTimelineIO is missing, a hand-written CMX3600 EDL is used instead.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from . import candidates as cg
from .report import UNLOCK_EPISODES, season_summary
from .schema import Promo, RunResult, Segment
from .textfmt import ep_tc, tc

FPS = 30
WAVES = {
    1: ("Wave 1: cold audiences (post first)",
        "Early-season moments a stranger can follow (cold-viewer standalone >= 4) that give nothing away "
        "(spoiler <= 1): they sell the premise to people who have never heard of the show."),
    2: ("Wave 2: escalation (post next)",
        "Strong conflict from across the season once the premise is out; later in the story or needs a "
        "little context, so it lands better on people who saw wave 1."),
    3: ("Wave 3: big reveals (warm audiences / retargeting)",
        "Spoiler >= 2: these give a twist away, so they go to people who already watched or engaged, "
        "to pull them back for the payoff."),
}


# ----------------------------------------------------------------------------- formatting
def segments_str(segments: list[Segment]) -> str:
    """'E01 00:39.00–01:23.70; E02 00:05.20–00:21.00'"""
    return "; ".join(f"E{s.episode:02d} {tc(s.start)}–{tc(s.end)}" for s in segments)


def frames(t: float, fps: int = FPS) -> int:
    return int(round(t * fps))


def timecode(n_frames: int, fps: int = FPS) -> str:
    """Frame count -> non-drop SMPTE 'HH:MM:SS:FF'."""
    s, f = divmod(max(0, n_frames), fps)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def reel(episode: int) -> str:
    return f"E{episode:02d}"[:8]      # CMX3600 reel names are at most 8 characters


def still_frame(result: RunResult, p: Promo) -> str:
    """Static-image pick: inside the camera shot covering the hook beat, ~0.5 s after the hook starts."""
    cand = next((c for c in result.candidates if c.id == p.candidate_id), None)
    for e in result.episodes if cand else []:
        try:
            return ep_tc(e.number, cg.still_frame_time(e, e.beat(cand.hook_beat_id)))
        except KeyError:
            continue
    return ""


def _md(text: str) -> str:
    return " ".join(str(text or "").split()).replace("|", "\\|")


def _cold(p: Promo) -> str:
    v = p.cold_viewer
    if v is None:
        return "not tested"
    lands = "hook lands" if v.hook_lands else "hook does not land"
    return f"{v.verdict} (standalone {v.standalone}/5, keep watching {v.keep_watching}/5, {lands})"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- tables
def promo_rows(result: RunResult) -> list[dict]:
    return [{
        "rank": p.rank, "wave": p.post_wave, "candidate_id": p.candidate_id, "kind": p.kind,
        "segments": segments_str(p.segments), "duration_s": round(p.duration, 2),
        "hook_time": p.hook_time, "hook_text": p.hook_text, "still_frame_time": still_frame(result, p),
        "cliffhanger_time": p.cliffhanger_time, "cliffhanger_text": p.cliffhanger_text,
        "open_question": p.open_question, "why_it_works": p.why_it_works,
        "cold_viewer": _cold(p), "cold_viewer_premise": p.cold_viewer.premise if p.cold_viewer else "",
        "flags": "; ".join(p.flags), "final_score": p.scores.get("final", ""),
        "variants": "; ".join(f"{v['label']} ({v['duration']:.1f} s)" for v in p.variants),
    } for p in result.promos]


def candidate_rows(result: RunResult) -> list[dict]:
    verdicts = {v.candidate_id: v for v in result.verdicts}
    promos = {p.candidate_id: p for p in result.promos}
    beats = cg.beat_index(result.episodes)
    rows = []
    for c in result.candidates:
        v, p = verdicts.get(c.id), promos.get(c.id)
        if p:
            status, reasons = f"promo #{p.rank} (wave {p.post_wave})", ""
        elif v and v.verdict == "fail":
            status, reasons = "rejected by cold viewer", "; ".join(v.confusions) or v.premise
        elif v:
            status = "passed cold viewer, not selected"
            reasons = "lost on score, diversity, per-episode cap or time overlap"
        else:
            status, reasons = "not cold-viewer tested", "below judge_top_k"
        h, k = beats.get(c.hook_beat_id), beats.get(c.cliff_beat_id)
        rows.append({
            "candidate_id": c.id, "kind": c.kind, "segments": segments_str(c.segments),
            "duration_s": round(c.duration, 2), "hook_beat_id": c.hook_beat_id,
            "hook_time": ep_tc(h.episode, h.start) if h else "", "hook_score": c.hook_score,
            "hook_type": c.hook_type or "", "cliff_beat_id": c.cliff_beat_id,
            "cliff_time": ep_tc(k.episode, k.start) if k else "", "cliff_score": c.cliff_score,
            "open_question": c.open_question, "spoiler": c.spoiler, "needs_context": "; ".join(c.needs_context),
            "prelim_score": c.prelim_score, "verdict": v.verdict if v else "",
            "standalone": v.standalone if v else "", "keep_watching": v.keep_watching if v else "",
            "hook_lands": v.hook_lands if v else "", "status": status, "reasons": reasons,
            "notes": "; ".join(c.notes),
        })
    return rows


def report_rows(result: RunResult) -> list[dict]:
    return [dict(r, flags="; ".join(r.get("flags", []))) for r in result.episode_report]


# ----------------------------------------------------------------------------- markdown
def promos_md(result: RunResult) -> str:
    out = [f"# {result.show}: promo shortlist", "",
           f"{len(result.promos)} promos from {len(result.episodes)} episodes "
           f"({len(result.candidates)} candidates, {len(result.verdicts)} cold-viewer tests). "
           f"Generated {result.created_at}. Times are episode-relative (MM:SS.ss).", ""]
    for wave, (title, why) in WAVES.items():
        ps = [p for p in result.promos if p.post_wave == wave]
        if not ps:
            continue
        out += [f"## {title}", "", f"*{why}*", ""]
        for p in ps:
            out += [f"### #{p.rank} · {p.candidate_id} · {p.duration:.1f} s · {p.kind}", "",
                    f"- **Cut:** {segments_str(p.segments)}",
                    f"- **Hook** ({p.hook_time}): {_md(p.hook_text)}",
                    f"- **Still frame:** {still_frame(result, p)}",
                    f"- **Cliffhanger** ({p.cliffhanger_time}): {_md(p.cliffhanger_text)}",
                    f"- **Open question:** {_md(p.open_question)}",
                    f"- **Why it works:** {_md(p.why_it_works) or '(not written)'}",
                    f"- **Cold viewer:** {_cold(p)}"
                    + (f": {_md(p.cold_viewer.premise)}" if p.cold_viewer else "")]
            if p.flags:
                out.append(f"- **Flags:** {_md('; '.join(p.flags))}")
            if p.variants:
                out.append("- **Variants:** " + "; ".join(
                    f"{v['label']} ({segments_str([Segment(**s) for s in v['segments']])}, {v['duration']:.1f} s)"
                    for v in p.variants))
            out.append("")
    if not result.promos:
        out.append("No promo passed the checks.")
    return "\n".join(out) + "\n"


def episode_report_md(result: RunResult) -> str:
    rows = result.episode_report
    s = season_summary(rows)

    def eps(items: list[dict]) -> str:
        return ", ".join(f"E{i['episode']:02d} ({i['score']})" for i in items) or "none"

    out = [f"# {result.show}: episode hooks and cliffhangers", "", "## Season summary", "",
           f"- Episodes analysed: {s['analyzed']} of {s['episodes']}",
           f"- Average opening hook: {s['avg_opening_hook']}/5; average ending cliffhanger: {s['avg_ending_cliff']}/5",
           f"- Weakest openings: {eps(s['weakest_openings'])}",
           f"- Weakest endings: {eps(s['weakest_endings'])}",
           "- Candidate unlock / paywall points (strongest cliffhangers in episodes "
           f"{UNLOCK_EPISODES[0]}–{UNLOCK_EPISODES[1]}): "
           + ("; ".join(f"E{u['episode']:02d} ({u['score']}): {_md(u['open_question'])}"
                        for u in s["unlock_candidates"]) or "none"),
           f"  - Assumption: {s['unlock_assumption']}",
           "- Flags: " + (", ".join(f"{k} × {v}" for k, v in s["flag_counts"].items()) or "none"),
           "", "## Episodes", "",
           "| Ep | Length | Opening hook | Ending cliffhanger | Open question | Pickup of previous | Flags |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        hook = (f"{r['opening_hook_score']} @ {r['opening_hook_time']}: {_md(r['opening_hook_text'])}"
                if r["opening_hook_score"] is not None else "–")
        end = (f"{r['ending_cliff_score']} @ {r['ending_cliff_time']}: {_md(r['ending_cliff_text'])}"
               if r["ending_cliff_score"] is not None else "–")
        pick = r["pickup_status"] + (f" ({r['pickup_delay_s']} s)" if r["pickup_delay_s"] is not None else "")
        out.append(f"| E{r['episode']:02d} | {tc(r['duration_s'])} | {hook} | {end} | "
                   f"{_md(r['ending_open_question'])} | {pick or '–'} | {_md('; '.join(r['flags']))} |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- editor
def edl_text(title: str, segments: list[Segment], fps: int = FPS) -> str:
    """Hand-written CMX3600 EDL (fallback when OpenTimelineIO is unavailable)."""
    lines, rec = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""], 0
    for n, s in enumerate(segments, start=1):
        a, b = frames(s.start, fps), frames(s.end, fps)
        lines.append(f"{n:03d}  {reel(s.episode):<8} V     C        {timecode(a, fps)} {timecode(b, fps)} "
                     f"{timecode(rec, fps)} {timecode(rec + b - a, fps)}")
        lines.append(f"* FROM CLIP NAME: {reel(s.episode)}.mp4")
        rec += b - a
    return "\n".join(lines) + "\n"


def _timeline(otio, title: str, segments: list[Segment], durations: dict[int, float], fps: int = FPS):
    rt, tr = otio.opentime.RationalTime, otio.opentime.TimeRange
    tl = otio.schema.Timeline(name=title)
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    for s in segments:
        a, b = frames(s.start, fps), frames(s.end, fps)
        ref = otio.schema.ExternalReference(
            target_url=f"{reel(s.episode)}.mp4",
            available_range=tr(rt(0, fps), rt(frames(durations.get(s.episode, s.end), fps), fps)))
        clip = otio.schema.Clip(name=reel(s.episode), media_reference=ref,
                                source_range=tr(rt(a, fps), rt(b - a, fps)))
        clip.metadata["cmx_3600"] = {"reel": reel(s.episode)}
        clip.metadata["hookcut"] = {"first_beat_id": s.first_beat_id, "last_beat_id": s.last_beat_id}
        track.append(clip)
    return tl


EDITOR_README = """HookCut editor handoff
=======================

One timeline per ranked promo, in three formats:
  promo_NN.otio  OpenTimelineIO (DaVinci Resolve 18.5+, or convert with otioconvert)
  promo_NN.edl   CMX3600 EDL (Premiere, Resolve, Avid; reel names E01, E02, ...)
  promo_NN.xml   Final Cut Pro 7 XML (Premiere Pro, Resolve)

Assumptions (check before conforming):
  * one video file per episode, named by episode number: E01.mp4, E02.mp4, ... (relink if yours differ);
  * {fps} fps, non-drop timecode, each file starting at 00:00:00:00 (screenplay time 0 = first frame);
  * cut points come from the screenplay timestamps, rounded to the nearest frame; the finished video
    may drift from the script, so trim by eye at each cut;
  * video track only: link the audio from the same source ranges.

Quick cut without an NLE (per segment, then concatenate):
  ffmpeg -ss <start s> -to <end s> -i E01.mp4 -c:v libx264 -c:a aac seg1.mp4

promos.csv next to this folder lists the same cuts with hooks, cliffhangers and flags.
"""


def write_editor(result: RunResult, folder: Path) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    try:
        import opentimelineio as otio
    except Exception:
        otio = None
    durations = {e.number: e.duration for e in result.episodes}
    for p in result.promos:
        name = f"promo_{p.rank:02d}"
        title = f"{name} {p.candidate_id}"
        edl: Optional[str] = None
        if otio is not None:
            tl = _timeline(otio, title, p.segments, durations)
            otio.adapters.write_to_file(tl, str(folder / f"{name}.otio"))
            paths[f"{name}.otio"] = str(folder / f"{name}.otio")
            try:
                edl = otio.adapters.write_to_string(tl, "cmx_3600")
            except Exception:
                edl = None
            try:
                (folder / f"{name}.xml").write_text(otio.adapters.write_to_string(tl, "fcp_xml"), encoding="utf-8")
                paths[f"{name}.xml"] = str(folder / f"{name}.xml")
            except Exception:
                pass
        (folder / f"{name}.edl").write_text(edl or edl_text(title, p.segments), encoding="utf-8")
        paths[f"{name}.edl"] = str(folder / f"{name}.edl")
    readme = EDITOR_README.format(fps=FPS)
    if otio is None:
        readme += "\nOpenTimelineIO was not installed for this run: only hand-written EDLs were produced.\n"
    (folder / "README.txt").write_text(readme, encoding="utf-8")
    paths["editor_readme"] = str(folder / "README.txt")
    return paths


# ----------------------------------------------------------------------------- main
def write_outputs(result: RunResult, out_dir) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def put(name: str, text: str) -> None:
        (out / name).write_text(text, encoding="utf-8")
        paths[name] = str(out / name)

    put("result.json", result.model_dump_json(indent=2))
    put("promos.json", json.dumps([p.model_dump() for p in result.promos], indent=2, ensure_ascii=False))
    for name, rows in (("promos.csv", promo_rows(result)), ("all_candidates.csv", candidate_rows(result)),
                       ("episode_report.csv", report_rows(result))):
        _write_csv(out / name, rows)
        paths[name] = str(out / name)
    put("promos.md", promos_md(result))
    put("episode_report.md", episode_report_md(result))
    paths.update({f"editor/{k}": v for k, v in write_editor(result, out / "editor").items()})
    return paths
