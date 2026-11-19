"""List the run's likely bad picks for docs/failures.md (ids, timecodes, scores and flags; no script text).

  .venv/bin/python -m eval.mine_failures --run outputs/<run>/result.json [--eval eval/results/<run>/eval.json]

Flags: high analyzer score but a cold-viewer fail, the two judges disagree, hook is visual-only (no spoken
line in the first 3 s: a transcript judge under-rates it), needs context, spoiler, the answer plays inside
the clip, multi-segment (continuity risk), a boundary cuts a spoken line (camera/dialogue track overlap),
hook later than 3 s. A human reads the flagged clips and writes the one-line diagnosis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from hookcut.schema import Candidate, RunResult
from hookcut.textfmt import ep_tc

from .common import EPS, SPEECH, candidate_cut_beats, episode_map, load_run, results_dir, run_name

HOOK_WINDOW_S = 3.0


def _timecodes(c: Candidate) -> str:
    return " + ".join(f"{ep_tc(s.episode, s.start)}-{ep_tc(s.episode, s.end)[4:]}" for s in c.segments)


def mine(result: RunResult, eval_rep: Optional[dict] = None, top_quantile: float = 0.75) -> list[dict]:
    eps = episode_map(result.episodes)
    scores = {s.beat_id: s for a in result.analyses for s in a.beats}
    run_v = {v.candidate_id: v for v in result.verdicts}
    rank = {p.candidate_id: p.rank for p in result.promos}
    ev = {r["id"]: r for r in (eval_rep or {}).get("clips", []) if r.get("set") == "tool"}
    prelims = [c.prelim_score for c in result.candidates]
    high = float(np.quantile(prelims, top_quantile)) if prelims else 0.0
    rows = []
    for c in result.candidates:
        flags = []
        rv, er = run_v.get(c.id), ev.get(c.id)
        cold_fail = (rv is not None and rv.verdict == "fail") or (er is not None and er.get("verdict") == "fail")
        if c.prelim_score >= high and cold_fail:
            flags.append("high_score_cold_fail")
        if rv is not None and er is not None and er.get("verdict") and rv.verdict != er["verdict"]:
            flags.append("judges_disagree")
        first = c.segments[0]
        ep = eps.get(first.episode)
        if ep is not None:
            opening = [b for b in ep.beats if first.start - EPS <= b.start <= first.start + HOOK_WINDOW_S]
            if not any(b.kind in SPEECH for b in opening):
                flags.append("visual_only_hook")
            try:
                hook = ep.beat(c.hook_beat_id)
                if hook.start - first.start > HOOK_WINDOW_S + EPS:
                    flags.append("hook_late")
            except KeyError:
                flags.append("hook_missing")
        if c.needs_context:
            flags.append(f"needs_context:{len(c.needs_context)}")
        if c.spoiler >= 2:
            flags.append(f"spoiler:{c.spoiler}")
        res_id = scores[c.cliff_beat_id].resolved_by if c.cliff_beat_id in scores else None
        if res_id:
            try:
                r = eps[int(res_id[1:3])].beat(res_id)
                if any(s.episode == r.episode and s.start - EPS <= r.start < s.end - EPS for s in c.segments):
                    flags.append("answer_inside")
            except (KeyError, ValueError):
                pass
        if c.kind == "multi":
            flags.append("multi_segment")
        if candidate_cut_beats(c, eps):
            flags.append("cuts_spoken_line")
        if flags:
            rows.append({
                "id": c.id, "rank": rank.get(c.id), "kind": c.kind, "timecodes": _timecodes(c),
                "duration": c.duration, "prelim": c.prelim_score, "hook": c.hook_score, "cliff": c.cliff_score,
                "run_verdict": f"{rv.verdict} {rv.standalone}/{rv.keep_watching}" if rv else None,
                "eval_verdict": (f"{er['verdict']} {er['standalone']}/{er['keep_watching']}"
                                 if er and er.get("verdict") else None),
                "flags": flags})
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, -r["prelim"]))
    return rows


def failures_md(rows: list[dict], name: str) -> str:
    L = [f"# Flagged picks: {name}", "", "Verdicts as `verdict standalone/keep_watching`. Read each flagged clip in "
         "the script before writing a diagnosis; flags are leads, not conclusions.", "",
         "| id | rank | kind | timecodes | dur | prelim | hook/cliff | run judge | eval judge | flags |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['id']} | {r['rank'] or '-'} | {r['kind']} | {r['timecodes']} | {r['duration']:.1f} | "
                 f"{r['prelim']:.2f} | {r['hook']:g}/{r['cliff']:g} | {r['run_verdict'] or '-'} | "
                 f"{r['eval_verdict'] or '-'} | {', '.join(r['flags'])} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Flag likely bad picks in a HookCut run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", help="eval/results/<run>/eval.json (adds the eval judge's verdicts)")
    ap.add_argument("--out", help="results folder (default eval/results/<run>)")
    a = ap.parse_args(argv)
    name = run_name(a.run)
    ev = json.loads(Path(a.eval).read_text()) if a.eval else None
    rows = mine(load_run(a.run), ev)
    out = Path(a.out) if a.out else results_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    md = failures_md(rows, name)
    (out / "flagged.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
