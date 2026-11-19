"""Compare two runs' episode analyses where different models scored the same episode (no API calls).

Usage: .venv/bin/python -m eval.model_compare --a outputs/run_pass1_mixed/result.json --b outputs/run_full/result.json
Writes eval/results/model_compare.md (beat ids, scores and timings only; no script text).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hookcut.schema import RunResult

NEAR_S = 3.0  # two picks "agree" if they are the same beat or start within 3 s


def _top(a, field):
    beats = [b for b in a.beats if getattr(b, field) > 0]
    return max(beats, key=lambda b: (getattr(b, field), -beats.index(b)), default=None)


def compare(ra: RunResult, rb: RunResult) -> dict:
    eps = {e.number: e for e in ra.episodes}
    A = {a.episode: a for a in ra.analyses if not a.summary.startswith("analysis failed")}
    B = {a.episode: a for a in rb.analyses if not a.summary.startswith("analysis failed")}
    rows = []
    for n in sorted(set(A) & set(B)):
        a, b = A[n], B[n]
        if a.model == b.model:
            continue
        ep = eps[n]
        start = {x.id: x.start for x in ep.beats}
        row = {"episode": n, "model_a": a.model, "model_b": b.model}
        for field in ("hook", "cliff"):
            ta, tb = _top(a, field), _top(b, field)
            if ta and tb:
                row[f"top_{field}_a"], row[f"top_{field}_b"] = ta.beat_id, tb.beat_id
                row[f"top_{field}_agree"] = ta.beat_id == tb.beat_id or abs(start[ta.beat_id] - start[tb.beat_id]) <= NEAR_S
            else:
                row[f"top_{field}_agree"] = None
        strong = lambda x: {s.beat_id for s in x.beats if max(s.hook, s.cliff) >= 4}
        sa, sb = strong(a), strong(b)
        row["strong_beats_a"], row["strong_beats_b"] = len(sa), len(sb)
        row["strong_overlap_jaccard"] = round(len(sa & sb) / len(sa | sb), 2) if sa | sb else None
        row["opening_a"], row["opening_b"] = a.opening_hook.score, b.opening_hook.score
        row["ending_a"], row["ending_b"] = a.ending_cliffhanger.score, b.ending_cliffhanger.score
        rows.append(row)
    agree = lambda k: [r[k] for r in rows if r.get(k) is not None]
    rate = lambda xs: (sum(xs), len(xs))
    return {"rows": rows, "top_hook_agree": rate(agree("top_hook_agree")),
            "top_cliff_agree": rate(agree("top_cliff_agree")),
            "mean_jaccard": round(sum(agree("strong_overlap_jaccard")) / max(1, len(agree("strong_overlap_jaccard"))), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="eval/results/model_compare.md")
    args = ap.parse_args()
    ra = RunResult.model_validate_json(Path(args.a).read_text())
    rb = RunResult.model_validate_json(Path(args.b).read_text())
    res = compare(ra, rb)
    h, c = res["top_hook_agree"], res["top_cliff_agree"]
    lines = ["# Same episodes, different analysis models", "",
             f"Episodes compared: {len(res['rows'])} (run A: `{args.a}`, run B: `{args.b}`)", "",
             f"- Strongest hook beat agrees (same beat or within {NEAR_S:.0f} s): **{h[0]}/{h[1]}**",
             f"- Strongest cliffhanger beat agrees: **{c[0]}/{c[1]}**",
             f"- Overlap of strong beats (score >= 4), mean Jaccard: **{res['mean_jaccard']}**", "",
             "| Ep | Model A | Model B | top hook A / B | agree | top cliff A / B | agree | strong A/B | Jaccard | opening A/B | ending A/B |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| E{r['episode']:02d} | {r['model_a']} | {r['model_b']} | {r.get('top_hook_a', '-')} / {r.get('top_hook_b', '-')} | "
                     f"{r['top_hook_agree']} | {r.get('top_cliff_a', '-')} / {r.get('top_cliff_b', '-')} | {r['top_cliff_agree']} | "
                     f"{r['strong_beats_a']}/{r['strong_beats_b']} | {r['strong_overlap_jaccard']} | "
                     f"{r['opening_a']}/{r['opening_b']} | {r['ending_a']}/{r['ending_b']} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
