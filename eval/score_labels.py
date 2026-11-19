"""Score the tool's best pick(s) per hold-out episode against human labels. No model calls.

  .venv/bin/python -m eval.score_labels --run outputs/<run>/result.json --labels outputs/labeling/labels.csv

A pick "hits" a label if temporal IoU >= 0.3, or its hook starts within +-3 s of the labelled start AND its
cliffhanger ends within +-3 s of the labelled end. hit@k: any of the tool's top-k picks for that episode hits
any label row for it. The last-window and random baselines are scored the same way, for free.
Writes eval/results/<run>/labels.json and labels.md (numbers only; the labeler's notes are not copied).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

from hookcut.schema import Candidate, Episode, RunResult

from . import HIT_IOU, HIT_TOL_S, HOLDOUT_EPISODES
from .baselines import last_window_baseline, random_window_baseline
from .common import ROOT, episode_map, load_run, results_dir, run_name
from .metrics import interval_iou, summarize


def parse_time(s: str) -> Optional[float]:
    """'01:04.50' / '1:04' / '64.5' -> seconds; '' -> None."""
    s = (s or "").strip()
    if not s:
        return None
    if ":" in s:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def read_labels(path: str | Path, episodes: Sequence[int] = HOLDOUT_EPISODES) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ep = int((row.get("episode") or "").strip())
                s, e = parse_time(row.get("best_start", "")), parse_time(row.get("best_end", ""))
            except ValueError:
                continue
            if s is None or e is None or ep not in episodes:
                continue
            out.setdefault(ep, []).append({"start": min(s, e), "end": max(s, e),
                                           "labeler": (row.get("labeler") or "").strip()})
    return out


def tool_ranking(result: RunResult) -> dict[int, list[Candidate]]:
    """Per episode, the tool's order: promos by final rank first, then the other candidates by prelim_score."""
    by_id = {c.id: c for c in result.candidates}
    ordered: list[Candidate] = []
    for p in sorted(result.promos, key=lambda p: p.rank):
        c = by_id.get(p.candidate_id)
        if c is not None:
            ordered.append(c.model_copy(update={"segments": p.segments}))
    seen = {c.id for c in ordered}
    ordered += sorted((c for c in result.candidates if c.id not in seen), key=lambda c: -c.prelim_score)
    out: dict[int, list[Candidate]] = {}
    for c in ordered:
        for ep in sorted({s.episode for s in c.segments}):
            out.setdefault(ep, []).append(c)
    return out


def _beat(eps: dict[int, Episode], ep: int, beat_id: str):
    try:
        b = eps[int(beat_id[1:3])].beat(beat_id)
    except (KeyError, ValueError):
        return None
    return b if b.episode == ep else None


def match(c: Candidate, ep: int, label: dict, eps: dict[int, Episode]) -> dict:
    iv = [(s.start, s.end) for s in c.segments if s.episode == ep]
    iou = interval_iou(iv, [(label["start"], label["end"])]) if iv else 0.0
    hook, cliff = _beat(eps, ep, c.hook_beat_id), _beat(eps, ep, c.cliff_beat_id)
    edges = bool(hook and cliff and abs(hook.start - label["start"]) <= HIT_TOL_S
                 and abs(cliff.end - label["end"]) <= HIT_TOL_S)
    return {"iou": round(iou, 3), "edges_match": edges, "hit": iou >= HIT_IOU or edges}


def score(ranked: dict[int, list[Candidate]], labels: dict[int, list[dict]], eps: dict[int, Episode],
          ks: Sequence[int] = (1, 3)) -> dict:
    per_ep = []
    for ep in sorted(labels):
        picks = ranked.get(ep, [])
        row = {"episode": ep, "n_labels": len(labels[ep]), "n_picks": len(picks)}
        ms = [[match(c, ep, lab, eps) for lab in labels[ep]] for c in picks]
        row["iou_at_1"] = max((m["iou"] for m in ms[0]), default=0.0) if ms else 0.0
        row["top1_id"] = picks[0].id if picks else None
        for k in ks:
            row[f"hit@{k}"] = any(m["hit"] for pm in ms[:k] for m in pm)
            row[f"best_iou@{k}"] = max((m["iou"] for pm in ms[:k] for m in pm), default=0.0)
        per_ep.append(row)
    agg = {"n_episodes": len(per_ep), "iou_at_1": summarize([r["iou_at_1"] for r in per_ep])}
    for k in ks:
        hits = [r[f"hit@{k}"] for r in per_ep]
        agg[f"hit@{k}"] = {"hits": sum(hits), "n": len(hits), **summarize(hits)}
    return {"aggregate": agg, "per_episode": per_ep}


def score_all(result: RunResult, labels: dict[int, list[dict]], seed: int = 7) -> dict:
    eps = episode_map(result.episodes)
    labelled = [eps[n] for n in sorted(labels) if n in eps]
    rand = random_window_baseline(labelled, seed=seed, per_episode=3)
    by_ep = lambda cs: {n: [c for c in cs if c.segments[0].episode == n] for n in labels}  # noqa: E731
    return {"tool": score(tool_ranking(result), labels, eps),
            "last_window": score(by_ep(last_window_baseline(labelled)), labels, eps),
            "random": score(by_ep(rand), labels, eps),
            "rule": {"hit_iou": HIT_IOU, "hit_tol_s": HIT_TOL_S}}


def labels_md(rep: dict, name: str) -> str:
    L = [f"# Human-label agreement: {name}", "",
         f"Hit = IoU >= {HIT_IOU} or hook and cliff each within +-{HIT_TOL_S:g} s of the labelled start/end. "
         "Denominator = hold-out episodes with at least one label.", "",
         "| method | episodes | hit@1 | hit@3 | mean IoU of top pick [90% CI] |", "|---|---|---|---|---|"]
    for m in ("tool", "last_window", "random"):
        a = rep[m]["aggregate"]
        if not a["n_episodes"]:
            L.append(f"| {m} | 0 | n/a | n/a | n/a |")
            continue
        iou = a["iou_at_1"]
        ci = f" [{iou['ci90'][0]:.2f}, {iou['ci90'][1]:.2f}]" if iou["ci90"] else ""
        L.append(f"| {m} | {a['n_episodes']} | {a['hit@1']['hits']}/{a['hit@1']['n']} | "
                 f"{a['hit@3']['hits']}/{a['hit@3']['n']} | {iou['mean']:.2f}{ci} |")
    L += ["", "Last-window has one pick per episode (hit@3 = hit@1); random has three seeded windows.", "",
          "| episode | labels | tool top pick | IoU@1 | hit@1 | hit@3 |", "|---|---|---|---|---|---|"]
    for r in rep["tool"]["per_episode"]:
        L.append(f"| {r['episode']} | {r['n_labels']} | {r['top1_id'] or '-'} | {r['iou_at_1']:.2f} | "
                 f"{'yes' if r['hit@1'] else 'no'} | {'yes' if r['hit@3'] else 'no'} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score HookCut picks against human labels on hold-out episodes.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", default=str(ROOT / "outputs" / "labeling" / "labels.csv"))
    ap.add_argument("--out", help="results folder (default eval/results/<run>)")
    a = ap.parse_args(argv)
    labels = read_labels(a.labels)
    if not labels:
        print(f"no filled-in hold-out labels in {a.labels} (episodes {list(HOLDOUT_EPISODES)}); nothing to score")
        return 1
    rep = score_all(load_run(a.run), labels)
    out = Path(a.out) if a.out else results_dir(run_name(a.run))
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps(rep, indent=2))
    md = labels_md(rep, run_name(a.run))
    (out / "labels.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
