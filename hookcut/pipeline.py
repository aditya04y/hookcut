"""HookCut end to end.

parse (code) -> analyze episodes (LLM, 1 call each) -> build candidate clips (code) -> cold-viewer test on
finalists (LLM, a different model) -> story graph (code + 1 show-level LLM call) -> rank with diversity and the
graph's rules (code) -> "why it works" lines (LLM, 1 call) -> per-episode hook/cliffhanger report (code)
-> validation + engagement layer + exports.

CLI:
  python -m hookcut.pipeline --input data/pack --out outputs/run_01
  python -m hookcut.pipeline --input samples --out outputs/sample --judge groq:openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from .config import build_client, load_config
from .llm.registry import load_secrets
from .parse import parse_file, parse_folder, parse_text
from .runlog import RunLog
from .schema import Episode, RunResult

Progress = Callable[[str, float], None]          # (message, fraction 0..1)
Inputs = Union[str, Path, Sequence[Union[str, Path]], Sequence[tuple[str, str]]]


def load_episodes(inputs: Inputs, episodes: Optional[str] = None) -> list[Episode]:
    """A folder, a list of .txt paths, or a list of (filename, text) pairs (uploads)."""
    if isinstance(inputs, (str, Path)) and Path(inputs).is_dir():
        eps = parse_folder(inputs)
    elif isinstance(inputs, (str, Path)):
        eps = [parse_file(inputs)]
    else:
        items = list(inputs)
        eps = [parse_text(i[1], i[0]) if isinstance(i, tuple) else parse_file(i) for i in items]
        eps.sort(key=lambda e: e.number)
    if episodes:  # "1-10" or "3,7,9"
        wanted: set[int] = set()
        for part in episodes.split(","):
            a, _, b = part.partition("-")
            wanted |= set(range(int(a), int(b or a) + 1))
        eps = [e for e in eps if e.number in wanted]
    return eps


def run_pipeline(inputs: Inputs, *, cfg: Optional[dict] = None, out_dir: Optional[Path] = None,
                 progress: Optional[Progress] = None, use_cache: bool = True, show: Optional[str] = None,
                 episodes: Optional[str] = None, env_file: Optional[str] = None) -> RunResult:
    from . import analyze, candidates, export, judge, rank, report, validate, writer

    cfg = cfg or load_config()
    say = progress or (lambda msg, frac: print(f"[{frac:4.0%}] {msg}", flush=True))
    secrets = load_secrets(env_file)
    log = RunLog(prices=cfg.get("prices_usd_per_mtok") or {})

    eps = load_episodes(inputs, episodes)
    if not eps:
        raise ValueError("no episodes found in input")
    say(f"parsed {len(eps)} episodes, {sum(len(e.beats) for e in eps)} beats", 0.02)
    for e in eps:
        for w in e.warnings:
            log.note("parse", f"E{e.number:02d}: {w}")

    clients = {s: build_client(cfg, s, secrets, use_cache) for s in ("analyze", "judge", "writer")}
    log.note("config", "models", models={s: c.spec for s, c in clients.items()})
    conc = cfg.get("concurrency") or {}
    pc = cfg.get("promo") or {}

    analyses = analyze.analyze_show(
        eps, clients["analyze"], log=log, concurrency=conc.get("analyze", 4),
        progress=lambda m, f: say(m, 0.05 + 0.55 * f))
    say(f"analyzed {len(analyses)} episodes", 0.60)

    cands = candidates.generate_candidates(eps, analyses, pc)
    say(f"built {len(cands)} candidate clips", 0.62)

    verdicts = judge.judge_candidates(
        cands, eps, clients["judge"], top_k=pc.get("judge_top_k", 40), log=log,
        concurrency=conc.get("judge", 2), progress=lambda m, f: say(m, 0.64 + 0.26 * f))
    say(f"cold-viewer tested {len(verdicts)} finalists", 0.90)

    head = dict(show=show or _show_name(inputs, eps), created_at=dt.datetime.now().isoformat(timespec="seconds"),
                config={"models": {s: c.spec for s, c in clients.items()}, "promo": pc})
    graph = build_graph(RunResult(**head, episodes=eps, analyses=analyses, candidates=cands, verdicts=verdicts),
                        cfg, secrets, use_cache, log, say)
    promos = rank.rank_promos(cands, verdicts, eps, analyses, pc, graph=graph)
    promos = writer.write_why_lines(promos, eps, clients["writer"], log=log)
    say(f"ranked {len(promos)} promos" + (" with the story graph" if graph is not None else ""), 0.95)

    result = RunResult(
        **head, episodes=eps, analyses=analyses, candidates=cands, verdicts=verdicts, promos=promos,
        episode_report=report.episode_report(eps, analyses), run_log=log.rows)
    issues = validate.validate_promos(promos, eps, pc, analyses)
    for i in issues:
        log.note("validate", json.dumps(i))
    result.run_log = log.rows
    layer_files = run_layers(result, cfg, log, Path(out_dir) if out_dir else None, say, graph=graph)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        export.write_outputs(result, out)
        (out / "run_summary.json").write_text(json.dumps(
            {"show": result.show, "episodes": len(eps), "candidates": len(cands), "finalists": len(verdicts),
             "promos": len(promos), "validation_issues": issues, "layers": layer_files, "cost": log.summary()},
            indent=2))
        log.write_jsonl(out / "run_log.jsonl")
    say("done", 1.0)
    return result


def build_graph(partial: RunResult, cfg: dict, secrets: dict, use_cache: bool, log: RunLog, say: Progress):
    """Content genome (docs/process/team_board.md, v2): the show-level story graph, built BEFORE ranking so it can shape the
    promo list (rank.py, docstring item 5). It reads the show name, episodes, analyses and candidates (CharacterNode.promos
    is filled in after ranking); one show-level LLM call when models.genome is set, cached like every call.
    Optional, and it can never fail the promo run: any error -> None, and ranking runs exactly as without a graph."""
    if not (cfg.get("layers") or {}).get("story_graph", True):
        return None
    try:
        from . import story_graph
        client = build_client(cfg, "genome", secrets, use_cache) if (cfg.get("models") or {}).get("genome") else None
        graph = story_graph.build_story_graph(partial, client, log=log)
        say("built the story graph", 0.92)
        return graph
    except Exception as e:  # noqa: BLE001
        log.note("genome", f"story graph skipped: {type(e).__name__}: {e}")
        return None


def run_layers(result: RunResult, cfg: dict, log: RunLog, out: Optional[Path], say: Progress, graph=None) -> dict:
    """What sits on top of the finished promo list: the story graph from build_graph (promo counts refreshed, then
    written as story_graph.json) and the engagement layer (activity matrices + models; SIMULATED unless real activity
    CSVs are given), which reads the final promos and the same graph. Neither can fail the promo run."""
    layers, files = cfg.get("layers") or {}, {}
    if graph is not None:
        try:
            from . import story_graph
            story_graph.refresh_promo_counts(graph, result)
            if out:
                out.mkdir(parents=True, exist_ok=True)
                files["story_graph"] = str(story_graph.write_story_graph(graph, out))
        except Exception as e:  # noqa: BLE001
            log.note("genome", f"story graph not written: {type(e).__name__}: {e}")
    if layers.get("engagement", True):
        try:
            from . import engagement
            ec = cfg.get("engagement") or {}
            rows = engagement.load_activity(result, graph=graph, seed=ec.get("seed"), viewers=ec.get("viewers"),
                                            paywall=ec.get("paywall_episode"))
            report = engagement.analyze(result, rows["impressions"], rows["sessions"], source=rows["source"],
                                        planted=rows["planted"], paywall=rows["paywall"],
                                        bootstrap=ec.get("bootstrap", 200), keys=rows["keys"], graph=graph)
            if out:
                files.update(engagement.write_engagement(report, out, rows["impressions"], rows["sessions"]))
            say(f"engagement layer ({rows['source']})", 0.99)
        except Exception as e:  # noqa: BLE001
            log.note("engagement", f"engagement skipped: {type(e).__name__}: {e}")
    return files


def _show_name(inputs: Inputs, eps: list[Episode]) -> str:
    if isinstance(inputs, (str, Path)) and Path(inputs).is_dir():
        name = Path(inputs).resolve().name
        return re.sub(r"[_-]+", " ", name).strip()
    return f"{len(eps)} uploaded episodes"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Find ranked promo clips in microdrama screenplays.")
    ap.add_argument("--input", required=True, help="folder of episode .txt files, or one file")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--config", default=None)
    ap.add_argument("--analyze", help="provider:model for beat scoring")
    ap.add_argument("--judge", help="provider:model for the cold-viewer test")
    ap.add_argument("--writer", help="provider:model for why-it-works lines")
    ap.add_argument("--episodes", help="subset, e.g. 1-10 or 3,7,9")
    ap.add_argument("--top", type=int, help="number of promos in the ranked list")
    ap.add_argument("--env-file", help="file with API keys (only provider key variables are read)")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {"analyze": a.analyze, "judge": a.judge, "writer": a.writer})
    if a.top:
        cfg.setdefault("promo", {})["final_top_n"] = a.top
    run_pipeline(a.input, cfg=cfg, out_dir=Path(a.out), use_cache=not a.no_cache, episodes=a.episodes,
                 env_file=a.env_file)


if __name__ == "__main__":
    main()
