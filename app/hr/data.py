"""Data for the reviewer pages: the saved sample run, one-episode slices of any run, and a live one-episode run
that reports each stage as it finishes."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from hookcut.schema import RunResult

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RUN = ROOT / "app" / "fixtures" / "sample_run.json"
EVAL_JSON = ROOT / "eval" / "results" / "run_full" / "eval.json"
SECRET_NAMES = ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                "HOOKCUT_BASE_URL", "HOOKCUT_API_KEY")
STEP_TITLES = ["Read the script", "Score every line", "Build the legal cuts", "Show each cut to a stranger",
               "Pick and explain"]


def copy_cloud_secrets() -> None:
    """On Streamlit Community Cloud, keys live in st.secrets; copy only the provider variables HookCut reads."""
    try:
        for name in SECRET_NAMES:
            if (value := st.secrets.get(name)) and not os.environ.get(name):
                os.environ[name] = str(value)
    except Exception:
        pass


def live_models_available() -> bool:
    copy_cloud_secrets()
    from hookcut.llm.registry import load_secrets
    s = load_secrets()
    return bool(s.get("GEMINI_API_KEY") or s.get("GROQ_API_KEY") or s.get("ANTHROPIC_API_KEY")
                or s.get("OPENAI_API_KEY") or s.get("HOOKCUT_BASE_URL"))


@st.cache_data(show_spinner=False)
def _read(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_result(path: Path | str) -> RunResult:
    p = Path(path)
    return RunResult.model_validate_json(_read(str(p), p.stat().st_mtime))


def sample_result() -> RunResult:
    return load_result(SAMPLE_RUN)


def local_runs() -> list[Path]:
    out = ROOT / "outputs"
    return sorted(out.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True) if out.exists() else []


def eval_stats() -> Optional[dict]:
    if EVAL_JSON.exists():
        try:
            return json.loads(EVAL_JSON.read_text())["stats"]
        except Exception:
            return None
    return None


def episode_slice(result: RunResult, number: int) -> RunResult:
    """Everything a run knows about one episode (cuts that stay inside it)."""
    cands = [c for c in result.candidates if all(s.episode == number for s in c.segments)]
    ids = {c.id for c in cands}
    return RunResult(
        show=result.show, created_at=result.created_at, config=result.config,
        episodes=[e for e in result.episodes if e.number == number],
        analyses=[a for a in result.analyses if a.episode == number],
        candidates=cands, verdicts=[v for v in result.verdicts if v.candidate_id in ids],
        promos=[p for p in result.promos if all(s.episode == number for s in p.segments)],
        episode_report=[r for r in result.episode_report if r.get("episode") == number], run_log=[])


def review_only_promo(r: RunResult):
    """When no cut passes the stranger test, the best cut it found, labelled for review only, so the reviewer can
    still see where it sits and why the stranger got lost."""
    from hookcut.rank import beat_text
    from hookcut.schema import Promo
    from hookcut.textfmt import ep_tc
    if not r.candidates:
        return None
    verdicts = {v.candidate_id: v for v in r.verdicts}
    c = max(r.candidates, key=lambda c: (c.id in verdicts, c.prelim_score))
    beats = {b.id: b for e in r.episodes for b in e.beats}
    hook, cliff = beats.get(c.hook_beat_id), beats.get(c.cliff_beat_id)
    if hook is None or cliff is None:
        return None
    v = verdicts.get(c.id)
    return Promo(rank=1, candidate_id=c.id, kind=c.kind, segments=c.segments, duration=c.duration,
                 hook_beat_id=c.hook_beat_id, cliff_beat_id=c.cliff_beat_id,
                 hook_time=ep_tc(hook.episode, hook.start), hook_text=beat_text(hook),
                 cliffhanger_time=ep_tc(cliff.episode, cliff.start), cliffhanger_text=beat_text(cliff),
                 open_question=(v.open_question if v else c.open_question) or "",
                 why_it_works="Shown for review only: no cut passed the stranger test.", post_wave=2,
                 flags=["review only"], cold_viewer=v)


def step_summaries(r: RunResult) -> list[str]:
    """One plain sentence per stage, from a finished (or replayed) one-episode run."""
    from hookcut.textfmt import tc
    ep = r.episodes[0]
    a = r.analyses[0] if r.analyses else None
    out = [f"{len(ep.beats)} lines in {len(ep.scenes)} scene{'s' if len(ep.scenes) != 1 else ''}, {tc(ep.duration)[:5]} long"]
    if a and a.beats:
        h = max(a.beats, key=lambda s: s.hook)
        c = max(a.beats, key=lambda s: s.cliff)
        out.append(f"strongest hook at {tc(ep.beat(h.beat_id).start)[:5]} ({h.hook}/5), "
                   f"strongest cliffhanger at {tc(ep.beat(c.beat_id).start)[:5]} ({c.cliff}/5)")
    else:
        out.append("no line scored as a hook or a cliffhanger")
    out.append(f"{len(r.candidates)} cut{'s' if len(r.candidates) != 1 else ''} pass the rules")
    counts = {k: sum(1 for v in r.verdicts if v.verdict == k) for k in ("pass", "borderline", "fail")}
    out.append(f"{counts['pass']} followed it, {counts['borderline']} half-followed, {counts['fail']} got lost")
    if r.promos:
        p = r.promos[0]
        from .components import cut_label
        out.append(f"best cut {cut_label(p.segments)} ({p.duration:.0f} s)")
    else:
        out.append("no cut good enough to recommend")
    return out


def run_one_episode(filename: str, text: str, on_step: Callable[[int, str, str], None]) -> RunResult:
    """Run every stage on a single episode, reporting each one as it finishes.
    on_step(index, state, message) with state in {"working", "done"}."""
    from hookcut import analyze, candidates, judge, rank, report, validate, writer
    from hookcut.config import build_client, load_config
    from hookcut.llm.registry import load_secrets
    from hookcut.parse import parse_text
    from hookcut.runlog import RunLog

    copy_cloud_secrets()
    on_step(0, "working", "")
    ep = parse_text(text, filename)
    if not ep.beats:
        raise ValueError("No timestamped lines found. The script needs blocks like **NAME (Mood):** followed by "
                         "Start: (00:01.00) | End: (00:03.00) and the line.")
    cfg = load_config()
    pc = {**cfg.get("promo", {}), "judge_top_k": 6, "final_top_n": 3}
    secrets = load_secrets()
    log = RunLog(prices=cfg.get("prices_usd_per_mtok") or {})
    partial = RunResult(show="Your episode", created_at=dt.datetime.now().isoformat(timespec="seconds"), episodes=[ep])
    on_step(0, "done", step_summaries(partial)[0])

    on_step(1, "working", "")
    analyses = analyze.analyze_show([ep], build_client(cfg, "analyze", secrets), log=log, concurrency=1)
    if analyses[0].summary.startswith("analysis failed"):
        detail = analyses[0].summary
        if any(code in detail for code in ("Error code: 400", "Error code: 401", "Error code: 403", "Invalid API Key")):
            raise RuntimeError("The AI models rejected the request, which on this deployment usually means an API key "
                               "in the app's Secrets is wrong (a placeholder or a partial key). The saved sample runs "
                               "still work.")
        raise RuntimeError("The scoring model did not answer (" + detail[17:160] + ")")
    partial.analyses = analyses
    on_step(1, "done", step_summaries(partial)[1])

    on_step(2, "working", "")
    cands = candidates.generate_candidates([ep], analyses, pc)
    partial.candidates = cands
    on_step(2, "done", step_summaries(partial)[2])

    on_step(3, "working", "")
    verdicts = judge.judge_candidates(cands, [ep], build_client(cfg, "judge", secrets), top_k=pc["judge_top_k"],
                                      log=log, concurrency=2)
    partial.verdicts = verdicts
    on_step(3, "done", step_summaries(partial)[3])

    on_step(4, "working", "")
    promos = rank.rank_promos(cands, verdicts, [ep], analyses, pc)
    if promos:
        promos = writer.write_why_lines(promos, [ep], build_client(cfg, "writer", secrets), log=log)
    partial.promos = promos
    partial.episode_report = report.episode_report([ep], analyses)
    partial.run_log = log.rows
    partial.config = {"models": {s: cfg["models"][s] for s in ("analyze", "judge", "writer")}, "promo": pc,
                      "issues": validate.validate_promos(promos, [ep], pc, analyses)}
    on_step(4, "done", step_summaries(partial)[4])
    return partial
