"""Cold-viewer evaluation: the tool's top-N promos vs two baselines, judged by the same function and model.

  HOOKCUT_ENV_FILE=... .venv/bin/python -m eval.run_eval --run outputs/<run>/result.json \
      --judge groq:openai/gpt-oss-120b --n 20 [--judge2 run|provider:model] [--dry-run]

Sets: tool (top-N promos), last_window (final ~45 s of N evenly spaced episodes), random (seeded random
25-45 s windows). Judge agreement: the tool's top-N judged by a second model (default: the run's own stored
verdicts, zero calls). Writes eval/results/<run>/eval.json + summary.md (numbers and ids only) and keeps the
judge's free-text answers in outputs/<run>/eval_verdicts.json (private: they paraphrase the script).
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import math
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from hookcut.llm.base import LLMClient, LLMResult
from hookcut.runlog import RunLog
from hookcut.schema import Candidate, ColdViewerVerdict, Episode, Promo, RunResult
from hookcut.textfmt import ep_tc

from . import MAX_JUDGE_CALLS, TUNING_EPISODES
from .baselines import evenly_spaced, last_window_baseline, random_window_baseline
from .common import (EPS, STORY, as_promo, candidate_cut_beats, episode_map, family, load_run, resolve_run,
                     results_dir, run_name, spec_families)
from .metrics import agreement, bootstrap_diff_ci, spearman, summarize

JudgeFn = Callable[..., list[ColdViewerVerdict]]
ValidateFn = Callable[[list[Promo], list[Episode], dict], list[dict]]
DEFAULT_JUDGES = ("groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b", "gemini:gemini-3.8-flash")
SETS = ("tool", "last_window", "random")


# ------------------------------------------------------------------ budget
class BudgetExceeded(RuntimeError):
    pass


def check_budget(planned: int, cap: int = MAX_JUDGE_CALLS) -> None:
    if planned > min(cap, MAX_JUDGE_CALLS):
        raise BudgetExceeded(f"planned {planned} judge calls > cap {min(cap, MAX_JUDGE_CALLS)}; lower --n / --n2")


class BudgetClient(LLMClient):
    """Wraps a client and refuses the call that would exceed the cap. Cache hits are free and not counted."""

    _lock = threading.Lock()   # shared by every BudgetClient, since two clients can share one counter

    def __init__(self, inner: LLMClient, cap: int, counter: Optional[dict] = None):
        super().__init__(inner.model, max_tokens=inner.max_tokens, temperature=inner.temperature,
                         reasoning=inner.reasoning, cache=None)
        self.inner, self.cap, self.provider = inner, cap, inner.provider
        self.counter = counter if counter is not None else {"calls": 0}

    @property
    def spec(self) -> str:
        return self.inner.spec

    def complete_json(self, system: str, user: str, schema=None, *, validate=None, retries: int = 2,
                      cache_tag: str = "") -> LLMResult:
        with self._lock:
            if self.counter["calls"] >= self.cap:
                raise BudgetExceeded(f"judge-call cap of {self.cap} reached")
            self.counter["calls"] += 1          # reserve the slot before calling (threads); a failed live
        res = self.inner.complete_json(system, user, schema, validate=validate, retries=retries,
                                       cache_tag=cache_tag)   # call still spent quota, so it stays counted
        if res.cached:
            with self._lock:
                self.counter["calls"] -= 1
        return res


# ------------------------------------------------------------------ sets
def tool_candidates(result: RunResult, n: int) -> list[Candidate]:
    """The tool's top-n promos as Candidates, using the promo's own segments (what would ship)."""
    by_id = {c.id: c for c in result.candidates}
    eps = episode_map(result.episodes)
    out = []
    for p in sorted(result.promos, key=lambda p: p.rank)[:n]:
        c = by_id.get(p.candidate_id)
        if c is None:   # rebuild from the promo: hook/cliff = first/last non-camera beat it covers
            story = [b for s in p.segments for b in eps[s.episode].beats
                     if b.kind != "camera" and b.start >= s.start - EPS and b.end <= s.end + EPS]
            if not story:
                continue
            c = Candidate(id=p.candidate_id, kind=p.kind, segments=p.segments, duration=p.duration,
                          hook_beat_id=story[0].id, cliff_beat_id=story[-1].id, hook_score=0, cliff_score=0,
                          prelim_score=0, notes=["rebuilt from promo"])
        elif [s.model_dump() for s in c.segments] != [s.model_dump() for s in p.segments]:
            c = c.model_copy(update={"segments": p.segments, "duration": p.duration,
                                     "notes": c.notes + ["segments from promo"]})
        out.append(c)
    return out


def baseline_sets(episodes: Sequence[Episode], n: int, seed: int = 7) -> dict[str, list[Candidate]]:
    eps = sorted(episodes, key=lambda e: e.number)
    picked = evenly_spaced(eps, n)
    per_ep = max(1, math.ceil(n / max(1, len(picked))))
    return {"last_window": last_window_baseline(picked)[:n],
            "random": random_window_baseline(picked, seed=seed, per_episode=per_ep)[:n]}


def plan(sets: dict[str, list[Candidate]], n2: int) -> dict:
    p = {k: len(v) for k, v in sets.items()}
    p["judge2"] = n2
    p["total"] = sum(p.values())
    return p


# ------------------------------------------------------------------ per-clip records
def _issue_code(issue: Any) -> str:
    if isinstance(issue, dict):
        for k in ("check", "code", "type", "rule", "kind", "issue"):
            if k in issue:
                return str(issue[k])
        return ",".join(sorted(issue))
    return str(issue)[:60]


def own_checks(c: Candidate, eps: dict[int, Episode], cfg: dict) -> dict:
    """Independent of the editor's validator: length, hook in the first 3 s, no line cut by a boundary."""
    lo, hi = float(cfg.get("min_s", 20)), float(cfg.get("max_s", 60))
    hook_window = float(cfg.get("hook_window_s", 3.0))
    first = c.segments[0]
    try:
        hook = eps[first.episode].beat(c.hook_beat_id)
        hook_ok = hook.episode == first.episode and first.start - EPS <= hook.start <= first.start + hook_window + EPS
    except KeyError:
        hook_ok = False
    cuts = candidate_cut_beats(c, eps)                                   # spoken lines: an error
    cut_other = [b for b in candidate_cut_beats(c, eps, STORY) if b not in cuts]   # action/sfx: noted only
    length_ok = lo - 0.5 <= c.duration <= hi + 0.5
    return {"length_ok": length_ok, "hook_in_window": hook_ok, "cut_beats": cuts, "cut_action_sfx": cut_other,
            "ok": length_ok and hook_ok and not cuts}


def clip_record(set_name: str, c: Candidate, v: Optional[ColdViewerVerdict], eps: dict[int, Episode], cfg: dict,
                promo: Optional[Promo], validate_fn: Optional[ValidateFn], episodes: list[Episode]) -> dict:
    issues = None
    if validate_fn is not None:
        try:
            issues = [_issue_code(i) for i in validate_fn([promo or as_promo(c, eps)], episodes, cfg)]
        except Exception as e:   # the validator is someone else's code: record, don't crash the eval
            issues = [f"validator_error:{type(e).__name__}"]
    ep_nums = sorted({s.episode for s in c.segments})
    return {
        "set": set_name, "id": c.id, "kind": c.kind, "episodes": ep_nums,
        "segments": [{"episode": s.episode, "start": s.start, "end": s.end} for s in c.segments],
        "timecodes": " + ".join(f"{ep_tc(s.episode, s.start)}-{ep_tc(s.episode, s.end)[4:]}" for s in c.segments),
        "duration": c.duration, "rank": promo.rank if promo else None,
        "prelim_score": c.prelim_score, "hook_score": c.hook_score, "cliff_score": c.cliff_score,
        "spoiler": c.spoiler, "needs_context": len(c.needs_context),
        "from_tuning_episode": any(e in TUNING_EPISODES for e in ep_nums),
        "judged": v is not None,
        "verdict": v.verdict if v else None, "standalone": v.standalone if v else None,
        "keep_watching": v.keep_watching if v else None, "hook_lands": v.hook_lands if v else None,
        "n_confusions": len(v.confusions) if v else None, "judge_model": v.model if v else None,
        "validator_issues": issues,   # variants are alternatives: their issues don't make the main cut invalid
        "valid": None if issues is None else not [i for i in issues if not i.startswith("variant")],
        "own_checks": own_checks(c, eps, cfg),
    }


def set_stats(records: list[dict]) -> dict:
    judged = [r for r in records if r["judged"]]
    verdicts = {k: sum(r["verdict"] == k for r in judged) for k in ("pass", "borderline", "fail")}
    valid = [r["valid"] for r in records if r["valid"] is not None]
    return {
        "n_clips": len(records), "n_judged": len(judged), "n_missing": len(records) - len(judged),
        "verdicts": verdicts,
        "pass_rate": summarize([r["verdict"] == "pass" for r in judged]),
        "standalone": summarize([r["standalone"] for r in judged]),
        "keep_watching": summarize([r["keep_watching"] for r in judged]),
        "hook_lands": summarize([r["hook_lands"] for r in judged]),
        "valid_validator": summarize(valid) if valid else {"n": 0, "mean": None, "ci90": None},
        "valid_own_checks": summarize([r["own_checks"]["ok"] for r in records]),
        "cuts_a_line": sum(bool(r["own_checks"]["cut_beats"]) for r in records),
        "validator_issue_counts": dict(Counter(i for r in records for i in (r["validator_issues"] or []))),
    }


def _verdict_dict(v: Optional[ColdViewerVerdict]) -> Optional[dict]:
    return v.model_dump() if v else None


# ------------------------------------------------------------------ the evaluation
def evaluate(result: RunResult, *, judge_fn: JudgeFn, client: LLMClient, n: int = 20,
             client2: Optional[LLMClient] = None, n2: int = 0, validate_fn: Optional[ValidateFn] = None,
             promo_cfg: Optional[dict] = None, log: Optional[RunLog] = None, seed: int = 7,
             concurrency: int = 2, cap: int = MAX_JUDGE_CALLS, say: Callable[[str], None] = print) -> dict:
    """Pure function of its inputs (judge/validator injected) so tests run without network."""
    cfg = promo_cfg or (result.config or {}).get("promo") or {}
    log = log or RunLog()
    eps = episode_map(result.episodes)
    sets = {"tool": tool_candidates(result, n), **baseline_sets(result.episodes, n, seed)}
    n2 = min(n2, len(sets["tool"])) if client2 is not None else 0
    p = plan(sets, n2)
    check_budget(p["total"], cap)

    counter = {"calls": 0}
    jc = BudgetClient(client, cap, counter)
    verdicts: dict[str, dict[str, ColdViewerVerdict]] = {}
    for name in SETS:
        say(f"judging {name}: {len(sets[name])} clips with {client.spec}")
        vs = judge_fn(sets[name], result.episodes, jc, top_k=len(sets[name]), log=log, concurrency=concurrency)
        verdicts[name] = {v.candidate_id: v for v in vs}

    promos = {pr.candidate_id: pr for pr in result.promos}
    records = {name: [clip_record(name, c, verdicts[name].get(c.id), eps, cfg,
                                  promos.get(c.id) if name == "tool" else None, validate_fn, result.episodes)
                      for c in cands] for name, cands in sets.items()}
    stats = {name: set_stats(recs) for name, recs in records.items()}
    stats["tool_excl_tuning"] = set_stats([r for r in records["tool"] if not r["from_tuning_episode"]])

    diffs = {}
    for base in ("last_window", "random"):
        for metric in ("pass", "keep_watching", "standalone"):
            def vals(recs):
                return [(r["verdict"] == "pass") if metric == "pass" else r[metric] for r in recs if r["judged"]]
            t, b = vals(records["tool"]), vals(records[base])
            diffs[f"tool_minus_{base}.{metric}"] = {
                "diff": (sum(t) / len(t) - sum(b) / len(b)) if t and b else None,
                "ci90": bootstrap_diff_ci(t, b), "n_tool": len(t), "n_base": len(b)}

    # judge agreement on the tool's clips
    tool_ids = [c.id for c in sets["tool"]]
    stored = {v.candidate_id: v for v in result.verdicts}
    for pr in result.promos:
        if pr.cold_viewer and pr.candidate_id not in stored:
            stored[pr.candidate_id] = pr.cold_viewer
    if client2 is not None and n2:
        say(f"judge2: re-judging tool top-{n2} with {client2.spec}")
        jc2 = BudgetClient(client2, cap, counter)
        vs2 = judge_fn(sets["tool"][:n2], result.episodes, jc2, top_k=n2, log=log, concurrency=concurrency)
        second, second_src, ids2 = {v.candidate_id: v for v in vs2}, client2.spec, tool_ids[:n2]
    else:
        second, second_src, ids2 = stored, "run's stored verdicts", tool_ids
    v1 = [_verdict_dict(verdicts["tool"].get(i)) for i in ids2]
    v2 = [_verdict_dict(second.get(i)) for i in ids2]
    agree = {"second_judge": second_src, **agreement(v1, v2),
             "second_models": sorted({d["model"] for d in v2 if d and d.get("model")})}

    tool_judged = [r for r in records["tool"] if r["judged"]]
    diagnostics = {"spearman_prelim_vs_keep_watching_tool": spearman(
        [r["prelim_score"] for r in tool_judged], [r["keep_watching"] for r in tool_judged])}

    tokens = {"input": 0, "output": 0, "live_calls": 0, "cached_calls": 0, "errors": 0}
    for row in log.rows:
        if row.get("item") == "note":
            continue
        if "error" in row:
            tokens["errors"] += 1
            continue
        if row.get("cached"):
            tokens["cached_calls"] += 1
        else:
            tokens["live_calls"] += 1
            tokens["input"] += row.get("input_tokens", 0)
            tokens["output"] += row.get("output_tokens", 0)

    private = {name: {i: _verdict_dict(v) for i, v in vs.items()} for name, vs in verdicts.items()}
    if client2 is not None and n2:
        private["judge2_tool"] = {i: _verdict_dict(v) for i, v in second.items()}
    return {"plan": p, "budget_counter": counter["calls"], "cap": min(cap, MAX_JUDGE_CALLS),
            "stats": stats, "diffs": diffs, "agreement": agree, "diagnostics": diagnostics,
            "tokens": tokens, "clips": [r for recs in records.values() for r in recs], "_private": private}


# ------------------------------------------------------------------ independence check
def used_families(result: RunResult, stage: str) -> dict[str, int]:
    """Families that actually answered a stage, from the run log (successful calls, cached or live). A verdict's
    `model` can hold a whole fallback chain, so the log is the truth; the config spec is the last resort."""
    rows = [r for r in result.run_log if r.get("stage") == stage and r.get("model") and "error" not in r]
    if rows:
        return dict(Counter(family(r["model"]) for r in rows))
    models = (result.config or {}).get("models") or {}
    return {f: 0 for f in spec_families(models.get(stage, "")) if f != "?"}


def choose_judge(result: RunResult, requested: Optional[str]) -> tuple[str, list[str]]:
    """Pick (or check) an eval judge whose family differs from the run's judge and analyzer."""
    run_judge, run_analyze = used_families(result, "judge"), used_families(result, "analyze")
    warnings = []
    spec = requested
    if not spec:
        spec = next((s for s in DEFAULT_JUDGES if family(s) not in {**run_judge, **run_analyze}), DEFAULT_JUDGES[0])
    fam = family(spec)
    if fam in run_judge:
        n_all = sum(run_judge.values())
        share = f"{run_judge[fam]} of {n_all} run judge calls" if n_all else "the run judge's config"
        warnings.append(f"eval judge family '{fam}' also judged in the run ({share}; calls by family: "
                        f"{run_judge}): not independent for those clips")
    if fam in run_analyze:
        warnings.append(f"eval judge family '{fam}' is the run's analyzer family ({run_analyze}): it may share its taste")
    return spec, warnings


def _build(spec: str, use_cache: bool, env_file: Optional[str]) -> LLMClient:
    from hookcut.config import build_client, load_config
    from hookcut.llm.registry import load_secrets
    cfg = load_config(None, {"judge": spec})          # explicit choice: no silent fallback models
    return build_client(cfg, "judge", load_secrets(env_file), use_cache)


# ------------------------------------------------------------------ report
def _fmt(s: dict, pct: bool = False) -> str:
    if not s or s.get("mean") is None:
        return "n/a"
    m, ci = s["mean"], s.get("ci90")
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    return f"{f(m)} [{f(ci[0])}, {f(ci[1])}] (n={s['n']})" if ci else f"{f(m)} (n={s['n']})"


def summary_md(rep: dict) -> str:
    st, L = rep["stats"], []
    L += [f"# Cold-viewer eval: {rep['run']}", "",
          f"Created {rep['created_at']}. Eval judge `{rep['judge']}`; run judge `{rep['run_models'].get('judge', '?')}`, "
          f"analyzer `{rep['run_models'].get('analyze', '?')}`. Planned calls {rep['plan']['total']} "
          f"(cap {rep['cap']}); live calls {rep['tokens']['live_calls']}, cached {rep['tokens']['cached_calls']}, "
          f"errors {rep['tokens']['errors']}; tokens in {rep['tokens']['input']:,} / out {rep['tokens']['output']:,}.",
          "", "Means with bootstrap 90% CI and n (judged clips). Verdict = pass / borderline / fail.", "",
          "| set | clips (judged) | pass rate | standalone 0-5 | keep_watching 0-5 | hook lands | verdicts p/b/f "
          "| validator clean | own checks clean | cuts a line |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name in (*SETS, "tool_excl_tuning"):
        s = st[name]
        v = s["verdicts"]
        L.append(f"| {name} | {s['n_clips']} ({s['n_judged']}) | {_fmt(s['pass_rate'], True)} | {_fmt(s['standalone'])} "
                 f"| {_fmt(s['keep_watching'])} | {_fmt(s['hook_lands'], True)} | {v['pass']}/{v['borderline']}/{v['fail']} "
                 f"| {_fmt(s['valid_validator'], True)} | {_fmt(s['valid_own_checks'], True)} | {s['cuts_a_line']} |")
    L += ["", "Validator issues by check (a clip can have several): "
          + "; ".join(f"{name}: " + (", ".join(f"{k} {v}" for k, v in sorted(st[name]["validator_issue_counts"].items()))
                                     or "none") for name in SETS) + ". Last-window clips keep the episode's final "
          "frames, so a trailing reaction shot can trip `cliff_at_end`; 'own checks' (length, hook in 3 s, no line "
          "cut) is the like-for-like structural comparison."]
    L += ["", "## Tool minus baseline", "", "| comparison | difference | 90% CI | n tool / n base |", "|---|---|---|---|"]
    for k, d in rep["diffs"].items():
        if d["diff"] is None:
            L.append(f"| {k} | n/a | | {d['n_tool']} / {d['n_base']} |")
            continue
        pct = k.endswith(".pass")
        f = (lambda x: f"{100 * x:+.0f} pts") if pct else (lambda x: f"{x:+.2f}")
        ci = f"[{f(d['ci90'][0])}, {f(d['ci90'][1])}]" if d["ci90"] else ""
        L.append(f"| {k} | {f(d['diff'])} | {ci} | {d['n_tool']} / {d['n_base']} |")
    a = rep["agreement"]
    L += ["", "## Judge agreement (tool clips)", "",
          f"Second judge: {a['second_judge']} {a.get('second_models', [])}. Pairs with both verdicts: n={a.get('n', 0)}."]
    if a.get("n"):
        k = a["kappa_pass"]
        sp = a["spearman_keep_watching"]
        L += [f"- verdict agreement (3-way): {100 * a['verdict_agreement']:.0f}%",
              f"- Cohen's kappa, pass vs not-pass: {'undefined (no variance)' if k is None else f'{k:.2f}'}",
              f"- Spearman on keep_watching: {'undefined' if sp is None else f'{sp:.2f}'}",
              f"- mean |standalone difference|: {a['mean_abs_diff_standalone']:.2f}",
              f"- pass rate, eval judge vs second: {100 * a['pass_rate_1']:.0f}% vs {100 * a['pass_rate_2']:.0f}%",
              "- Caveat: these clips were selected using the run's judge, so its pass rate is inflated and kappa is "
              "depressed by low variance."]
    sp = rep["diagnostics"]["spearman_prelim_vs_keep_watching_tool"]
    L += ["", "## Diagnostics", "",
          f"- Spearman(analyzer prelim_score, eval keep_watching) on tool clips: {'n/a' if sp is None else f'{sp:.2f}'}"]
    if rep["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in rep["warnings"]]
    L += ["", "Per-clip ids, timecodes and scores: eval.json. Judge free text (private): "
          "outputs/<run>/eval_verdicts.json.", ""]
    return "\n".join(L)


def write_report(rep: dict, out_dir: Path, private_path: Optional[Path]) -> None:
    private = rep.pop("_private", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "summary.md").write_text(summary_md(rep))
    if private_path is not None and private is not None:
        private_path.write_text(json.dumps(private, indent=2))


# ------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cold-viewer eval of a HookCut run vs baselines.")
    ap.add_argument("--run", required=True, help="outputs/<run>/result.json (or the folder)")
    ap.add_argument("--judge", help="provider:model; default: first of %s not used by the run" % (DEFAULT_JUDGES,))
    ap.add_argument("--judge2", default="run", help="'run' (stored run verdicts, 0 calls) or provider:model")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n2", type=int, help="clips re-judged by --judge2 (default: what fits under the cap)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-calls", type=int, default=MAX_JUDGE_CALLS, help=f"can only lower the {MAX_JUDGE_CALLS} cap")
    ap.add_argument("--env-file")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    ap.add_argument("--out", help="results folder (default eval/results/<run>)")
    a = ap.parse_args(argv)

    cap = min(a.max_calls, MAX_JUDGE_CALLS)
    result = load_run(a.run)
    spec, warnings = choose_judge(result, a.judge)
    sets = {"tool": tool_candidates(result, a.n), **baseline_sets(result.episodes, a.n, a.seed)}
    base_calls = sum(len(v) for v in sets.values())
    use2 = a.judge2 and a.judge2 != "run"
    if use2:
        n2 = a.n2 if a.n2 is not None else max(0, min(len(sets["tool"]), cap - base_calls))
        if family(a.judge2) == family(spec):
            warnings.append(f"judge2 {a.judge2} is the same family as the eval judge: agreement will look high")
    else:
        n2 = 0
        fams = used_families(result, "judge")
        if fams.get(family(spec), 0) * 2 > sum(fams.values()):
            warnings.append("judge2=run but most run verdicts came from the eval judge's family; pass --judge2")
    p = plan(sets, n2)
    print(f"plan (worst case, before cache): tool {p['tool']} + last_window {p['last_window']} + random {p['random']}"
          f" + judge2 {p['judge2']} ({a.judge2 if use2 else 'stored run verdicts'}) = {p['total']} judge calls "
          f"(cap {cap}); eval judge {spec}")
    for w in warnings:
        print(f"WARNING: {w}")
    try:
        check_budget(p["total"], cap)
    except BudgetExceeded as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if a.dry_run:
        return 0

    from hookcut.judge import judge_candidates          # lazy: the analyst's stage
    validate_fn = None
    try:
        from hookcut.validate import validate_promos     # lazy: the editor's stage
        if "analyses" in inspect.signature(validate_promos).parameters:   # enables the answer-inside check
            def validate_fn(ps, e, c):
                return validate_promos(ps, e, c, analyses=result.analyses)
        else:
            validate_fn = validate_promos
    except ImportError:
        warnings.append("hookcut.validate not importable: validator column is n/a")
    client = _build(spec, not a.no_cache, a.env_file)
    client2 = _build(a.judge2, not a.no_cache, a.env_file) if use2 and n2 else None
    log = RunLog()
    rep = evaluate(result, judge_fn=judge_candidates, client=client, n=a.n, client2=client2, n2=n2,
                   validate_fn=validate_fn, log=log, seed=a.seed, concurrency=a.concurrency, cap=cap)
    run_dir = resolve_run(a.run).parent
    rep = {"run": run_name(a.run), "created_at": dt.datetime.now().isoformat(timespec="seconds"),
           "judge": client.spec, "run_models": (result.config or {}).get("models") or {},
           "protocol": {"n": a.n, "seed": a.seed, "tuning_episodes": list(TUNING_EPISODES)},
           "warnings": warnings, **rep}
    out = Path(a.out) if a.out else results_dir(rep["run"])
    write_report(rep, out, run_dir / "eval_verdicts.json")
    log.write_jsonl(run_dir / "eval_log.jsonl")
    print(summary_md(rep))
    print(f"wrote {out / 'eval.json'}, {out / 'summary.md'}; private verdicts in {run_dir / 'eval_verdicts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
