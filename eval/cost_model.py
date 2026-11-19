"""What HookCut costs at real-world volume, from a run's own token log.

  .venv/bin/python -m eval.cost_model --run outputs/<run> [--shows 200 --episodes 50]

Measured: tokens per call / per episode / per show from run_log.jsonl (cached rows carry the tokens of the
original call, so a fully cached rerun still measures a fresh run). Priced: per model choice, from the YAML block
at the end of docs/research/open_tools.md plus the Claude prices in the lead's brief. Free tier: shows/day under each
model's daily caps. Human time: labelled assumptions. Writes eval/results/<run>/cost.md and cost.json.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import yaml

from .common import ROOT, resolve_run, results_dir, run_name

OPEN_TOOLS = ROOT / "research" / "open_tools.md"
EXTRA_PRICES = {   # USD per 1M tokens, given in the lead's brief (2026-11-19); not in open_tools.md
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "note": "lead's brief 2026-11-19"},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "note": "lead's brief 2026-11-19"},
    "gemini-3.8-flash@2027": {"input": 1.50, "output": 7.50, "note": "open_tools.md: price from 2027-01-01"},
}
# Free-tier caps per model (docs/research/open_tools.md, 2026-11-19; Groq per model). Gemini's per-model caps are
# shown only in AI Studio; 3.8 Flash's 20 requests/day is the lead's observation noted in config.yaml.
FREE_CAPS = {
    "openai/gpt-oss-120b": {"tpd": 200_000, "rpd": 1_000, "tpm": 8_000},
    "openai/gpt-oss-20b": {"tpd": 200_000, "rpd": 1_000, "tpm": 8_000},
    "qwen/qwen3.8-27b": {"tpd": 200_000, "rpd": 1_000, "tpm": 8_000},
    "gemini-3.8-flash": {"rpd": 20},     # config.yaml note (observed); TPM/TPD: TODO from AI Studio
    "gemini-3.5-flash-lite": None,       # TODO: AI Studio. config.yaml: "free tier covers a 50-episode show in a day"
}
SCENARIOS = [  # (label, {stage: model}); "*" = every stage
    ("As configured, paid tiers (3.5 Flash-Lite analyze + Qwen3.8 27B judge + gpt-oss-120b writer)",
     {"analyze": "gemini-3.5-flash-lite", "judge": "qwen/qwen3.8-27b", "writer": "openai/gpt-oss-120b"}),
    ("Quality preset (3.8 Flash analyze + Qwen3.8 27B judge + gpt-oss-120b writer)",
     {"analyze": "gemini-3.8-flash", "judge": "qwen/qwen3.8-27b", "writer": "openai/gpt-oss-120b"}),
    ("All Groq gpt-oss-120b (paid)", {"*": "openai/gpt-oss-120b"}),
    ("All Groq gpt-oss-20b (paid)", {"*": "openai/gpt-oss-20b"}),
    ("All Gemini 3.8 Flash (paid, 2026 price)", {"*": "gemini-3.8-flash"}),
    ("All Gemini 3.8 Flash (paid, from 2027-01-01)", {"*": "gemini-3.8-flash@2027"}),
    ("Gemini 3.5 Flash-Lite analyze + gpt-oss-20b judge/writer",
     {"analyze": "gemini-3.5-flash-lite", "*": "openai/gpt-oss-20b"}),
    ("All Claude Opus 5.5", {"*": "claude-opus-5-5"}),
    ("All Claude Haiku 5.5", {"*": "claude-haiku-5-5"}),
    ("Claude Opus 5.5 analyze + Haiku 5.5 judge/writer", {"analyze": "claude-opus-5-5", "*": "claude-haiku-5-5"}),
]
# Human side: ASSUMPTIONS, not measurements. Change them here; the report prints them.
HUMAN = {
    "watch_min_per_episode": None,      # None = the show's measured mean episode length
    "pick_overhead_min_per_show": 30,   # ASSUMPTION: scrubbing back, noting timecodes, choosing
    "review_min_per_item": 1.5,         # ASSUMPTION: read hook/cliff/why + watch a 20-60 s render
    "shortlist_items": 20,
}


# ------------------------------------------------------------------ inputs
def load_prices(path: Path = OPEN_TOOLS, config_prices: Optional[dict] = None) -> dict[str, dict]:
    """prices_usd_per_mtok from the ```yaml block(s) in open_tools.md, then config.yaml, then EXTRA_PRICES."""
    prices: dict[str, dict] = {}
    if path.exists():
        for block in re.findall(r"```yaml\s*\n(.*?)```", path.read_text(), re.S):
            try:
                d = yaml.safe_load(block)
            except yaml.YAMLError:
                continue
            if isinstance(d, dict) and isinstance(d.get("prices_usd_per_mtok"), dict):
                prices.update({k: dict(v, note=v.get("note", "open_tools.md")) for k, v in
                               d["prices_usd_per_mtok"].items()})
    prices.update(config_prices or {})
    for k, v in EXTRA_PRICES.items():
        prices.setdefault(k, v)
    return prices


def load_rows(run: str | Path) -> tuple[list[dict], dict]:
    """(run log rows, run meta) from a run folder: run_log.jsonl, else result.json's run_log."""
    rj = resolve_run(run)
    folder = rj.parent
    meta: dict = {}
    rows: list[dict] = []
    if (folder / "run_log.jsonl").exists():
        rows = [json.loads(line) for line in (folder / "run_log.jsonl").read_text().splitlines() if line.strip()]
    if rj.exists():
        r = json.loads(rj.read_text())
        rows = rows or r.get("run_log", [])
        eps = r.get("episodes", [])
        meta = {"show": r.get("show"), "episodes": len(eps), "promo": (r.get("config") or {}).get("promo") or {},
                "models": (r.get("config") or {}).get("models") or {},
                "mean_episode_s": (sum(e.get("duration", 0) for e in eps) / len(eps)) if eps else None}
    if (folder / "run_summary.json").exists():
        s = json.loads((folder / "run_summary.json").read_text())
        meta.setdefault("episodes", s.get("episodes"))
        meta["wall_s"] = (s.get("cost") or {}).get("wall_s")
    return rows, meta


# ------------------------------------------------------------------ measure
def measure(rows: list[dict]) -> dict[str, dict]:
    """Per stage: calls, errors, tokens (cached rows included: they carry the original call's tokens)."""
    by: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "cached": 0, "input": 0, "output": 0,
                                               "max_call_tokens": 0, "latency_s": 0.0, "items": set(),
                                               "models": Counter()})
    for r in rows:
        if r.get("item") == "note":
            continue
        s = by[r["stage"]]
        if "error" in r and "input_tokens" not in r:
            s["errors"] += 1
            continue
        s["calls"] += 1
        s["cached"] += int(bool(r.get("cached")))
        s["input"] += r.get("input_tokens", 0)
        s["output"] += r.get("output_tokens", 0)
        s["max_call_tokens"] = max(s["max_call_tokens"], r.get("input_tokens", 0) + r.get("output_tokens", 0))
        s["latency_s"] += r.get("latency_s", 0.0) or 0.0
        s["items"].add(r.get("item"))
        if r.get("model"):
            s["models"][r["model"]] += 1
    return {k: {**v, "items": len(v["items"]), "models": dict(v["models"].most_common())} for k, v in by.items()}


def fresh_calls(folders: Sequence[str | Path]) -> list[dict]:
    """Per (stage, model): mean tokens and latency of LIVE (uncached, successful) calls in other runs' logs:
    a check that the tokens carried by cached rows match what fresh calls cost."""
    acc: dict[tuple, list] = defaultdict(list)
    for f in folders:
        rows, _ = load_rows(f)
        for r in rows:
            if r.get("item") == "note" or r.get("cached") or "error" in r or not r.get("model"):
                continue
            acc[(r["stage"], r["model"])].append(r)
    return [{"stage": st, "model": m, "calls": len(rs),
             "mean_input": sum(r.get("input_tokens", 0) for r in rs) / len(rs),
             "mean_output": sum(r.get("output_tokens", 0) for r in rs) / len(rs),
             "mean_latency_s": sum(r.get("latency_s", 0) or 0 for r in rs) / len(rs)}
            for (st, m), rs in sorted(acc.items())]


def per_show(m: dict[str, dict], run_episodes: int, show_episodes: int = 50, judge_top_k: int = 40) -> dict:
    """Tokens for one show of `show_episodes` episodes: analyze scales per episode, judge per finalist,
    everything else (writer) is per show as measured."""
    out = {}
    for stage, s in m.items():
        if not s["calls"]:
            continue
        if stage == "analyze":
            k = show_episodes / max(1, run_episodes)
            basis = f"{s['calls']} calls for {run_episodes} episodes, x{k:.2f}"
        elif stage == "judge":
            k = judge_top_k / s["calls"]
            basis = f"{s['calls']} calls, scaled to {judge_top_k} finalists"
        else:
            k = 1.0
            basis = "per show, as measured"
        out[stage] = {"input": s["input"] * k, "output": s["output"] * k, "calls": s["calls"] * k, "basis": basis}
    return out


def scenario_cost(show: dict, mapping: dict[str, str], prices: dict[str, dict]) -> dict:
    total, missing, parts = 0.0, [], {}
    for stage, t in show.items():
        model = mapping.get(stage, mapping.get("*"))
        p = prices.get(model or "")
        if not p:
            missing.append(f"{stage}:{model}")
            continue
        c = (t["input"] * float(p["input"]) + t["output"] * float(p["output"])) / 1e6
        parts[stage] = c
        total += c
    return {"per_show_usd": total, "by_stage": parts, "missing_prices": missing}


def free_capacity(show: dict, models: dict[str, str]) -> list[dict]:
    """Shows/day each stage's free model allows (caps: tokens/day, requests/day, tokens/minute; any may be
    missing). `models`: stage -> the model that actually answered most calls (or a provider:model spec)."""
    out = []
    for stage, t in show.items():
        spec = models.get(stage, "")
        model = spec.split(" -> ")[0].split(":", 1)[-1] if ":" in spec.split(" -> ")[0] else spec
        caps = FREE_CAPS.get(model, "unknown")
        tokens = t["input"] + t["output"]
        if not caps:
            out.append({"stage": stage, "model": model, "shows_per_day": None,
                        "note": "TODO: per-model free caps only in AI Studio"})
            if stage == "analyze":   # its configured fallback runs on Groq, whose caps are known
                out += free_capacity({stage: t}, {stage: "groq:openai/gpt-oss-120b"})
                out[-1]["note"] = ("if analysis falls back to Groq gpt-oss-120b (shares the writer's quota). "
                                   + out[-1]["note"]).strip()
            continue
        if caps == "unknown":
            out.append({"stage": stage, "model": model, "shows_per_day": None, "note": "no free-tier caps recorded"})
            continue
        by_tok = caps["tpd"] / tokens if tokens and "tpd" in caps else float("inf")
        by_req = caps["rpd"] / t["calls"] if t["calls"] and "rpd" in caps else float("inf")
        per_call = tokens / t["calls"] if t["calls"] else 0
        row = {"stage": stage, "model": model, "shows_per_day": min(by_tok, by_req),
               "note": "requests/day cap only (token caps TODO)" if "tpd" not in caps else ""}
        if "tpm" in caps:
            row["minutes_per_show_at_tpm"] = tokens / caps["tpm"]
            if per_call > caps["tpm"]:
                row["note"] = (f"a mean call (~{per_call:,.0f} tok) is above the {caps['tpm']:,} tokens/min cap: "
                               "expect rate-limit retries and long waits")
        out.append(row)
    return out


def human_time(mean_episode_s: Optional[float], episodes: int = 50, h: dict = HUMAN) -> dict:
    watch = h["watch_min_per_episode"] or ((mean_episode_s or 120) / 60)
    today = episodes * watch + h["pick_overhead_min_per_show"]
    hook = h["review_min_per_item"] * h["shortlist_items"]
    return {**h, "today_min_per_show": today, "hookcut_min_per_show": hook, "watch_min_per_episode": watch}


# ------------------------------------------------------------------ report
def build(run: str | Path, shows: int = 200, episodes: int = 50, prices: Optional[dict] = None,
          fresh: Sequence[str | Path] = ()) -> dict:
    rows, meta = load_rows(run)
    m = measure(rows)
    run_eps = meta.get("episodes") or m.get("analyze", {}).get("items") or 1
    top_k = int((meta.get("promo") or {}).get("judge_top_k", 40))
    show = per_show(m, run_eps, episodes, top_k)
    if prices is None:
        from hookcut.config import load_config
        prices = load_prices(config_prices=load_config().get("prices_usd_per_mtok") or {})
    tok_show = sum(t["input"] + t["output"] for t in show.values())
    scen = []
    for label, mapping in SCENARIOS:
        c = scenario_cost(show, mapping, prices)
        scen.append({"label": label, "mapping": mapping, **c, "per_month_usd": c["per_show_usd"] * shows,
                     "per_episode_usd": c["per_show_usd"] / episodes})
    return {"run": run_name(run), "run_episodes": run_eps, "measured": m,   # no show title: results are public
            "per_show_tokens": show, "tokens_per_show": tok_show, "tokens_per_episode": tok_show / episodes,
            "volume": {"shows_per_month": shows, "episodes_per_show": episodes,
                       "tokens_per_month": tok_show * shows},
            "scenarios": scen,
            "free_tier": free_capacity(show, {st: next(iter(s["models"]), "") for st, s in m.items()}),
            "fresh_calls": fresh_calls(fresh) if fresh else [],
            "human": human_time(meta.get("mean_episode_s"), episodes), "wall_s": meta.get("wall_s"),
            "prices_used": {k: prices[k] for k in sorted({v for _, mp in SCENARIOS for v in mp.values()}) if k in prices}}


def cost_md(r: dict) -> str:
    v, h = r["volume"], r["human"]
    L = [f"# Cost at volume: {r['run']}", "",
         f"Measured on {r['run_episodes']} episodes (run `{r['run']}`)"
         + (f" (wall time {r['wall_s'] / 60:.1f} min" + (", mostly cache hits: see live-call latency below)"
            if sum(x["cached"] for x in r["measured"].values()) * 2 > sum(x["calls"] for x in r["measured"].values())
            else ")") if r.get("wall_s") else "")
         + f". Volume: {v['shows_per_month']} shows x {v['episodes_per_show']} episodes = "
         f"{v['shows_per_month'] * v['episodes_per_show']:,} episodes/month.", "",
         "## Measured tokens", "", "| stage | calls (cached) | errors | input tok | output tok | max per call | models |",
         "|---|---|---|---|---|---|---|"]
    for st, s in r["measured"].items():
        L.append(f"| {st} | {s['calls']} ({s['cached']}) | {s['errors']} | {s['input']:,} | {s['output']:,} | "
                 f"{s['max_call_tokens']:,} | {', '.join(f'{k} x{v}' for k, v in s['models'].items())} |")
    L += ["", "Cached rows carry the token counts of the original call, so a cached rerun still measures a fresh run."]
    if r.get("fresh_calls"):
        L += ["", "Check against live (uncached) calls in other runs:", "",
              "| stage | model | live calls | mean input | mean output | mean latency |", "|---|---|---|---|---|---|"]
        for f in r["fresh_calls"]:
            L.append(f"| {f['stage']} | {f['model']} | {f['calls']} | {f['mean_input']:,.0f} | {f['mean_output']:,.0f} "
                     f"| {f['mean_latency_s']:.1f} s |")
    L += ["", "## Per show (50 episodes)", "", "| stage | input tok | output tok | calls | basis |", "|---|---|---|---|---|"]
    for st, t in r["per_show_tokens"].items():
        L.append(f"| {st} | {t['input']:,.0f} | {t['output']:,.0f} | {t['calls']:.0f} | {t['basis']} |")
    L += ["", f"Total ~{r['tokens_per_show']:,.0f} tokens per show (~{r['tokens_per_episode']:,.0f} per episode); "
          f"~{v['tokens_per_month'] / 1e6:,.0f}M tokens/month.", "",
          "## Price per model choice (same token counts for every model)", "",
          "| models | per episode | per show | per month |", "|---|---|---|---|"]
    for s in r["scenarios"]:
        if s["missing_prices"]:
            L.append(f"| {s['label']} | TODO: no price for {', '.join(s['missing_prices'])} | | |")
        else:
            L.append(f"| {s['label']} | ${s['per_episode_usd']:.4f} | ${s['per_show_usd']:.2f} | "
                     f"${s['per_month_usd']:,.0f} |")
    L += ["", "Caveats: token counts are the run's models'; other models tokenize differently and reasoning models "
          "bill thinking as output (Gemini output prices include thinking; the OpenAI-compatible endpoint may not "
          "report all thinking tokens, so Gemini rows are a floor). Groq and Gemini batch tiers are ~50% "
          "off; prompt caching of the fixed system prompts lowers input cost further. Free-tier Gemini data may be "
          "used to improve Google's products, so production scripts belong on a paid tier.", "",
          "## Free tier: how many shows per day fit", "", "| stage | model | shows/day | min per show at TPM cap | note |",
          "|---|---|---|---|---|"]
    for f in r["free_tier"]:
        spd = "unknown" if f["shows_per_day"] is None else f"{f['shows_per_day']:.2f}"
        mins = f"{f['minutes_per_show_at_tpm']:.0f}" if "minutes_per_show_at_tpm" in f else ""
        L.append(f"| {f['stage']} | {f['model']} | {spd} | {mins} | {f['note']} |")
    L += ["", "The slowest stage sets the pace; Groq caps are per model, so the writer and an eval judge on the same "
          "model share one daily budget.", "",
          "## Human time (ASSUMPTIONS, not measured)", "",
          f"- Today: watch {v['episodes_per_show']} episodes x {h['watch_min_per_episode']:.1f} min + "
          f"{h['pick_overhead_min_per_show']} min picking = **~{h['today_min_per_show']:.0f} min per show** before a "
          f"first promo is chosen; {v['shows_per_month']} shows = ~{h['today_min_per_show'] * v['shows_per_month'] / 60:,.0f} h/month.",
          f"- With HookCut: review a {h['shortlist_items']}-item shortlist at {h['review_min_per_item']} min each = "
          f"**~{h['hookcut_min_per_show']:.0f} min per show**; ~{h['hookcut_min_per_show'] * v['shows_per_month'] / 60:,.0f} h/month.",
          "- Not counted: rendering, posting, and the human's call on taste. The shortlist exists the day the script "
          "locks, before any video.", ""]
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cost of HookCut at volume, from a run's token log.")
    ap.add_argument("--run", required=True, help="outputs/<run> or its result.json")
    ap.add_argument("--shows", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--fresh", nargs="*", default=[], help="other run folders whose live calls cross-check tokens")
    ap.add_argument("--out", help="results folder (default eval/results/<run>)")
    a = ap.parse_args(argv)
    r = build(a.run, a.shows, a.episodes, fresh=a.fresh)
    out = Path(a.out) if a.out else results_dir(r["run"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "cost.json").write_text(json.dumps(r, indent=2, default=str))
    md = cost_md(r)
    (out / "cost.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
