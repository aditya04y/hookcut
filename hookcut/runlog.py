"""Records every model call (tokens, latency, cache, errors) so each run reports its own cost."""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .llm.base import LLMResult


class RunLog:
    def __init__(self, prices: Optional[dict] = None):
        self.rows: list[dict] = []
        self.prices = prices or {}
        self._lock = threading.Lock()
        self.t0 = time.time()

    def record(self, stage: str, item: str, result: Optional[LLMResult] = None, error: Optional[str] = None,
               **extra) -> None:
        row = {"ts": round(time.time() - self.t0, 2), "stage": stage, "item": item, **extra}
        if result is not None:
            row.update(provider=result.provider, model=result.model, input_tokens=result.input_tokens,
                       output_tokens=result.output_tokens, latency_s=result.latency_s, cached=result.cached,
                       attempts=result.attempts)
        if error:
            row["error"] = error[:500]
        with self._lock:
            self.rows.append(row)

    def note(self, stage: str, message: str, **extra) -> None:
        self.record(stage, "note", message=message, **extra)

    def _price(self, model: str) -> tuple[float, float]:
        p = self.prices.get(model) or {}
        return float(p.get("input") or 0.0), float(p.get("output") or 0.0)

    def summary(self) -> dict:
        by = defaultdict(lambda: {"calls": 0, "cached": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0,
                                  "latency_s": 0.0, "est_cost_usd": 0.0})
        for r in self.rows:
            if r.get("item") == "note":
                continue
            k = f"{r['stage']}|{r.get('provider', '?')}:{r.get('model', '?')}"
            s = by[k]
            s["calls"] += 1
            s["cached"] += int(bool(r.get("cached")))
            s["errors"] += int("error" in r)
            if not r.get("cached"):
                s["input_tokens"] += r.get("input_tokens", 0)
                s["output_tokens"] += r.get("output_tokens", 0)
                s["latency_s"] += r.get("latency_s", 0.0)
                pin, pout = self._price(r.get("model", ""))
                s["est_cost_usd"] += (r.get("input_tokens", 0) * pin + r.get("output_tokens", 0) * pout) / 1e6
        total = {k: round(sum(v[k] for v in by.values()), 4) for k in
                 ("calls", "cached", "errors", "input_tokens", "output_tokens", "latency_s", "est_cost_usd")}
        return {"wall_s": round(time.time() - self.t0, 1), "total": total,
                "by_stage_model": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                                   for k, v in by.items()}}

    def write_jsonl(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
