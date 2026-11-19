"""Load config.yaml, apply env/CLI overrides, and build the per-stage LLM clients."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional

import yaml

from .llm.base import LLMClient
from .llm.cache import DiskCache
from .llm.fallback import FallbackClient
from .llm.registry import load_secrets, make_client

ROOT = Path(__file__).resolve().parent.parent
STAGES = ("analyze", "judge", "writer", "genome")


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    cfg = copy.deepcopy(cfg or {})
    cfg.setdefault("models", {})
    for stage in STAGES:  # env overrides, e.g. HOOKCUT_JUDGE_MODEL=groq:openai/gpt-oss-120b
        if v := os.environ.get(f"HOOKCUT_{stage.upper()}_MODEL"):
            cfg["models"][stage] = v
            cfg.setdefault("fallbacks", {})[stage] = []
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k in STAGES:
            cfg["models"][k] = v
            cfg.setdefault("fallbacks", {})[k] = []   # an explicit choice means no silent substitution
        else:
            cfg[k] = v
    return cfg


def build_client(cfg: dict, stage: str, secrets: Optional[dict] = None, use_cache: bool = True) -> LLMClient:
    """Configured model first, then its fallbacks. The AI Adda quota ledger skips models whose provider reported an
    exhausted DAILY quota (remembered in .cache/quota.json until the reset time), and every client records new
    daily-quota errors so the next run starts on a live model instead of burning retries."""
    from .adda.quota import QuotaLedger, with_quota   # lazy import: adda imports this module
    secrets = secrets if secrets is not None else load_secrets()
    cache = DiskCache(ROOT / cfg.get("cache_dir", ".cache/llm"), enabled=use_cache)
    opts = (cfg.get("model_options") or {}).get(stage, {})
    specs = [cfg["models"][stage]] + list((cfg.get("fallbacks") or {}).get(stage, []))
    ledger = QuotaLedger(ROOT / cfg.get("quota_ledger", ".cache/quota.json"))
    specs = [s for s in specs if not ledger.is_exhausted(s)] or specs   # all exhausted: try anyway (may have reset)
    clients = []
    for spec in specs:
        try:
            clients.append(with_quota(make_client(spec, secrets=secrets, cache=cache, **opts), ledger))
        except Exception:
            continue  # missing key for a fallback provider: skip it
    if not clients:
        raise RuntimeError(f"no usable model for stage {stage!r} (tried {specs})")
    return clients[0] if len(clients) == 1 else FallbackClient(clients)
