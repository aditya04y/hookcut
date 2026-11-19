"""Tiny on-disk cache so reruns (and interview demos) don't spend quota twice."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    def __init__(self, folder: str | Path = ".cache/llm", enabled: bool = True):
        self.folder = Path(folder)
        self.enabled = enabled
        if enabled:
            self.folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        p = self.folder / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def put(self, key: str, value: dict) -> None:
        if self.enabled:
            (self.folder / f"{key}.json").write_text(json.dumps(value))
