"""Remember provider-reported daily quota exhaustion across runs."""
from __future__ import annotations

import fcntl
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from ..config import ROOT
from ..llm.base import LLMError
from ..llm.fallback import FallbackClient

DAILY = re.compile(
    r"per[\s_]*day\b|perdayper|/\s*day\b|\b(?:tpd|rpd)\b|daily[\s_-]+(?:token|request|quota|limit)",
    re.I,
)
DELAY = re.compile(r"""retryDelay['"]?\s*:\s*['"]?\s*(\d+(?:\.\d+)?)\s*s|try again in\s+([\d.hms\s]+)""", re.I)
UNITS = {"h": 3600, "m": 60, "s": 1}


def _daily_limit(error: Exception | str | dict) -> tuple[str, float] | None:
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    body = getattr(error, "body", None)
    parts = [str(error)]
    if body is not None:
        parts.append(json.dumps(body, default=str))
    message = " ".join(parts)
    if status is None and isinstance(error, dict):
        details = error.get("error", error)
        if isinstance(details, dict) and isinstance(details.get("code"), int):
            status = details["code"]
    if status is None and (match := re.search(
        r"\b(?:http|error(?:\s+code)?|status(?:\s+code)?)\s*[:=]?\s*([45]\d{2})\b", message, re.I
    )):
        status = int(match[1])
    if (status is not None and status != 429) or not DAILY.search(message):
        return None
    if status is None and not re.search(
        r"\b429\b|resource_exhausted|rate[\s_-]*limit|tokens per day|requests per day", message, re.I
    ):
        return None

    wait = None
    if match := DELAY.search(message):
        if match[1]:
            wait = float(match[1])
        else:
            wait = sum(float(number) * UNITS[unit.lower()]
                       for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", match[2], re.I))
    if wait is None and response is not None:
        retry_after = str(getattr(response, "headers", {}).get("retry-after", ""))
        if re.fullmatch(r"\d+(?:\.\d+)?", retry_after):
            wait = float(retry_after)
    reason = "Daily token quota" if re.search(r"tokens per day|\btpd\b|daily token", message, re.I) else "Daily quota"
    return reason, wait if wait is not None else 86400


class QuotaLedger:
    def __init__(self, path: str | Path = ROOT / ".cache/quota.json", *, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.clock = clock

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _active(self) -> dict:
        now = self.clock()
        return {spec: row for spec, row in self._read().items()
                if isinstance(row, dict) and isinstance(row.get("reset_at"), (int, float))
                and row["reset_at"] > now}

    def status(self) -> dict[str, dict]:
        """Active limits keyed by model spec, with reset times in UTC."""
        return {spec: {"reset_at": datetime.fromtimestamp(row["reset_at"], timezone.utc).isoformat(),
                       "reason": row.get("reason", "Daily quota")}
                for spec, row in sorted(self._active().items())}

    def is_exhausted(self, spec: str) -> bool:
        return spec in self._active()

    def record_error(self, spec: str, exc_or_message: Exception | str | dict) -> bool:
        if (limit := _daily_limit(exc_or_message)) is None:
            return False
        reason, wait = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize the read/merge/replace across workers. Readers see complete snapshots.
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active = self._active()
            reset = self.clock() + wait
            if reset > active.get(spec, {}).get("reset_at", 0):
                active[spec] = {"reset_at": reset, "reason": reason}
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as file:
                    pending = Path(file.name)
                    try:
                        json.dump(active, file, indent=2)
                        file.close()
                        pending.replace(self.path)
                    finally:
                        pending.unlink(missing_ok=True)
        return True


class _QuotaClient:
    def __init__(self, client, ledger: QuotaLedger):
        self.client = client
        self.ledger = ledger

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete_json(self, *args, **kwargs):
        if self.ledger.is_exhausted(self.spec):
            raise LLMError(f"{self.spec}: daily quota exhausted")
        try:
            return self.client.complete_json(*args, **kwargs)
        except Exception as exc:
            self.ledger.record_error(self.spec, exc)
            raise


def with_quota(client, ledger: QuotaLedger):
    """Guard each fallback separately so provider errors are seen before aggregation."""
    if isinstance(client, FallbackClient):
        return FallbackClient([with_quota(member, ledger) for member in client.clients])
    if isinstance(client, _QuotaClient) and client.ledger is ledger:
        return client
    return _QuotaClient(client, ledger)
