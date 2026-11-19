"""Model policy, quota expiry, and fallback behavior with no provider calls."""
from datetime import datetime, timezone

from hookcut.adda.quota import QuotaLedger, with_quota
from hookcut.adda.router import Router
from hookcut.config import load_config
from hookcut.llm.base import LLMClient, LLMResult
from hookcut.llm.fallback import FallbackClient

GEMINI = "gemini:gemini-3.8-flash"
GROQ = "groq:openai/gpt-oss-120b"
GEMINI_429 = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "80432s"},
        ],
    },
}
GROQ_429 = "Error 429: Rate limit reached on tokens per day (TPD). Please try again in 2h3m4.5s."


def test_ledger_reads_daily_reset_and_ignores_minute_limits(tmp_path):
    now = [1_800_000_000.0]
    path = tmp_path / "quota.json"
    ledger = QuotaLedger(path, clock=lambda: now[0])
    assert not ledger.record_error(GEMINI, {
        "error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                  "details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel", "retryDelay": "45s"}]}
    })
    assert not ledger.record_error(GROQ, "429 rate limit reached on tokens per minute (TPM). Try again in 30s.")
    assert not ledger.record_error(GROQ, "HTTP 403: tokens per day unavailable for this account")
    assert not path.exists()
    assert ledger.record_error(GEMINI, GEMINI_429)
    assert ledger.record_error(GROQ, GROQ_429)

    restored = QuotaLedger(path, clock=lambda: now[0])
    status = restored.status()
    assert datetime.fromisoformat(status[GEMINI]["reset_at"]) == datetime.fromtimestamp(now[0] + 80432, timezone.utc)
    assert datetime.fromisoformat(status[GROQ]["reset_at"]) == datetime.fromtimestamp(now[0] + 7384.5, timezone.utc)
    assert all(restored.is_exhausted(spec) for spec in (GEMINI, GROQ))
    now[0] += 80432
    assert restored.status() == {}


def test_router_respects_tiers_families_price_and_quota(tmp_path):
    ledger = QuotaLedger(tmp_path / "quota.json")
    router = Router(ledger)
    assert router.choose("analyze", quality="quality")[0] == GEMINI
    assert router.choose("analyze", quality="budget")[0] == GROQ
    assert router.choose("analyze")[0] == "gemini:gemini-3.5-flash-lite"
    assert router.choose("judge", exclude_families=("Gemini", "gpt-oss")) == ["groq:qwen/qwen3.8-27b"]
    assert router.family("openrouter:openai/gpt-oss-120b") == "gpt-oss"
    assert router.choose("analyze", max_cost=1.0) == [GROQ]
    assert ledger.record_error(GEMINI, GEMINI_429)
    assert GEMINI not in router.choose("analyze", quality="quality")
    choice = next(row for row in router.decisions("analyze") if row["spec"] == GEMINI)
    assert "Daily quota resets" in choice["reason"]
    prices = load_config()["prices_usd_per_mtok"]
    for task in router.tasks:
        for row in router.decisions(task):
            assert row["reason"]
            price = prices[row["spec"].split(":", 1)[1]]
            assert (row["price_in"], row["price_out"]) == (price["input"], price["output"])


def test_quota_guard_remembers_a_fallback_failure_before_error_aggregation(tmp_path):
    class FakeClient(LLMClient):
        calls = 0

        def complete_json(self, *args, **kwargs):
            self.calls += 1
            if self.spec == GROQ:
                raise RuntimeError(GROQ_429)
            return LLMResult({}, "{}", self.provider, self.model)

    primary = FakeClient(GROQ.split(":", 1)[1])
    primary.provider = "groq"
    backup = FakeClient("qwen/qwen3.8-27b")
    backup.provider = "groq"
    ledger = QuotaLedger(tmp_path / "quota.json")
    client = with_quota(FallbackClient([primary, backup]), ledger)
    for _ in range(2):
        assert client.complete_json("system", "user").model == backup.model
    assert ledger.is_exhausted(GROQ)
    assert primary.calls == 1
    assert backup.calls == 2
