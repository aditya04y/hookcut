"""Model-neutral client layer: JSON extraction, validate->re-ask, cache, fallback, spec parsing. No network."""
import pytest
from pydantic import BaseModel

from hookcut.llm.base import LLMClient, LLMError, extract_json
from hookcut.llm.cache import DiskCache
from hookcut.llm.fallback import FallbackClient
from hookcut.llm.registry import parse_spec


class Scripted(LLMClient):
    """Returns canned replies in order; counts calls."""
    provider = "fake"

    def __init__(self, replies, **kw):
        super().__init__("fake-model", **kw)
        self.replies = list(replies)
        self.calls = 0

    def _call(self, system, user, json_mode):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 100, 20


class Pick(BaseModel):
    beat_id: str
    score: int


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Sure! Here it is: {"a": 3} hope that helps') == {"a": 3}


def test_validate_failure_triggers_reask():
    c = Scripted(['{"beat_id": "E01.B999", "score": 4}', '{"beat_id": "E01.B002", "score": 4}'])

    def check(d):
        if d["beat_id"] != "E01.B002":
            raise ValueError(f"unknown beat id {d['beat_id']}")

    r = c.complete_json("sys", "user", Pick, validate=check)
    assert r.data["beat_id"] == "E01.B002" and r.attempts == 2 and c.calls == 2
    assert r.input_tokens == 200 and r.output_tokens == 40


def test_schema_failure_exhausts_retries():
    c = Scripted(['{"beat_id": 1}', "not json", '{"score": "x"}'])
    with pytest.raises(LLMError):
        c.complete_json("sys", "user", Pick, retries=2)


def test_cache_hit_skips_model(tmp_path):
    cache = DiskCache(tmp_path)
    a = Scripted(['{"beat_id": "E01.B001", "score": 3}'], cache=cache)
    a.complete_json("sys", "user", Pick)
    b = Scripted([], cache=cache)               # would crash if called
    r = b.complete_json("sys", "user", Pick)
    assert r.cached and r.data["score"] == 3 and b.calls == 0


def test_fallback_uses_next_model():
    bad = Scripted([TimeoutError("slow")])
    good = Scripted(['{"beat_id": "E01.B001", "score": 5}'])
    r = FallbackClient([bad, good]).complete_json("sys", "user", Pick)
    assert r.data["score"] == 5 and good.calls == 1


def test_fallback_all_fail():
    with pytest.raises(LLMError):
        FallbackClient([Scripted([RuntimeError("x")]), Scripted([RuntimeError("y")])]).complete_json("s", "u")


def test_parse_spec():
    assert parse_spec("groq:openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    with pytest.raises(ValueError):
        parse_spec("nomodel")
    with pytest.raises(ValueError):
        parse_spec("acme:model")


def test_build_client_skips_models_with_exhausted_daily_quota(tmp_path, monkeypatch):
    from hookcut import config as cfgmod
    from hookcut.adda.quota import QuotaLedger
    ledger_path = tmp_path / "quota.json"
    ledger = QuotaLedger(ledger_path)
    payload = ('429 RESOURCE_EXHAUSTED {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", '
               '"retryDelay": "80432s"}')
    ledger.record_error("gemini:gemini-3.8-flash", payload)
    cfg = {"models": {"analyze": "gemini:gemini-3.8-flash"},
           "fallbacks": {"analyze": ["gemini:gemini-3.5-flash-lite"]},
           "quota_ledger": str(ledger_path), "cache_dir": str(tmp_path / "cache")}
    client = cfgmod.build_client(cfg, "analyze", secrets={"GEMINI_API_KEY": "x"}, use_cache=False)
    assert client.spec == "gemini:gemini-3.5-flash-lite"      # exhausted model skipped
