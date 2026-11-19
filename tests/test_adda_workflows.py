"""Exercise Adda workflows on the published sample and its offline fixture."""
import json
import re
from pathlib import Path

from hookcut import config
from hookcut.adda import runner
from hookcut.adda.quota import QuotaLedger
from hookcut.adda.router import Router
from hookcut.adda.tools import get_tool, list_tools, read_run
from hookcut.llm.base import LLMClient, LLMResult
from hookcut.llm.fallback import FallbackClient

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
DEMO = ROOT / "app/fixtures/demo_run.json"
LITE = "gemini:gemini-3.5-flash-lite"
OSS = "groq:openai/gpt-oss-120b"


def test_registry_cards_and_reuse():
    cards = {card["name"]: card for card in list_tools()}
    assert {
        "parse_episodes", "analyze_episodes", "build_candidates", "cold_viewer_test", "rank_promos",
        "write_why_lines", "episode_report", "season_summary", "validate_promos", "export_outputs", "build_studio",
    } <= cards.keys()
    analysis = cards["analyze_episodes"]
    assert analysis["inputs"]["required"] == ["episodes"]
    assert analysis["outputs"]["type"] == "array"
    assert analysis["uses_model"] == "analyze"
    review = runner.list_workflows()["script_review"]
    assert analysis["team"] == "promo" and review.team == "script"
    assert any(step.tool == "analyze_episodes" for step in review.steps)


def test_quick_check_needs_no_model_or_secrets(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("quick_check must not build a provider client")

    monkeypatch.setattr(config, "build_client", unexpected)
    monkeypatch.setattr(runner, "load_secrets", unexpected)
    router = Router(QuotaLedger(tmp_path / "quota.json"))
    out = tmp_path / "check"
    manifest = runner.run_workflow("quick_check", SAMPLES, out, router=router)
    validation = json.loads((out / "quick_check.json").read_bytes())
    assert validation["episodes"] == 4
    assert validation["beats"] > 0
    assert validation["warnings"] == []
    assert manifest == json.loads((out / "run_manifest.json").read_bytes())
    assert manifest["cost"]["calls"] == 0


def test_promo_finder_runs_with_fakes_and_judge_excludes_actual_analysis_families(tmp_path, monkeypatch):
    analyses = {a.episode: a.model_dump() for a in read_run(DEMO).analyses}
    build_specs = {}

    class FakeClient(LLMClient):
        def __init__(self, spec: str, task: str):
            provider, _, model = spec.partition(":")
            super().__init__(model)
            self.provider, self.task = provider, task

        def complete_json(self, system, user, schema=None, *, validate=None, **kwargs):
            if self.task == "analyze":
                episode = int(re.match(r"EPISODE (\d+)", user)[1])
                if self.provider == "gemini" and episode >= 2:
                    raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel retryDelay: 3600s")
                data = analyses[episode]
            elif self.task == "judge":
                assert user.startswith("CLIP (")
                data = {"premise": "Nora confronts a betrayal.", "open_question": "Who can she trust?",
                        "hook_lands": True, "standalone": 5, "keep_watching": 5}
            else:
                facts = [json.loads(line) for line in user.splitlines()[1:]]
                data = {"lines": [{"id": fact["id"], "why": "The reveal raises a question the cut leaves open."}
                                  for fact in facts]}
            if validate is not None:
                validate(data)
            return LLMResult(data, json.dumps(data), self.provider, self.model, input_tokens=1000, output_tokens=100)

    def fake_build_client(cfg, stage, secrets, use_cache):
        specs = build_specs[stage] = [cfg["models"][stage], *cfg["fallbacks"][stage]]
        return FallbackClient([FakeClient(spec, stage) for spec in specs])

    monkeypatch.setattr(config, "build_client", fake_build_client)
    monkeypatch.setattr(runner, "load_secrets", lambda: {})
    cfg = config.load_config()
    cfg["promo"]["embeddings"] = "jaccard"
    cfg["concurrency"]["analyze"] = 1
    ledger = QuotaLedger(tmp_path / "quota.json")
    router = Router(ledger)
    out = tmp_path / "promo"
    manifest = runner.run_workflow("promo_finder", SAMPLES, out, router=router, cfg=cfg, use_cache=False)
    result = read_run(out)
    assert len(result.episodes) == len(result.analyses) == 4
    assert result.candidates and result.verdicts and result.promos
    assert all("leaves open" in promo.why_it_works for promo in result.promos)
    assert sorted({a.model for a in result.analyses}) == sorted((LITE, OSS))
    assert set(build_specs["judge"]) == {"groq:qwen/qwen3.8-27b"}
    assert all(a.model == build_specs["judge"][0] for a in result.verdicts)
    assert ledger.is_exhausted(LITE)
    steps = {step["tool"]: step for step in manifest["steps"]}
    assert steps["analyze_episodes"]["models"] == sorted((LITE, OSS))
    assert steps["cold_viewer_test"]["excluded_families"] == ["gemini", "gpt-oss"]
    assert steps["cold_viewer_test"]["est_cost_usd"] > 0
    assert {"result.json", "promos.json", "episode_report.md", "run_summary.json"} <= manifest["artifacts"].keys()
    assert manifest == json.loads((out / "run_manifest.json").read_bytes())

    review_out = tmp_path / "review"
    review = runner.run_workflow("script_review", SAMPLES, review_out, router=router, cfg=cfg)
    assert len(read_run(review_out).episode_report) == 4
    assert json.loads((review_out / "season_summary.json").read_bytes())["analyzed"] == 4
    graph_status = "complete" if get_tool("story_graph") else "skipped"
    assert review["steps"][-1]["status"] == graph_status


def test_audience_update_uses_fixture_without_changing_it(tmp_path):
    before = DEMO.read_bytes()
    out = tmp_path / "audience"
    manifest = runner.run_workflow("audience_update", DEMO, out, simulate=True)
    studio = json.loads((out / "studio.json").read_bytes())
    assert studio["source"].casefold() == "simulated"
    assert studio["rows"]
    assert "matrix" not in studio
    assert DEMO.read_bytes() == before
    status = "complete" if get_tool("engagement") else "skipped"
    assert manifest["steps"][-1]["status"] == status
