"""run_pipeline end to end with fake clients (no network, synthetic samples/): the story graph is built after the
cold-viewer test and before ranking, rank_promos gets it, story_graph.json is written with the final promo counts,
engagement reads the same graph, and a story-graph failure never fails the promo run."""
import json
import re
from pathlib import Path

from hookcut import engagement, pipeline, rank, story_graph
from hookcut.llm.base import LLMClient, LLMResult
from hookcut.schema import RunResult

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
DEMO = ROOT / "app" / "fixtures" / "demo_run.json"


def run_with_fakes(tmp_path, monkeypatch, graph_fails=False):
    analyses = {a.episode: a.model_dump() for a in RunResult.model_validate_json(DEMO.read_text()).analyses}
    calls, seen = [], {}

    class FakeClient(LLMClient):
        provider = "fake"

        def __init__(self, stage: str):
            super().__init__(stage)
            self.stage = stage

        def complete_json(self, system, user, schema=None, *, validate=None, **kwargs):
            calls.append(self.stage)
            if self.stage == "analyze":
                data = analyses[int(re.match(r"EPISODE (\d+)", user)[1])]
            elif self.stage == "judge":
                data = {"premise": "A bride is betrayed.", "open_question": "Who can she trust?",
                        "hook_lands": True, "standalone": 5, "keep_watching": 5}
            elif self.stage == "writer":
                data = {"lines": [{"id": json.loads(x)["id"], "why": "It cuts on an open question."}
                                  for x in user.splitlines()[1:]]}
            else:
                data = {"notes": ["fake show-level pass"]}
            if validate is not None:
                validate(data)
            return LLMResult(data, json.dumps(data), self.provider, self.model)

    real_graph, real_rank = story_graph.build_story_graph, rank.rank_promos

    def graph_spy(result, client=None, **kw):
        calls.append("graph")
        seen["graph_input_promos"] = len(result.promos)
        if graph_fails:
            raise RuntimeError("graph broke")
        seen["graph"] = real_graph(result, client, **kw)
        return seen["graph"]

    def rank_spy(*args, graph=None, **kw):
        calls.append("rank")
        seen["rank_graph"] = graph
        return real_rank(*args, graph=graph, **kw)

    def engagement_spy(result, graph=None, **kw):
        seen["engagement_graph"] = graph
        raise RuntimeError("engagement skipped in this test")

    monkeypatch.setattr(pipeline, "build_client", lambda cfg, stage, secrets=None, use_cache=True: FakeClient(stage))
    monkeypatch.setattr(pipeline, "load_secrets", lambda env_file=None: {})
    monkeypatch.setattr(story_graph, "build_story_graph", graph_spy)
    monkeypatch.setattr(rank, "rank_promos", rank_spy)
    monkeypatch.setattr(engagement, "load_activity", engagement_spy)
    cfg = pipeline.load_config()
    cfg["promo"]["embeddings"] = "jaccard"
    cfg["concurrency"] = {"analyze": 1, "judge": 1}
    out = tmp_path / "run"
    result = pipeline.run_pipeline(SAMPLES, cfg=cfg, out_dir=out, use_cache=False, progress=lambda m, f: None)
    return result, calls, seen, out


def test_story_graph_is_built_before_ranking_and_shapes_it(tmp_path, monkeypatch):
    result, calls, seen, out = run_with_fakes(tmp_path, monkeypatch)
    order = [c for c in calls if c != "analyze"]
    last_judge = max(i for i, c in enumerate(order) if c == "judge")
    assert order[last_judge + 1:] == ["graph", "genome", "rank", "writer"]
    assert seen["graph_input_promos"] == 0                      # the graph only needs analyses and candidates
    assert isinstance(seen["rank_graph"], story_graph.StoryGraph) and seen["rank_graph"] is seen["graph"]
    assert seen["engagement_graph"] is seen["graph"]            # engagement still runs last, on the same graph
    assert result.promos and all(p.why_it_works == "It cuts on an open question." for p in result.promos)
    saved = story_graph.load_story_graph(out)
    assert saved.model == "fake:genome" and any(c.promos for c in saved.characters)   # counts after ranking
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["layers"]["story_graph"].endswith("story_graph.json") and summary["promos"] == len(result.promos)


def test_a_story_graph_failure_never_fails_the_promo_run(tmp_path, monkeypatch):
    result, calls, seen, out = run_with_fakes(tmp_path, monkeypatch, graph_fails=True)
    assert result.promos and seen["rank_graph"] is None and "genome" not in calls
    assert not (out / "story_graph.json").exists()
    assert any(r.get("stage") == "genome" and "story graph skipped" in str(r) for r in result.run_log)
    plain = rank.rank_promos(result.candidates, result.verdicts, result.episodes, result.analyses,
                             result.config["promo"])
    assert [p.candidate_id for p in plain] == [p.candidate_id for p in result.promos]
