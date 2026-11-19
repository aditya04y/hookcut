import json
from pathlib import Path

from hookcut.genome import build_genome
from hookcut.schema import Character, RunResult


DEMO = Path(__file__).resolve().parents[1] / "app" / "fixtures" / "demo_run.json"


def example():
    return RunResult.model_validate_json(DEMO.read_text())


def test_features_are_deterministic_and_each_edge_has_resolvable_evidence():
    run = example()
    graph = build_genome(run)
    assert graph == json.loads(json.dumps(build_genome(run)))
    assert set(graph["features"]) == {c.id for c in run.candidates}
    for edge in graph["edges"]:
        assert edge["source"] in graph["nodes"] and edge["target"] in graph["nodes"]
        assert edge["basis"] and edge["beat_ids"]
        assert all(f"beat:{beat_id}" in graph["nodes"] for beat_id in edge["beat_ids"])
    features = graph["features"]["C0001"]
    assert {"hook:reveal", "cliff:decision", "stakes:money", "emotion:surprise", "tension:rises"}.issubset(features)
    assert features == sorted(set(features))
    assert all(graph["nodes"][feature]["kind"] == "feature" for feature in features)
    assert set(features) == {edge["target"] for edge in graph["edges"]
                             if edge["source"] == "clip:C0001" and edge["relation"] == "has_feature"}


def test_clip_evidence_uses_segment_times_and_leaves_later_answer_out_of_features():
    run = example()
    run.candidates = [run.candidates[0]]
    cut = run.candidates[0]
    ep = run.episodes[0]
    cliff = ep.beat(cut.cliff_beat_id)
    answer = next(b for b in ep.beats if b.start >= cut.segments[0].end)
    answer.text = "He bursts into tears and sobs."
    next(score for score in run.analyses[0].beats if score.beat_id == cliff.id).resolved_by = answer.id
    graph = build_genome(run)
    assert "payoff:outside_cut" in graph["features"][cut.id]
    assert "emotion:grief" not in graph["features"][cut.id]
    assert any(e["source"] == f"beat:{cliff.id}" and e["relation"] == "answered_by"
               and e["target"] == f"beat:{answer.id}" for e in graph["edges"])
    assert all(e["target"] != f"beat:{answer.id}" for e in graph["edges"] if e["relation"] == "contains")
    assert graph["nodes"][f"beat:{answer.id}"]["start"] == answer.start


def test_explicit_relationship_claims_and_model_roles_keep_source_and_direction():
    run = example()
    old = run.analyses[2]
    old.characters = [
        Character(name="Nora Vale", aliases=["Nora"], role="Miles's older sister"),
        Character(name="Miles", role="Protector"),
    ]
    graph = build_genome(run)
    # Both edges name their origin. The dialogue claim and cast annotation point to different evidence.
    edges = [e for e in graph["edges"] if e["source"] == "character:nora vale"
             and e["target"] == "character:miles"]
    by_relation = {e["relation"]: e for e in edges}
    assert by_relation["calls_sibling"]["beat_ids"] == ["E03.B017"]
    assert by_relation["described_as_sibling_of"]["basis"] == "episode character description"
    assert "E03.B017" in by_relation["described_as_sibling_of"]["beat_ids"]
    assert graph["nodes"]["role:ally"]["dimension"] == "role"


def test_missing_analysis_and_shared_alias_do_not_invent_named_relationships():
    run = example()
    analysis = run.analyses[2]
    analysis.characters = [Character(name="First Stranger", aliases=["Miles"]),
                           Character(name="Second Stranger", aliases=["Miles"])]
    ambiguous = build_genome(run)
    assert not any(e["relation"] == "calls_sibling" for e in ambiguous["edges"])
    run.analyses = []
    graph = build_genome(run)
    assert graph["features"]["C0001"]
    assert not any(n.startswith("role:") for n in graph["nodes"])
    assert any(e["relation"] == "calls_sibling" for e in graph["edges"])


def test_joined_clip_uses_only_selected_parts_of_each_episode():
    run = example()
    candidate = next(c for c in run.candidates if c.kind == "multi")
    run.candidates = [candidate]
    graph = build_genome(run)
    assert "structure:multi" in graph["features"][candidate.id]
    included = {edge["target"] for edge in graph["edges"] if edge["relation"] == "contains"}
    expected = {f"beat:{b.id}" for s in candidate.segments for ep in run.episodes if ep.number == s.episode
                for b in ep.beats if s.start <= b.start < s.end and b.end <= s.end}
    assert included == expected
    assert {e["target"] for e in graph["edges"] if e["relation"] == "drawn_from"} == {
        f"episode:{s.episode}" for s in candidate.segments}


def test_story_graph_is_optional_and_adds_show_context_per_cut():
    from hookcut.story_graph import build_story_graph
    run = example()
    run.analyses[2].characters = [Character(name="Nora Vale", aliases=["Nora"], role="Miles's older sister"),
                                  Character(name="Miles", role="Protector")]
    plain = build_genome(run)
    assert build_genome(run, None) == plain                      # existing callers see no change
    story = build_story_graph(run)
    graph = build_genome(run, story)
    assert graph == build_genome(run, story.model_dump())        # a loaded JSON dict works too
    for cid, features in plain["features"].items():
        assert set(features) <= set(graph["features"][cid])     # context only adds features
    added = {f for cid in plain["features"] for f in set(graph["features"][cid]) - set(plain["features"][cid])}
    assert {"bond:family", "pull:high"} <= added and any(f.startswith("loop:opens_") for f in added)
    assert all(graph["nodes"][f]["kind"] == "feature" and graph["nodes"][f]["label"] for f in added)
    # the bond edge between the two characters cites beats inside the cut
    bond = next(e for e in graph["edges"] if e["relation"] == "story_sister")
    assert (bond["source"], bond["target"]) == ("character:nora-vale", "character:miles")
    for edge in graph["edges"]:
        assert edge["source"] in graph["nodes"] and edge["target"] in graph["nodes"] and edge["beat_ids"]
        assert all(f"beat:{beat_id}" in graph["nodes"] for beat_id in edge["beat_ids"])
    # a question raised and answered inside the same cut is flagged as spent
    cut = next(c for c in run.candidates if c.id == "C0006")
    loop = next(l for l in story.loops if l.opened_beat == "E03.B012")
    loop.answered_ep, loop.answered_beat, loop.status = 3, "E03.B017", "answered"
    assert "loop:spent" in build_genome(run, story)["features"][cut.id]
