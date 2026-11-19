"""story_graph.py without network: a synthetic five-episode show built in code and a fake LLM client."""
import json
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from hookcut import story_graph as sg
from hookcut.llm.base import LLMClient
from hookcut.runlog import RunLog
from hookcut.schema import (Beat, BeatScore, Candidate, Character, Episode, EpisodeAnalysis, EpisodeEdge, Pickup,
                            Promo, RunResult, Scene, Segment)

ROOT = Path(__file__).resolve().parents[1]

# (scene, kind, speaker, text); every beat lasts 3 s. Invented show, invented lines.
SCRIPT = {
    1: [(1, "camera", None, "A garden wedding at dusk."),
        (1, "dialogue", "LADY MARA", "Where is my groom?"),
        (1, "dialogue", "LADY IN GREEN", "He chose me, sister."),
        (1, "action", None, "Ivy slaps Mara in front of the guests."),
        (1, "dialogue", "FATHER", "You will marry the stranger instead."),
        (2, "dialogue", "HOODED STRANGER", "I will take her."),
        (2, "action", None, "Mara stares at the hooded stranger."),
        (2, "dialogue", "LADY MARA", "Who are you?")],
    2: [(1, "dialogue", "HOODED STRANGER", "Your sister will regret this."),
        (1, "dialogue", "MARA", "Why help me?"),
        (2, "dialogue", "LORD VALE", "The girl has no dowry."),
        (2, "dialogue", "LADY VALE", "Then she has no place here."),
        (2, "action", None, "Ivy smirks at Mara."),
        (2, "dialogue", "IVY", "You chose a man with empty pockets."),
        (2, "dialogue", "MARA", "Empty pockets beat a stolen crown."),
        (2, "dialogue", "GARDENER", "Someone has been in the orchard.")],
    3: [(1, "dialogue", "LADY IN GREEN", "I saw your husband's face."),
        (1, "dialogue", "THEO", "My name is Theo, and I am the heir."),
        (1, "action", None, "Mara drops the lantern."),
        (1, "dialogue", "MARA", "You lied to me."),
        (2, "dialogue", "IVY", "The heir? Impossible."),
        (2, "dialogue", "THEO", "Ivy, you will answer for the wedding.")],
    4: [(1, "dialogue", "OLD PELL", "I found this in the orchard."),
        (1, "dialogue", "MARA", "A ring."),
        (1, "dialogue", "THEO VANCE", "Then she broke in."),
        (2, "dialogue", "FATHER BRAM", "The council will hear this."),
        (2, "dialogue", "IVY", "You cannot prove anything."),
        (2, "dialogue", "MARA", "Watch me.")],
    5: [(1, "dialogue", "MARA", "It ends today."),
        (1, "dialogue", "IVY", "Then it ends with you.")],
}


def make_episode(n: int) -> Episode:
    beats = [Beat(id=f"E{n:02d}.B{i:03d}", episode=n, scene=scene, kind=kind, speaker=speaker, text=text,
                  start=(i - 1) * 3.0, end=i * 3.0) for i, (scene, kind, speaker, text) in enumerate(SCRIPT[n], 1)]
    scenes = [Scene(number=s, start=min(b.start for b in beats if b.scene == s),
                    end=max(b.end for b in beats if b.scene == s)) for s in sorted({b.scene for b in beats})]
    return Episode(number=n, source_file=f"ep{n}.txt", scenes=scenes, beats=beats, duration=beats[-1].end)


def score(n, b, hook=0, cliff=0, kind=None, question=None, resolved=None):
    return BeatScore(beat_id=f"E{n:02d}.B{b:03d}", hook=hook, cliff=cliff, hook_type=kind, open_question=question,
                     resolved_by=f"E{n:02d}.B{resolved:03d}" if resolved else None, reason="synthetic")


def analysis(n, chars, beats, ending, question, pickup, first_new=None):
    return EpisodeAnalysis(
        episode=n, summary=f"Synthetic episode {n}.", characters=[Character(**c) for c in chars], beats=beats,
        opening_hook=EpisodeEdge(beat_id=f"E{n:02d}.B001", score=2),
        ending_cliffhanger=EpisodeEdge(beat_id=f"E{n:02d}.B{ending[0]:03d}", score=ending[1]),
        ending_open_question=question,
        pickup=Pickup(status=pickup, first_new_beat_id=f"E{n:02d}.B{first_new:03d}" if first_new else None,
                      delay_s=3.0 if first_new else None))


def make_run() -> RunResult:
    analyses = [
        analysis(1, [dict(name="Lady Mara", role="Jilted bride"),
                     dict(name="Ivy", aliases=["LADY IN GREEN"], role="Mara's jealous sister"),
                     dict(name="Father", role="Mara's cold father"),
                     dict(name="The Hooded Stranger", role="Mysterious stranger")],
                 [score(1, 3, 5, 2, "reveal", "Will Mara fight back?", resolved=4),
                  score(1, 4, 5, 3, "violence", "Will the guests intervene?"),
                  score(1, 5, 4, 4, "threat"), score(1, 8, 2, 5, "mystery", "Who is he?")],
                 (8, 5), "Who is the hooded stranger?", "not_applicable"),
        analysis(2, [dict(name="Mara", aliases=["Lady Mara"], role="Defiant bride"),
                     dict(name="The Hooded Stranger", role="Mara's new husband"),
                     dict(name="Lord Vale", role="Proud host"), dict(name="Lady Vale", role="Cold hostess"),
                     dict(name="Ivy", role="Spiteful sister")],
                 [score(2, 1, 4, 2, "threat"), score(2, 6, 5, 3, "humiliation", "Will Mara answer?", resolved=7),
                  score(2, 8, 2, 5, "mystery", "Who was in the orchard?")],
                 (8, 5), "Who broke into the orchard?", "immediate", first_new=2),
        analysis(3, [dict(name="Theo", aliases=["Hooded Stranger", "LADY IN GREEN"],
                          role="Mara's husband, the hidden heir"),
                     dict(name="Mara", role="Wife"), dict(name="Ivy", role="Jealous sister")],
                 [score(3, 2, 5, 4, "reveal"), score(3, 4, 4, 3, "shocking_line", "Will Mara forgive him?"),
                  score(3, 6, 4, 5, "threat")],
                 (6, 5), "Will Ivy be punished?", "delayed", first_new=2),
        analysis(4, [dict(name="Old Pell", role="Loyal gardener"), dict(name="Mara", role="Wife"),
                     dict(name="Theo Vance", role="Mara's husband"), dict(name="Father Bram", role="Village priest"),
                     dict(name="Ivy", role="Cornered sister")],
                 [score(4, 1, 4, 3, "mystery", "What did he find?", resolved=2), score(4, 5, 4, 4, "threat"),
                  score(4, 6, 3, 5)],
                 (6, 5), "Can Mara prove Ivy's guilt?", "unrelated"),
        EpisodeAnalysis(episode=5, summary="analysis failed: timeout", beats=[], opening_hook=EpisodeEdge(),
                        ending_cliffhanger=EpisodeEdge(), pickup=Pickup(status="not_applicable")),
    ]
    episodes = [make_episode(n) for n in SCRIPT]
    seg = lambda n, a, b: Segment(episode=n, start=(a - 1) * 3.0, end=b * 3.0, first_beat_id=f"E{n:02d}.B{a:03d}",
                                  last_beat_id=f"E{n:02d}.B{b:03d}")
    candidates = [
        Candidate(id="C0001", kind="single", segments=[seg(1, 2, 6)], duration=15, hook_beat_id="E01.B003",
                  cliff_beat_id="E01.B005", hook_score=5, cliff_score=4, hook_type="reveal"),
        Candidate(id="C0002", kind="single", segments=[seg(3, 1, 4)], duration=12, hook_beat_id="E03.B002",
                  cliff_beat_id="E03.B004", hook_score=5, cliff_score=3, hook_type="reveal",
                  open_question="Will Mara forgive him?"),
    ]
    promo = Promo(rank=1, candidate_id="C0002", kind="single", segments=candidates[1].segments, duration=12,
                  hook_time="E03 00:03.00", hook_text="-", cliffhanger_time="E03 00:09.00", cliffhanger_text="-",
                  open_question="-", why_it_works="-", post_wave=1)
    return RunResult(show="The Glass Orchard", created_at="2026-11-19", episodes=episodes, analyses=analyses,
                     candidates=candidates, promos=[promo])


CANNED = {
    "characters": [
        {"id": "theo", "same_as": ["Lady Mara"], "name": "Theo Vance", "archetype": "hidden-identity protector"},
        {"id": "old-pell", "same_as": ["gardener"], "archetype": "loyal ally"},
        {"id": "lord-vale", "same_as": ["lady-vale"], "archetype": "complicit parent"},
        {"id": "nobody"},
        {"id": "ivy", "archetype": "evil twin"},
    ],
    "relationships": [
        {"a": "ivy", "b": "mara", "kind": "family", "label": "sister", "first_ep": 1,
         "changes": [{"ep": 2, "to": "rival", "beat_id": "E02.B999"}]},
        {"a": "theo", "b": "mara", "kind": "romance", "label": "stranger", "first_ep": 1,
         "changes": [{"ep": 2, "to": "husband", "beat_id": "B1"}]},
        {"a": "gardener", "b": "mara", "kind": "power", "label": "servant", "first_ep": 2},
    ],
    "reveals": [{"character": "theo", "ep": 3, "beat_id": "E03.B002", "what": "is the heir"}],
    "ending_questions": [
        {"loop": "L-E01-end", "kind": "identity", "answered_ep": 3, "answered_beat": "B4"},
        {"loop": "L-E02-end", "kind": "mystery", "answered_ep": 4, "answered_beat": "E04.B002"},
        {"loop": "L-E03-end", "kind": "justice", "answered_ep": None, "answered_beat": None},
        {"loop": "L-E09-end", "answered_ep": 2},
    ],
    "threads": [{"question": "Will Ivy pay for the stolen wedding?", "kind": "revenge", "opened_ep": 1,
                 "answered_ep": None},
                {"question": "A thread answered before it opens", "kind": "mystery", "opened_ep": 3,
                 "answered_ep": 2}],
    "episodes": [{"ep": 1, "tropes": ["betrayal at the altar", "space opera"],
                  "emotions": ["humiliation", "shock", "anger"]},
                 {"ep": 9, "tropes": ["revenge"]}],
    "notes": ["Every episode ends on a question about Ivy."],
}


class FakeClient(LLMClient):
    provider = "fake"

    def __init__(self, reply=CANNED, fail=False):
        super().__init__("canned")
        self.reply, self.fail, self.seen = reply, fail, []

    def _call(self, system, user, json_mode):
        self.seen.append(user)
        if self.fail:
            raise RuntimeError("provider down")
        return json.dumps(self.reply), 1000, 400


def by_id(graph):
    return {c.id: c for c in graph.characters}


def test_entity_resolution_merges_names_aliases_and_records_the_reveal():
    graph = sg.build_story_graph(make_run())
    chars = by_id(graph)
    # honorifics stripped, full name joins the given name, descriptor alias merged into the named character
    assert chars["mara"].episodes_present == [1, 2, 3, 4, 5]
    assert {"Theo Vance", "The Hooded Stranger"} <= set(chars["theo"].aliases)
    assert "hooded-stranger" not in chars and "theo-vance" not in chars
    # titles keep Lord and Lady Vale apart; a role prefix never becomes a given name
    assert {"lord-vale", "lady-vale", "father", "father-bram"} <= set(chars)
    # E03 wrongly lists LADY IN GREEN as Theo's alias: the gender guard drops it and the label stays Ivy's
    rows = {(r.alias, r.canonical): r for r in graph.aliases}
    assert rows[("LADY IN GREEN", "ivy")].episodes == [1, 3]
    assert not any(r.alias == "LADY IN GREEN" and r.canonical == "theo" for r in graph.aliases)
    assert any("contradicts the character's gender" in w for w in graph.warnings)
    assert "gardener" in chars and not chars["gardener"].role          # a label nobody claimed
    reveal = next(a for a in graph.alerts if a.type == "identity_reveal")
    assert (reveal.ep, reveal.characters, reveal.confidence) == (3, ["theo"], 0.55)
    assert "E03.B002" in reveal.beat_ids                               # first line under the revealed name
    assert chars["ivy"].archetype == "scheming sister" and chars["mara"].archetype == "wronged heroine"


def test_titles_and_role_words_are_time_scoped_roles_not_identity_aliases():
    """The title changes hands: Lady Vale is the duchess in E02, Mara by E04. Read as an alias, 'Duchess of Vale'
    would hand the new duchess's E05 lines to Lady Vale (the reused-company-name error of the bidders resolver)."""
    run = make_run()
    an = {a.episode: a for a in run.analyses}
    next(c for c in an[2].characters if c.name == "Lady Vale").aliases = ["Duchess of Vale"]
    next(c for c in an[4].characters if c.name == "Mara").aliases = ["Duchess", "the new maid"]
    ep = {e.number: e for e in run.episodes}
    ep[2].beats[2].text = "The duchess says the girl has no dowry."             # E02.B003, scene 2 with Lady Vale
    ep[4].beats[3].text = "The council will hear this, Duchess."                # E04.B004, scene 2 with Mara
    ep[5].beats.append(Beat(id="E05.B003", episode=5, scene=1, kind="dialogue", speaker="DUCHESS OF VALE",
                            text="Then it ends with my word.", start=6.0, end=9.0))   # E05's analysis failed
    graph = sg.build_story_graph(run)
    chars = by_id(graph)
    spans = lambda cid: [(r.role, r.first_ep, r.last_ep, r.evidence_beat) for r in chars[cid].roles]
    assert spans("lady-vale") == [("duchess of vale", 2, 2, "E02.B003")]
    assert spans("mara") == [("duchess", 4, 4, "E04.B004")]                  # 'new maid': no evidence beat, left out
    roles = {"Duchess of Vale", "Duchess", "the new maid"}
    assert not any(roles & set(c.aliases) for c in graph.characters)
    assert not any(r.alias in roles for r in graph.aliases)
    label = next(r for r in graph.aliases if r.alias == "DUCHESS OF VALE")
    assert label.canonical not in ("lady-vale", "mara") and 5 not in chars["lady-vale"].episodes_present
    assert sg.find_character(graph, "Duchess") is None
    # a description the show uses as someone's name stays an identity: the hidden-identity reveal survives
    assert "The Hooded Stranger" in chars["theo"].aliases
    assert any(a.type == "identity_reveal" and a.characters == ["theo"] for a in graph.alerts)
    assert any("title/role aliases" in n for n in graph.notes)
    # the rule: a description the show labels someone by is an identity (a disguise), a title never is
    assert not sg._role_alias("cart driver", {"cart driver"}, set()) and sg._role_alias("cart driver", set(), set())
    assert sg._role_alias("duchess of vale", {"duchess of vale"}, set())
    assert not sg._role_alias("hattie the young maid", set(), {"hattie"})


def test_projection_statistics_and_queries_follow_the_bidder_formulas():
    graph = sg.build_story_graph(make_run())
    chars = by_id(graph)
    edge = next(e for e in graph.relationships if {e.a, e.b} == {"ivy", "mara"})
    assert (edge.kind, edge.label, edge.first_ep, edge.basis) == ("family", "sister", 1, "analysis roles")
    assert edge.shared_scenes == 5 and edge.shared_episodes == 5      # E05 has no analysis but has beats
    assert edge.jaccard == round(5 / (chars["ivy"].scenes + chars["mara"].scenes - 5), 3)
    assert edge.conflict_intensity == round(min(100, sum(edge.score_parts.values())), 1)
    assert edge.core == (edge.conflict_intensity >= sg.CORE_CONFLICT_MIN)
    husband = next(e for e in graph.relationships if {e.a, e.b} == {"theo", "mara"})
    assert husband.label == "husband" and not husband.changes          # husband in E2-E4: no turn
    top = sg.top_conflict_pairs(graph, 3)
    assert [t["conflict_intensity"] for t in top] == sorted((t["conflict_intensity"] for t in top), reverse=True)
    profile = sg.character_profile(graph, "LADY MARA")
    assert profile["id"] == "mara" and profile["co_stars"][0]["shared_scenes"] >= 1
    assert profile["clips"] == 2 and profile["promos"] == 1 and 0 < profile["promo_pull"] <= 1
    assert chars["mara"].lines == 8 and chars["mara"].screen_time_s > 0


def test_loops_answered_in_episode_by_pickup_proxy_or_left_open():
    graph = sg.build_story_graph(make_run())
    loops = {l.id: l for l in graph.loops}
    assert (loops["L-E01.B003"].status, loops["L-E01.B003"].answered_beat) == ("answered", "E01.B004")
    assert loops["L-E01.B004"].status == "open" and loops["L-E01.B004"].scope == "scene"
    assert "L-E01.B008" not in loops                                   # the ending question covers that beat
    assert (loops["L-E01-end"].answered_ep, loops["L-E01-end"].basis) == (2, "pickup")
    assert loops["L-E03-end"].status == "open"                         # E04 does not pick it up
    episodes = {e.ep: e for e in graph.episodes}
    assert episodes[4].loops_carried == ["L-E03-end", "L-E04-end"]
    assert 0 < episodes[1].tension <= 1


def test_llm_pass_repairs_bad_ids_merges_and_logs_one_call():
    run, client, log = make_run(), FakeClient(), RunLog()
    graph = sg.build_story_graph(run, client, log=log)
    chars = by_id(graph)
    assert len(client.seen) == 1 and "CHARACTERS" in client.seen[0] and "L-E01-end" in client.seen[0]
    assert [(r["stage"], r["item"], r["provider"]) for r in log.rows] == [("genome", "show", "fake")]
    assert graph.model == "fake:canned" and graph.prompt_version.startswith("story_graph_system@")
    assert "gardener" not in chars and any(r.alias == "GARDENER" and r.canonical == "old-pell" for r in graph.aliases)
    assert {"lord-vale", "lady-vale"} <= set(chars)                    # refused merge
    assert chars["theo"].name == "Theo Vance" and chars["old-pell"].archetype == "loyal ally"
    assert chars["ivy"].archetype == "scheming sister"                 # off-list archetype ignored
    loops = {l.id: l for l in graph.loops}
    assert (loops["L-E01-end"].answered_ep, loops["L-E01-end"].answered_beat) == (3, "E03.B004")   # "B4" repaired
    assert (loops["L-E01-end"].kind, loops["L-E01-end"].basis) == ("identity", "llm")
    assert loops["L-E03-end"].status == "open" and loops["L-E03-end"].kind == "justice"
    threads = [l for l in graph.loops if l.scope == "thread"]
    assert [t.status for t in threads] == ["open", "open"]
    servant = next(e for e in graph.relationships if {e.a, e.b} == {"old-pell", "mara"})
    assert servant.label == "servant"                                  # merged-away id remapped to its survivor
    rival = next(e for e in graph.relationships if {e.a, e.b} == {"ivy", "mara"})
    assert [(c.ep, c.to, c.beat_id) for c in rival.changes] == [(2, "rival", None)]
    spouse = next(e for e in graph.relationships if {e.a, e.b} == {"theo", "mara"})
    assert [(c.ep, c.to, c.beat_id) for c in spouse.changes] == [(2, "husband", "E02.B001")]
    reveal = next(a for a in graph.alerts if a.type == "identity_reveal")
    assert reveal.confidence == 0.85 and reveal.evidence["what"] == "is the heir"   # confirmed by the LLM
    reversals = [a for a in graph.alerts if a.type == "relationship_reversal"]
    assert {a.ep for a in reversals} == {2} and all(a.severity == "high" for a in reversals)
    assert graph.episodes[0].tropes == ["betrayal at the altar"]
    text = " ".join(graph.warnings)
    for expected in ("E02.B999", "repaired beat id 'B4' -> E03.B004", "episode 9 does not exist",
                     "kept lord-vale and lady-vale apart", "unknown id 'nobody'", "L-E09-end", "space opera",
                     "alias strings instead of ids", "evil twin", "not after E03"):
        assert expected in text, expected


def test_failed_call_keeps_the_deterministic_layer():
    run, log = make_run(), RunLog()
    plain = sg.build_story_graph(run)
    graph = sg.build_story_graph(run, FakeClient(fail=True), log=log)
    assert graph.model == "deterministic"
    assert graph.characters == plain.characters and graph.loops == plain.loops
    assert "failed" in graph.warnings[-1] and "error" in log.rows[0]


def test_long_show_goes_in_batches_with_earlier_open_questions():
    client, log = FakeClient(), RunLog()
    sg.build_story_graph(make_run(), client, log=log, max_prompt_tokens=1, batch_episodes=2)
    assert [r["item"] for r in log.rows] == ["show 1/3", "show 2/3", "show 3/3"]
    assert re.search(r"this message: E05-E05", client.seen[2])
    assert "EARLIER OPEN QUESTIONS" in client.seen[2] and "L-E03-end" in client.seen[2]


def test_episode_features_work_without_a_graph_and_get_richer_with_one():
    run = make_run()
    plain = sg.episode_features(run)
    assert sorted(plain) == [1, 2, 3, 4, 5]
    for features in plain.values():
        assert all(isinstance(v, float) for v in features.values())
        assert all(sg.feature_doc(name) for name in features), [n for n in features if not sg.feature_doc(n)]
    assert plain[5]["analysis_failed"] == 1 and plain[1]["analysis_failed"] == 0
    assert plain[1]["pickup_not_applicable"] == 1 and plain[2]["pickup_immediate"] == 1
    assert plain[1]["hook_type_reveal"] == 1 and plain[1]["ending_cliff"] == 5
    assert plain[2]["loops_closed"] >= 1 and plain[2]["payoffs_old"] == 0
    assert plain[3]["identity_reveals"] == 1 and plain[1]["episode_index_frac"] == 0
    rich = sg.episode_features(run, sg.build_story_graph(run, FakeClient()))
    assert rich[2]["relationship_changes"] == 2 and rich[2]["relationship_reversals"] == 2
    assert rich[3]["payoffs_old"] == 1                                 # E01's question answered two episodes later
    assert rich[3]["oldest_open_loop_age"] == 2                        # the open revenge thread from E01
    assert rich[1]["trope_betrayal_at_the_altar"] == 1 and rich[1]["emotion_humiliation"] == 1
    assert rich[1]["max_conflict_intensity"] > 0


def test_round_trip_and_cli(tmp_path):
    run = make_run()
    (tmp_path / "result.json").write_text(run.model_dump_json())
    sg.main(["--run", str(tmp_path), "--no-llm"])
    graph = sg.load_story_graph(tmp_path)
    assert graph == sg.load_story_graph(tmp_path / "story_graph.json")
    assert graph.model_dump() == json.loads(sg.build_story_graph(run).model_dump_json())


def _render(root, result_path, graph_path):
    import sys
    sys.path.insert(0, root)
    from app.genome_view import show_genome
    from hookcut.schema import RunResult
    from hookcut.story_graph import load_story_graph
    result = RunResult.model_validate_json(open(result_path).read())
    show_genome(result, load_story_graph(graph_path) if graph_path else None)


def test_genome_view_renders_with_and_without_a_saved_graph(tmp_path):
    run = make_run()
    (tmp_path / "result.json").write_text(run.model_dump_json())
    sg.write_story_graph(sg.build_story_graph(run, FakeClient()), tmp_path)
    for graph_path in (str(tmp_path / "story_graph.json"), ""):
        app = AppTest.from_function(_render, args=(str(ROOT), str(tmp_path / "result.json"), graph_path)).run()
        assert not app.exception
        assert app.get("graphviz_chart")
        assert len(app.dataframe) >= 3
    dot = __import__("app.genome_view", fromlist=["network_dot"]).network_dot(sg.build_story_graph(run))
    assert dot.startswith("graph story {")
    assert '"ivy" -- "mara"' in dot or '"mara" -- "ivy"' in dot
