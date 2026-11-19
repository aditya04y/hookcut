"""Engagement layer: both activity grains, the planted simulator, the model, its checks and decisions."""
import json
import random
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hookcut import audience, engagement, studio
from hookcut.engagement import Session
from hookcut.schema import RunResult

DEMO = Path(__file__).resolve().parents[1] / "app/fixtures/demo_run.json"
HOOKS = ["violence", "reveal", "shocking_line", "humiliation", "threat", "romance", "mystery"]


def demo() -> RunResult:
    return RunResult.model_validate_json(DEMO.read_bytes())


def season(n=30, seed=3) -> RunResult:
    """A longer synthetic season: the demo's four episodes repeated with random analysis scores."""
    base, rng = demo(), random.Random(seed)

    def rename(beat_id, m):
        return f"E{m:02d}{beat_id[3:]}" if beat_id else beat_id

    run = base.model_copy(deep=True)
    run.episodes, run.analyses, run.episode_report = [], [], []
    candidates = list(base.candidates)  # episodes 1-4 keep the shortlisted promos' cuts
    for m in range(1, n + 1):
        src = (m - 1) % 4
        ep = base.episodes[src].model_copy(deep=True)
        ep.number = m
        for b in ep.beats:
            b.id, b.episode = rename(b.id, m), m
        a = base.analyses[src].model_copy(deep=True)
        a.episode = m
        for s in a.beats:
            s.beat_id = rename(s.beat_id, m)
            s.hook, s.cliff, s.spoiler = rng.randint(1, 5), rng.randint(1, 5), rng.randint(0, 3)
            s.hook_type = rng.choice(HOOKS)
            s.open_question, s.resolved_by = ("What happens next?" if rng.random() < 0.6 else None), None
        for edge in (a.opening_hook, a.ending_cliffhanger):
            edge.beat_id = rename(edge.beat_id, m)
        a.opening_hook.score, a.ending_cliffhanger.score = rng.randint(1, 5), rng.randint(2, 5)
        a.pickup.status = "not_applicable" if m == 1 else rng.choice(["immediate"] * 3 + ["delayed", "unrelated"])
        a.pickup.first_new_beat_id = rename(a.pickup.first_new_beat_id, m)
        run.episodes.append(ep)
        run.analyses.append(a)
        if m > 4:
            for c in base.candidates:
                if c.kind == "single" and c.segments[0].episode == src + 1:
                    cut = c.model_copy(deep=True, update={"id": f"C{len(candidates) + 1:04d}"})
                    for seg in cut.segments:
                        seg.episode, seg.first_beat_id, seg.last_beat_id = (
                            m, rename(seg.first_beat_id, m), rename(seg.last_beat_id, m))
                    cut.hook_beat_id, cut.cliff_beat_id = rename(cut.hook_beat_id, m), rename(cut.cliff_beat_id, m)
                    # Vary per-cut fields so they are not tied to the six template cuts' wording.
                    cut.hook_score, cut.cliff_score = float(rng.randint(3, 5)), float(rng.randint(3, 5))
                    cut.hook_type, cut.spoiler = rng.choice(HOOKS), rng.randint(0, 3)
                    cut.needs_context = ["Who is she?"] if rng.random() < 0.3 else []
                    cut.duration = round(min(60.0, max(20.0, cut.duration + rng.uniform(-10, 10))), 2)
                    candidates.append(cut)
    run.candidates = candidates
    return run


@pytest.fixture(scope="module")
def season_report():
    run = season()
    report = engagement.build_engagement(run, simulate_activity=True, viewers=3000, paywall=12, bootstrap=40)
    return run, report


# --------------------------------------------------------------------------------------------- schema
def test_session_csv_roundtrip_dedupes_retries_and_keeps_provenance():
    run = demo()
    rows = [Session("v1", "new", 1, True, 1.0, True, True, "imp-1"),
            Session("v1", "new", 2, True, 0.4, False, False),
            Session("v1", "new", 3, False, 0.0, False, False)]
    text = engagement.sessions_csv(rows + rows[:1])
    assert engagement.read_sessions_csv(text, run) == rows
    header = text.splitlines()[0].split(",")
    assert header == list(engagement.SESSION_COLUMNS)
    minimal = "viewer_id,cohort,episode,started,watch_fraction,completed,continued_next\nv9,new,2,1,0.5,0,0\n"
    assert engagement.read_sessions_csv(minimal, run)[0].source == "reported"


@pytest.mark.parametrize(("change", "message"), [
    ({"episode": 99}, "unknown episode"),
    ({"watch_fraction": 1.5}, "watch_fraction"),
    ({"watch_fraction": float("nan")}, "watch_fraction"),
    ({"started": False}, "not started"),
    ({"watch_fraction": 0.5}, "completed session"),
    ({"episode": 4}, "last episode"),
    ({"viewer_id": ""}, "viewer_id"),
    ({"source": "other"}, "source"),
])
def test_session_validation(change, message):
    good = Session("v1", "new", 1, True, 1.0, True, True)
    with pytest.raises(ValueError, match=message):
        engagement.read_sessions_csv(engagement.sessions_csv([replace(good, **change)]), demo())


def test_session_conflicts_and_missing_columns():
    run, good = demo(), Session("v1", "new", 1, True, 1.0, True, True)
    with pytest.raises(ValueError, match="conflicting"):
        engagement.check_sessions([good, replace(good, continued_next=False)], run)
    with pytest.raises(ValueError, match="one source"):
        engagement.check_sessions([good, replace(good, episode=2, source="synthetic")], run)
    with pytest.raises(ValueError, match="Missing CSV columns"):
        engagement.read_sessions_csv("viewer_id,episode\nv1,1\n", run)


def test_episode_matrix_keeps_observed_skip_distinct_from_never_offered():
    run = demo()
    rows = [Session("v1", "new", 1, True, 1.0, True, False), Session("v1", "new", 2, False, 0.0, False, False),
            Session("v2", "returning", 3, True, 0.3, False, False)]
    matrix = engagement.episode_matrix(run, rows)
    assert set(matrix["v1"]) == {1, 2} and matrix["v1"][2].started is False   # offered, not played
    assert 3 not in matrix["v1"] and 1 not in matrix["v2"]                    # never offered
    assert [engagement.session_state(matrix["v1"][e]) for e in (1, 2)] == [2, 0]
    assert set(engagement.episode_matrix(run, rows, cohort="returning")) == {"v2"}


# ------------------------------------------------------------------------------------------- simulator
def test_simulation_is_seeded_labelled_and_internally_consistent():
    run = demo()
    planted = engagement.planted_config(viewers=600)
    impressions, sessions = engagement.simulate(run, planted)
    assert (impressions, sessions) == engagement.simulate(run, planted)
    assert engagement.simulate(run, engagement.planted_config(planted, seed=8))[0] != impressions
    assert {e.source for e in impressions} == {s.source for s in sessions} == {"synthetic"}
    # Both files pass the importers, so a round trip is lossless.
    assert audience.read_events_csv(audience.events_csv(impressions), run) == impressions
    assert engagement.read_sessions_csv(engagement.sessions_csv(sessions), run) == sessions
    starts = {e.impression_id: e for e in impressions if e.continued}
    linked = [s for s in sessions if s.exposure]
    assert linked and all(s.exposure in starts and starts[s.exposure].viewer_id == s.viewer_id for s in linked)
    shown = {e.clip_key for e in impressions}
    keys = audience.clip_keys(run)
    assert len(shown) < len(set(keys.values()))                     # cold-start cuts exist
    assert any(keys[p.candidate_id] not in shown for p in run.promos)
    by_viewer = {}
    for s in sessions:
        by_viewer.setdefault(s.viewer_id, []).append(s)
    for rows in by_viewer.values():                                 # one run of consecutive episodes per viewer
        assert [r.episode for r in rows] == list(range(rows[0].episode, rows[0].episode + len(rows)))
        assert all(r.continued_next for r in rows[:-2])
        last = rows[-1]
        assert not last.started or last.episode == 4                # stop -> the next episode is an observed skip


def test_logit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)) * [1.0, 2.0, 0.5]
    truth = np.array([0.8, -0.3, 0.0])
    n = np.full(400, 200.0)
    k = rng.binomial(200, 1 / (1 + np.exp(-(0.2 + X @ truth)))).astype(float)
    fit = engagement.Logit(X, n, k, lam=0.01)
    assert np.allclose(fit.slopes, truth, atol=0.05)
    assert abs(fit.predict(np.zeros(3))[0] - 1 / (1 + np.exp(-0.2))) < 0.03


def test_story_graph_features_join_as_unplanted_extras(monkeypatch):
    fake = types.ModuleType("hookcut.story_graph")
    fake.episode_features = lambda result, graph=None: {
        a.episode: {"ending_cliff": 99.0, "loops_carried": float(a.episode % 3), "kind": "text"} for a in result.analyses}
    monkeypatch.setitem(sys.modules, "hookcut.story_graph", fake)
    feats, source = engagement.episode_features(demo())
    assert source == "EpisodeAnalysis + story_graph"
    assert feats[2]["graph:loops_carried"] == 2.0 and feats[2]["ending_cliff"] != 99.0
    assert "graph:kind" not in feats[2]
    fake.episode_features = lambda result, graph=None: 1 / 0
    assert "story_graph failed" in engagement.episode_features(demo())[1]


# ------------------------------------------------------------------------------- model and validation
def test_planted_effects_recovered_and_holdout_beats_base_rate(season_report):
    run, report = season_report
    assert report["label"] == "SIMULATED" and report["recovery"]["label"] == "SIMULATED"
    for outcome in ("hold", "start", "complete", "continue"):
        r = report["recovery"][outcome]
        assert r["sign_agreement"] >= 0.8, outcome
        assert r["spearman_planted"] > 0.5, outcome
        assert r["null_false_alarms"] <= max(2, 0.25 * r["null_features"]), outcome
    for outcome in ("complete", "continue"):
        h = report["holdout"][outcome]
        assert h["model_brier"] < h["base_rate_brier"] and h["model_auc"] > 0.55, outcome
    effects = {e["feature"]: e for e in report["models"]["continue"]["effects"]}
    pay = effects["paywall_next"]
    # One paywalled episode: a big negative estimate, but its interval reaches zero because a single
    # episode's effect cannot be separated from that episode's own noise.
    assert pay["log_odds"] < -0.8 and pay["support"] == 1 and pay["units"] > 10
    assert effects["ending_cliff"]["lo"] > 0 and effects["ending_cliff"]["support"] == pay["units"]
    assert report["models"]["continue"]["screened"]          # story-graph extras, one at a time


def test_episode_decisions_find_the_paywall_and_name_reasons(season_report):
    run, report = season_report
    episodes = report["episodes"]
    assert episodes["dropoff"][0]["episode"] == 11                 # planted paywall at 12
    assert "Next episode is behind the paywall" in episodes["dropoff"][0]["drivers"]
    assert all(d["where"] in ("inside the episode", "after the ending") for d in episodes["dropoff"])
    curve = episodes["retention"]["all"]
    assert curve[0]["observed"] == 1.0 and all(a["observed"] >= b["observed"] for a, b in zip(curve, curve[1:])
                                               if b["observed"] is not None)
    assert set(episodes["retention"]) == {"all", "new", "returning", "romance-first", "thriller-first"}
    paywall = episodes["paywall"]
    assert all(5 <= c["last_free_episode"] <= 15 for c in paywall["candidates"])
    assert set(paywall["agreement"]) <= set(paywall["season_summary_unlock"])
    groups = {e["group"] for e in episodes["endings"]}
    assert {"ending cliffhanger score", "questions left unresolved", "ending question kind"} <= groups


def test_hybrid_ranking_scores_cold_cuts_and_differs_by_cohort(season_report):
    run, report = season_report
    ranking = report["promo_ranking"]
    for objective in ("start", "hold"):
        rows = ranking[objective]["all"]
        assert [r["wave"] for r in rows] == sorted(r["wave"] for r in rows)
        cold = [r for r in rows if r["cold_start"]]
        assert cold and all(r["score"] == r["prior"] for r in cold)
    orders = {c: [r["candidate_id"] for r in rows] for c, rows in ranking["hold"].items()}
    assert len({tuple(o) for o in orders.values()}) > 1


def test_power_calculation_matches_textbook():
    assert engagement.sample_size(0.5, 0.05) == 1565
    assert engagement.sample_size(0.1, 0.02) > engagement.sample_size(0.1, 0.03)
    design = engagement.ab_design(demo(), 0.5, 0.1)
    assert design["pair"]["cliff_beat_id"] == "E03.B019"
    assert {s["metric"] for s in design["sample_size"]} == {"3-second hold", "story start"}


def test_studio_genome_prior_scores_cold_cuts_and_default_is_unchanged():
    run = demo()
    default = studio.build_studio(run, simulate=True, cohort="returning")
    assert studio.build_studio(run, simulate=True, cohort="returning", prior="neighbors")["rows"] == default["rows"]
    genome = studio.build_studio(run, simulate=True, cohort="returning", prior="genome")
    cold = next(r for r in genome["rows"] if r["impressions"] == 0)
    assert 0 < cold["forecast"] < 1 and "story features" in cold["reason"] and not cold["neighbors"]
    assert genome["evaluation"]["model_brier"] is not None
    assert studio.build_studio(run, prior="genome")["rows"][0]["forecast"] is None   # no data -> editorial order
    with pytest.raises(ValueError, match="prior"):
        studio.build_studio(run, prior="magic")


def test_real_run_if_present():
    path = Path(__file__).resolve().parents[1] / "outputs/run_full/result.json"
    if not path.exists():
        pytest.skip("no saved full run")
    run = RunResult.model_validate_json(path.read_bytes())
    report = engagement.build_engagement(run, bootstrap=20)        # no CSV given -> SIMULATED
    assert report["matrices"]["episode"]["episodes"] == len(run.episodes)
    assert report["matrices"]["promo"]["cuts"] == len(set(audience.clip_keys(run).values()))
    assert all(report["recovery"][o]["sign_agreement"] >= 0.8 for o in engagement.OUTCOMES)


# ------------------------------------------------------------------------------------- outputs and app
def test_cli_writes_labelled_files_and_replays_them_offline(tmp_path, capsys):
    out = tmp_path / "run"
    engagement.main(["--run", str(DEMO), "--simulate", "--viewers", "500", "--bootstrap", "10", "--out", str(out)])
    saved = json.loads((out / "engagement.json").read_text())
    assert saved["label"] == "SIMULATED" and saved["planted"]["label"] == "SIMULATED"
    assert all(saved[s]["label"] == "SIMULATED" for s in ("matrices", "holdout", "promo_ranking", "episodes"))
    promo_csv, episode_csv = (out / saved["samples"][k] for k in ("promo", "episode"))
    assert "simulated" in promo_csv.name and "synthetic" in promo_csv.read_text()
    assert "[SIMULATED]" in capsys.readouterr().out
    # Replaying the saved rows as an upload keeps them SIMULATED (they are marked synthetic).
    engagement.main(["--run", str(DEMO), "--events", str(promo_csv), "--sessions", str(episode_csv),
                     "--bootstrap", "5", "--out", str(tmp_path / "replay")])
    replay = json.loads((tmp_path / "replay/engagement.json").read_text())
    assert replay["label"] == "SIMULATED" and replay["recovery"] is None
    # Reported rows never leave viewer-level samples behind.
    impressions = audience.read_events_csv(promo_csv.read_text(), demo())
    reported = audience.events_csv([replace(e, source="reported") for e in impressions])
    (tmp_path / "reported.csv").write_text(reported)
    engagement.main(["--run", str(DEMO), "--events", str(tmp_path / "reported.csv"), "--bootstrap", "5",
                     "--out", str(tmp_path / "reported")])
    text = (tmp_path / "reported/engagement.json").read_text()
    assert json.loads(text)["label"] == "REPORTED" and "heatmaps" not in json.loads(text)
    assert "sim-v" not in text and not list((tmp_path / "reported").glob("*.csv"))
    with pytest.raises(SystemExit):
        engagement.main(["--run", str(DEMO)])
    with pytest.raises(ValueError, match="No activity"):
        engagement.build_engagement(demo(), simulate_activity=False)
    with pytest.raises(ValueError, match="not both"):
        engagement.build_engagement(demo(), simulate_activity=True, events_csv=reported)


def test_engagement_view_renders_banner_charts_and_cohort_switch():
    from streamlit.testing.v1 import AppTest

    def page():
        from pathlib import Path

        from app.engagement_view import show_engagement
        from hookcut.schema import RunResult
        path = Path.cwd() / "app/fixtures/demo_run.json"
        show_engagement(RunResult.model_validate_json(path.read_bytes()))

    app = AppTest.from_function(page, default_timeout=60).run()
    assert not app.exception
    assert any("SIMULATED ACTIVITY" in w.value for w in app.warning)
    assert len(app.get("arrow_vega_lite_chart") or app.get("vega_lite_chart")) >= 4
    app.selectbox(key="engagement-cohort").select("romance-first").run()
    assert not app.exception
    app.selectbox(key="engagement-paywall").select(3).run()
    assert not app.exception and any("SIMULATED" in w.value for w in app.warning)


def test_audience_studio_offers_the_story_feature_prior():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(DEMO.parents[1] / "workbench.py"), default_timeout=60).run()
    next(b for b in app.button if b.label.startswith("Demo run")).click().run()
    studio_tab = next(t for t in app.tabs if t.label == "Audience studio")
    studio_tab.radio(key=next(r.key for r in studio_tab.radio if "activity-source" in r.key)).set_value(
        "Simulated activity").run()
    studio_tab = next(t for t in app.tabs if t.label == "Audience studio")
    studio_tab.radio(key=next(r.key for r in studio_tab.radio if "activity-prior" in r.key)).set_value("genome").run()
    next(t for t in app.tabs if t.label == "Audience studio").selectbox[1].select("returning").run()
    next(t for t in app.tabs if t.label == "Audience studio").selectbox[2].select("C0006").run()
    assert not app.exception
    assert any("story features" in m.value for m in next(t for t in app.tabs if t.label == "Audience studio").markdown)
