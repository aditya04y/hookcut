"""End-to-end contracts for the audience workbench, including holdout leakage."""
import json
import sys
from pathlib import Path

import pytest

from hookcut import audience, studio
from hookcut.schema import RunResult

DEMO = Path(__file__).resolve().parents[1] / "app/fixtures/demo_run.json"


@pytest.fixture
def run():
    return RunResult.model_validate_json(DEMO.read_bytes())


def test_editorial_default_and_partial_saved_run(run):
    run.candidates.pop(0)  # A legacy handoff may have kept a promo but dropped its candidate.
    out = studio.build_studio(run)
    assert out["source"] == "none"
    assert [r["candidate_id"] for r in out["rows"]] == [p.candidate_id for p in run.promos]
    assert all(r["forecast"] is None and r["impressions"] == 0 for r in out["rows"])
    assert out["evaluation"]["model_brier"] is None
    assert out["matrix"] == []
    assert out["workflow"][0]["executor"] == "Hand-authored example"


def test_cohorts_change_the_pick_and_cold_start_has_grounded_support(run):
    new = studio.build_studio(run, simulate=True, cohort="new")
    returning = studio.build_studio(run, simulate=True, cohort="returning")
    assert new["rows"][0]["candidate_id"] == "C0001"
    assert returning["rows"][0]["candidate_id"] == "C0003"
    assert [r["wave"] for r in returning["rows"]] == [1, 1, 2, 2, 3]
    assert sum(cell["impressions"] for cell in returning["matrix"]) == 320
    assert returning["rows"][0]["impressions"] == returning["cohort_stats"]["returning"]["C0003"]["impressions"]
    assert sum(r["impressions"] for r in returning["cohort_stats"]["new"].values()) == 320
    cold = next(r for r in returning["rows"] if r["impressions"] == 0)
    assert 0 < cold["forecast"] < 1
    for neighbor in cold["neighbors"]:
        common = (set(returning["genome"]["features"][neighbor["candidate_id"]]) & set(cold["features"]))
        assert common == set(neighbor["shared"])
        assert neighbor["impressions"] > 0
    assert cold["neighbors"]
    assert returning["experiment"]["cliff_beat_id"] == "E03.B019"


def test_uploaded_evidence_changes_goal_without_crossing_posting_waves(run):
    keys = audience.clip_keys(run)
    clips = {c.id: c for c in run.candidates}
    events = []
    for i in range(40):
        for cid in ["C0001", "C0003"]:
            is_start = cid == "C0003" and i < 20
            seconds = clips[cid].duration if is_start else (4 if cid == "C0001" else 0)
            events.append(audience.Exposure(f"{i}-{cid}", f"v{i}", "test", keys[cid], seconds, is_start))
    csv = audience.events_csv(events)
    hold = studio.build_studio(run, events_csv=csv, objective="hold")
    starts = studio.build_studio(run, events_csv=csv)
    assert starts["source"] == "imported"
    assert hold["rows"][0]["candidate_id"] == "C0001"
    assert starts["rows"][0]["candidate_id"] == "C0003"
    assert all(e["impressions"] == 1 for e in starts["matrix"])
    assert starts["evaluation"]["evaluated_impressions"] == len(events)
    assert starts["rows"][0]["continued"] == 20


def test_upload_roundtrip_and_duplicate_cut_dont_inflate_evidence(run):
    synthetic = audience.events_csv(audience.demo_events(run))
    original = studio.build_studio(run, events_csv=synthetic)
    run.candidates.append(run.candidates[0].model_copy(update={"id": "alias"}))
    out = studio.build_studio(run, events_csv=synthetic)
    assert out["source"] == "simulated"
    assert out["evaluation"] == original["evaluation"]
    assert out["patterns"] == original["patterns"]
    assert out["rows"] == original["rows"]
    assert len(out["matrix"]) == len(original["matrix"])


def test_holdout_excludes_other_cuts_sharing_any_episode(run):
    # All three cuts are from episode 2 or a join between episodes 2 and 3.
    keys = audience.clip_keys(run)
    records = [audience.Exposure(f"v{i}", f"v{i}", "a", keys[cid], 0, False)
               for i, cid in enumerate(("C0003", "C0004", "C0005"))]
    out = studio.build_studio(run, events_csv=audience.events_csv(records))
    assert out["evaluation"] == {"evaluated_impressions": 0, "folds": 0, "model_brier": None, "baseline_brier": None}


def test_holdout_matches_hand_calculated_constant_rate_on_two_disjoint_cuts(run):
    keys = audience.clip_keys(run)
    c1, c2 = run.candidates[0], run.candidates[2]  # E1 and E2
    records = [audience.Exposure("i1", "v1", "all", keys[c1.id], c1.duration, True),
               audience.Exposure("i2", "v2", "all", keys[c2.id], 0, False)]
    out = studio.build_studio(run, events_csv=audience.events_csv(records))
    # With the other episode as the only train row, a held-out success receives
    # a probability of 0.25 and the failure receives 0.75 in the smoothed baseline.
    assert out["evaluation"]["baseline_brier"] == pytest.approx(0.75**2)
    assert out["evaluation"]["folds"] == 2


def test_cli_aggregate_export_and_offline_replay(tmp_path, monkeypatch):
    outfile, csv = tmp_path / "report.json", tmp_path / "demo.csv"
    base = ["studio", "--result", str(DEMO), "--out", str(outfile), "--cohort", "new"]
    monkeypatch.setattr(sys, "argv", base + ["--simulate", "--demo-csv", str(csv)])
    studio.main()
    report = json.loads(outfile.read_text())
    assert all("viewer_id" not in key for key in report)
    assert "matrix" not in report and "demo-v" not in outfile.read_text()
    monkeypatch.setattr(sys, "argv", base + ["--events", str(csv)])
    studio.main()
    assert json.loads(outfile.read_text()) == report
