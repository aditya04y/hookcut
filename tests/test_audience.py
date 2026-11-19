import csv
import io
from dataclasses import replace
from pathlib import Path

import pytest

from hookcut.audience import (
    Exposure, activity_matrix, aggregate, clip_keys, demo_events, events_csv, read_events_csv, summarize,
)
from hookcut.schema import RunResult


@pytest.fixture()
def run():
    fixture = Path(__file__).resolve().parents[1] / "app" / "fixtures" / "demo_run.json"
    return RunResult.model_validate_json(fixture.read_text())


def test_funnel_counts_use_observed_denominators_and_impression_ids(run):
    first, second = run.candidates[:2]
    keys = clip_keys(run)
    start = Exposure("i1", "v1", "new", keys[first.id], 0, False)
    completed = replace(start, impression_id="i2", watch_seconds=first.duration, continued=True)
    held = replace(start, impression_id="i3", viewer_id="v2", cohort="returning", watch_seconds=3)
    events = [start, start, completed, held]
    stats = summarize(run, events)
    assert (stats[first.id].impressions, stats[first.id].held_3s,
            stats[first.id].completed, stats[first.id].continued) == (3, 2, 1, 1)
    assert (stats[first.id].hold_rate, stats[first.id].completion_rate,
            stats[first.id].continuation_rate) == (2/3, 1/3, 1)
    assert stats[second.id].hold_rate is None
    assert summarize(run, events, "returning")[first.id].continuation_rate is None
    matrix = activity_matrix(run, events, "new")
    assert set(matrix) == {"v1"} and set(matrix["v1"]) == {first.id}
    assert matrix["v1"][first.id].impressions == 2
    assert matrix["v1"][first.id].hold_rate == 0.5


def test_simulation_roundtrip_sparse_cohorts_and_unlaunched_shortlist(run):
    events = demo_events(run)
    assert events and demo_events(run) == events and demo_events(run, seed=20) != events
    assert {e.source for e in events} == {"synthetic"}
    assert read_events_csv(events_csv(events + [events[-1]]), run) == events
    assert not read_events_csv(events_csv([]), run)
    stats = summarize(run, events)
    assert any(stats[p.candidate_id].impressions == 0 for p in run.promos)
    new = summarize(run, events, "new")
    returning = summarize(run, events, "returning")
    mysteries = [c.id for c in run.candidates if c.hook_type == "mystery"]
    reveals = [c.id for c in run.candidates if c.hook_type == "reveal"]

    def pooled_hold(stats, ids):
        return sum(stats[i].held_3s for i in ids) / sum(stats[i].impressions for i in ids)

    assert pooled_hold(new, reveals) > pooled_hold(new, mysteries)
    assert pooled_hold(returning, mysteries) > pooled_hold(returning, reveals)
    assert demo_events(run.model_copy(update={"candidates": [], "promos": []})) == []


def test_clip_identity_and_unknown_cut_protection(run):
    keys = clip_keys(run)
    target = run.candidates[0]
    renumbered = target.model_copy(update={"id": "C9876", "hook_score": 0})
    assert clip_keys(run.model_copy(update={"candidates": [renumbered]}))["C9876"] == keys[target.id]
    other_cut = target.model_copy(deep=True)
    other_cut.segments[0].start += 0.1
    assert clip_keys(run.model_copy(update={"candidates": [other_cut]}))[target.id] != keys[target.id]
    revised = run.model_copy(deep=True)
    revised.episodes[0].beat(target.hook_beat_id).text = "A new opening."
    assert clip_keys(revised)[target.id] != keys[target.id]
    assert clip_keys(run.model_copy(update={"show": "Different show"}))[target.id] != keys[target.id]


@pytest.mark.parametrize(("change", "message"), [
    ({"clip_key": "C0001"}, "unknown clip"),
    ({"continued": True, "watch_seconds": 5}, "complete promo"),
    ({"watch_seconds": -1}, "watch_seconds"),
    ({"watch_seconds": 5000}, "watch_seconds"),
    ({"watch_seconds": float("nan")}, "watch_seconds"),
    ({"watch_seconds": float("inf")}, "watch_seconds"),
    ({"viewer_id": ""}, "viewer_id"),
    ({"source": "something else"}, "source"),
])
def test_upload_validation(run, change, message):
    sample = demo_events(run)[0]
    with pytest.raises(ValueError, match=message):
        read_events_csv(events_csv([replace(sample, **change)]), run)


def test_csv_rejects_conflicts_missing_columns_and_bad_values(run):
    sample = demo_events(run)[0]
    with pytest.raises(ValueError, match="conflicting"):
        read_events_csv(events_csv([sample, replace(sample, viewer_id="different")]), run)
    with pytest.raises(ValueError, match="one source"):
        read_events_csv(events_csv([sample, replace(sample, impression_id="i2", source="reported")]), run)
    with pytest.raises(ValueError, match="Missing CSV"):
        read_events_csv("viewer_id,cohort\nv1,new\n", run)
    header, row = list(csv.reader(io.StringIO(events_csv([sample]))))
    row[header.index("continued")] = "maybe"
    with pytest.raises(ValueError, match="row 2"):
        read_events_csv(",".join(header) + "\n" + ",".join(row), run)
    row[header.index("continued")] = "0"
    event, = read_events_csv(",".join(header[:-1]) + "\n" + ",".join(row[:-1]), run)
    assert event.source == "reported"


def test_physical_cut_aliases_share_one_impression_across_cohorts(run, monkeypatch):
    first, second = run.candidates[:2]
    alias = first.model_copy(update={"id": "renamed", "hook_score": 1})
    run = run.model_copy(update={"candidates": [first, alias, second]})
    keys = clip_keys(run)
    skip = Exposure("skip", "v1", "new", keys[first.id], 0, False)
    click = replace(skip, impression_id="click", viewer_id="v2", cohort="returning",
                    watch_seconds=first.duration, continued=True)
    events = [skip, click, click]
    assert summarize(run, events)[alias.id].impressions == 2
    assert activity_matrix(run, events)["v1"][alias.id].impressions == 1

    def fail_if_rehashed(result):
        pytest.fail("provided clip keys were recomputed")
    monkeypatch.setattr("hookcut.audience.clip_keys", fail_if_rehashed)
    activity = aggregate(run, iter(events), "returning", keys=keys)
    assert activity.representative == {first.id: first.id, alias.id: first.id, second.id: second.id}
    assert set(activity.counts) == {first.id, second.id}
    assert activity.counts[first.id].impressions == activity.counts[first.id].continued == 1
    assert activity.counts[second.id].impressions == 0
    assert activity.cohorts_by_cut["new"][first.id].continued == 0
    assert activity.cohorts_by_cut["returning"][first.id].continued == 1
    assert set(activity.viewer_cells) == {"v2"} and set(activity.viewer_cells["v2"]) == {first.id}
