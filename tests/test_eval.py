"""Evaluation harness tests. No network: judges are fakes, data is synthetic (eval/synthetic.py)."""
from __future__ import annotations

import csv
import json
import math

import pytest

from eval import HOLDOUT_EPISODES, MAX_JUDGE_CALLS, cost_model, labeling, run_eval, score_labels
from eval.baselines import evenly_spaced, last_window_baseline, random_window_baseline, snap_window
from eval.common import STORY, as_promo, candidate_cut_beats, episode_map, family
from eval.metrics import (agreement, bootstrap_ci, bootstrap_diff_ci, cohen_kappa, interval_iou, rankdata,
                          spearman, summarize)
from eval.synthetic import synthetic_episode, synthetic_run, write_synthetic_run
from hookcut.llm.base import LLMClient, LLMResult
from hookcut.schema import Candidate, ColdViewerVerdict


# ------------------------------------------------------------------ fakes
class FakeClient(LLMClient):
    provider = "fake"

    def __init__(self, model="fake-judge", cached_ids=()):
        super().__init__(model)
        self.calls, self.cached_ids = 0, set(cached_ids)

    def complete_json(self, system, user, schema=None, *, validate=None, retries=2, cache_tag=""):
        cached = user in self.cached_ids
        if not cached:
            self.calls += 1
        return LLMResult(data={"id": user}, text="{}", provider="fake", model=self.model, input_tokens=100,
                         output_tokens=20, cached=cached)


def fake_judge(cands, episodes, client, *, top_k, log, concurrency=2, progress=None):
    """Same contract as hookcut.judge.judge_candidates. Tool clips pass, last-window borderline, random fail."""
    out = []
    for c in sorted(cands, key=lambda c: -c.prelim_score)[:top_k]:
        try:
            res = client.complete_json("sys", c.id)
        except Exception as e:
            log.record("judge", c.id, error=str(e))
            continue
        log.record("judge", c.id, res)
        sa, kw, hook = (5, 5, True) if c.id.startswith("C") else (4, 3, True) if c.id.startswith("LW") else (2, 1, False)
        v = "pass" if hook and sa >= 4 and kw >= 4 else "fail" if sa <= 2 or kw <= 2 else "borderline"
        out.append(ColdViewerVerdict(candidate_id=c.id, premise=f"premise for {c.id}", conflict="x",
                                     open_question="q?", confusions=[], hook_lands=hook, standalone=sa,
                                     keep_watching=kw, verdict=v, model=client.spec))
    return out


@pytest.fixture
def run():
    return synthetic_run()


# ------------------------------------------------------------------ baselines
def _check_clean(c: Candidate, eps):
    Candidate.model_validate(c.model_dump())
    seg = c.segments[0]
    ep = eps[seg.episode]
    assert 20 <= c.duration <= 60
    assert candidate_cut_beats(c, eps, STORY) == []                     # no line / action cut in half
    story = [b for b in ep.beats if b.kind != "camera" and seg.start - 0.05 <= b.start and b.end <= seg.end + 0.05]
    assert c.hook_beat_id == story[0].id and c.cliff_beat_id == story[-1].id
    assert ep.beat(c.hook_beat_id).start - seg.start <= 3.0 + 0.05      # hook plays in the first 3 s
    assert ep.beat(seg.first_beat_id).start >= seg.start - 0.05
    assert ep.beat(seg.last_beat_id).end <= seg.end + 0.05
    assert c.prelim_score > 0


def test_last_window_is_clean_and_ends_at_episode_end(run):
    eps = episode_map(run.episodes)
    cands = last_window_baseline(run.episodes, seconds=45)
    assert len(cands) == len(run.episodes)
    assert len({c.id for c in cands}) == len(cands)
    for c in cands:
        _check_clean(c, eps)
        seg = c.segments[0]
        assert seg.end == pytest.approx(eps[seg.episode].duration)
        assert abs(c.duration - 45) <= 5


def test_random_window_is_clean_seeded_and_counted(run):
    eps = episode_map(run.episodes)
    a = random_window_baseline(run.episodes, seed=7, per_episode=2)
    b = random_window_baseline(run.episodes, seed=7, per_episode=2)
    c = random_window_baseline(run.episodes, seed=8, per_episode=2)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    assert [x.segments[0].start for x in a] != [x.segments[0].start for x in c]
    assert len(a) == 2 * len(run.episodes)
    for x in a:
        _check_clean(x, eps)
        assert 20 <= x.duration <= 60


def test_snap_never_lands_inside_a_long_action():
    ep = synthetic_episode(1, 96.0)
    action = next(b for b in ep.beats if b.kind == "action")          # 31.0-36.5
    t0, t1 = snap_window(ep, 33.0, 70.0)
    assert not (action.start + 0.05 < t0 < action.end - 0.05)
    assert 20 <= t1 - t0 <= 60


def test_evenly_spaced():
    picked = evenly_spaced(list(range(1, 51)), 20)
    assert len(picked) == len(set(picked)) == 20 and picked[0] == 1 and picked[-1] == 50
    assert evenly_spaced([1, 2, 3], 10) == [1, 2, 3]


def test_baseline_as_promo_passes_the_editors_validator(run):
    validate = pytest.importorskip("hookcut.validate").validate_promos
    eps = episode_map(run.episodes)
    for c in last_window_baseline(run.episodes) + random_window_baseline(run.episodes):
        checks = {i["check"] for i in validate([as_promo(c, eps)], run.episodes, run.config["promo"])}
        assert not checks - {"cliff_at_end"}, (c.id, checks)   # a naive window may run past its last line


# ------------------------------------------------------------------ metrics
def test_bootstrap_ci_and_summarize():
    assert bootstrap_ci([3, 3, 3]) == (3.0, 3.0)
    lo, hi = bootstrap_ci([0, 1] * 10)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert bootstrap_ci([]) is None
    s = summarize([1, 0, None, 1])
    assert s["n"] == 3 and s["mean"] == pytest.approx(2 / 3)
    assert bootstrap_diff_ci([1] * 5, [0] * 5) == (1.0, 1.0)


def test_cohen_kappa_toy():
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None                   # undefined: no variance


def test_spearman_and_ranks():
    assert list(rankdata([10, 20, 20, 30])) == [1, 2.5, 2.5, 4]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(4.5 / math.sqrt(4.5 * 5))
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_interval_iou():
    assert interval_iou([(0, 10)], [(5, 15)]) == pytest.approx(1 / 3)
    assert interval_iou([(0, 5), (10, 15)], [(0, 15)]) == pytest.approx(10 / 15)
    assert interval_iou([(0, 5)], [(6, 9)]) == 0.0
    assert interval_iou([(2, 8)], [(2, 8)]) == pytest.approx(1.0)


def test_agreement_toy():
    mk = lambda v, kw, sa=4: {"verdict": v, "keep_watching": kw, "standalone": sa}  # noqa: E731
    a = [mk("pass", 5), mk("pass", 4), mk("fail", 1), mk("borderline", 3)]
    b = [mk("pass", 5), mk("borderline", 3), mk("fail", 2), mk("borderline", 3, 2)]
    r = agreement(a, b + [None][:0])
    assert r["n"] == 4 and r["verdict_agreement"] == pytest.approx(0.75)
    assert r["kappa_pass"] == pytest.approx(cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]))
    assert r["mean_abs_diff_standalone"] == pytest.approx(0.5)
    assert agreement([a[0], None], [b[0], b[1]])["n"] == 1


# ------------------------------------------------------------------ budget guard
def test_budget_guard_trips():
    run_eval.check_budget(MAX_JUDGE_CALLS)
    with pytest.raises(run_eval.BudgetExceeded):
        run_eval.check_budget(MAX_JUDGE_CALLS + 1)
    with pytest.raises(run_eval.BudgetExceeded):
        run_eval.check_budget(60, cap=50)                               # a lower cap is honoured
    with pytest.raises(run_eval.BudgetExceeded):
        run_eval.check_budget(80, cap=1000)                             # the 70 cap cannot be raised


def test_budget_client_counts_live_calls_only():
    inner = FakeClient(cached_ids={"hit"})
    bc = run_eval.BudgetClient(inner, cap=2)
    assert bc.spec == "fake:fake-judge"
    bc.complete_json("s", "a")
    bc.complete_json("s", "hit")                                        # cache hit: free
    bc.complete_json("s", "b")
    with pytest.raises(run_eval.BudgetExceeded):
        bc.complete_json("s", "c")
    assert inner.calls == 2


def test_cli_refuses_over_budget_before_any_call(tmp_path, capsys):
    path = write_synthetic_run(tmp_path / "big", n_episodes=12)
    rc = run_eval.main(["--run", str(path), "--judge", "groq:openai/gpt-oss-120b", "--n", "40"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "plan" in captured.out and "REFUSED" in captured.err
    assert run_eval.main(["--run", str(path), "--n", "5", "--dry-run"]) == 0


def test_evaluate_refuses_over_budget(run):
    with pytest.raises(run_eval.BudgetExceeded):
        run_eval.evaluate(run, judge_fn=fake_judge, client=FakeClient(), n=20, client2=FakeClient("b"), n2=20,
                          cap=30, say=lambda m: None)


# ------------------------------------------------------------------ end to end (fake judge)
def test_evaluate_end_to_end(run, tmp_path):
    validate = pytest.importorskip("hookcut.validate").validate_promos
    client = FakeClient()
    rep = run_eval.evaluate(run, judge_fn=fake_judge, client=client, n=6, validate_fn=validate,
                            say=lambda m: None)
    st = rep["stats"]
    assert st["tool"]["n_clips"] == 6 and st["tool"]["n_judged"] == 6
    assert st["tool"]["pass_rate"]["mean"] == 1.0 and st["random"]["pass_rate"]["mean"] == 0.0
    assert st["last_window"]["n_clips"] == len(run.episodes)            # one per episode when n > episodes
    assert st["random"]["n_clips"] == 6
    assert rep["plan"]["total"] == client.calls == rep["budget_counter"]
    assert rep["diffs"]["tool_minus_random.pass"]["diff"] == pytest.approx(1.0)
    # the deliberately bad pick (cuts a line) is caught by both our check and the editor's validator
    bad = next(r for r in rep["clips"] if r["id"] == "C0099")
    assert bad["own_checks"]["cut_beats"] and not bad["own_checks"]["ok"]
    assert "mid_line_cut" in bad["validator_issues"] and bad["valid"] is False
    assert st["tool"]["cuts_a_line"] >= 1
    # agreement vs the run's stored verdicts: same 6 clips
    assert rep["agreement"]["n"] == 6 and rep["agreement"]["second_judge"] == "run's stored verdicts"

    full = {"run": "synthetic", "created_at": "now", "judge": client.spec, "run_models": run.config["models"],
            "warnings": [], **rep}
    run_eval.write_report(full, tmp_path / "res", tmp_path / "private.json")
    public = (tmp_path / "res" / "eval.json").read_text() + (tmp_path / "res" / "summary.md").read_text()
    assert "Synthetic line" not in public and "premise for" not in public     # no script or judge text
    assert "premise for C0001" in (tmp_path / "private.json").read_text()
    assert json.loads((tmp_path / "res" / "eval.json").read_text())["stats"]["tool"]["n_judged"] == 6


def test_evaluate_with_second_judge(run):
    c1, c2 = FakeClient("a"), FakeClient("b")
    rep = run_eval.evaluate(run, judge_fn=fake_judge, client=c1, n=4, client2=c2, n2=3, say=lambda m: None)
    assert c2.calls == 3 and rep["agreement"]["n"] == 3 and rep["agreement"]["verdict_agreement"] == 1.0
    assert rep["plan"]["judge2"] == 3


def test_family_and_judge_choice(run):
    assert family("groq:openai/gpt-oss-120b") == "openai"
    assert family("groq:qwen/qwen3.8-27b") == "qwen"
    assert family("gemini:gemini-3.8-flash") == "google"
    assert family("anthropic:claude-opus-5-5") == "anthropic"
    spec, warns = run_eval.choose_judge(run, None)          # run used gemini (analyze) + qwen (judge)
    assert family(spec) not in {"google", "qwen"} and not warns
    _, warns = run_eval.choose_judge(run, "groq:qwen/qwen3.8-27b")
    assert any("not independent" in w for w in warns)


# ------------------------------------------------------------------ labels
def test_labels_template_has_no_script_text():
    rows = list(csv.DictReader(open(labeling.TEMPLATE)))
    assert list(rows[0]) == ["episode", "best_start", "best_end", "hook_note", "cliff_note", "labeler"]
    assert [int(r["episode"]) for r in rows] == list(HOLDOUT_EPISODES)
    assert all(not any(v for k, v in r.items() if k != "episode") for r in rows)


def test_labeling_packet(run, tmp_path):
    written = labeling.write_packet(run.episodes, tmp_path, holdout=(1, 3, 99))
    assert [p.name for p in written] == ["E01.md", "E03.md"]
    assert "PRIVATE" in written[0].read_text() and "Synthetic line 1 of episode 1." in written[0].read_text()
    labels = tmp_path / "labels.csv"
    labels.write_text("episode,best_start,best_end,hook_note,cliff_note,labeler\n1,00:10.00,00:40.00,,,AY\n")
    labeling.write_packet(run.episodes, tmp_path, holdout=(1,))
    assert "AY" in labels.read_text()                                      # never overwritten


def test_parse_time_and_read_labels(tmp_path):
    assert score_labels.parse_time("01:04.50") == pytest.approx(64.5)
    assert score_labels.parse_time("64.5") == pytest.approx(64.5)
    assert score_labels.parse_time("") is None
    p = tmp_path / "l.csv"
    p.write_text("episode,best_start,best_end,hook_note,cliff_note,labeler\n5,1:00,1:40,,,A\n5,,,,,\n2,0:10,0:40,,,A\n"
                 "12,0:50,0:20,,,A\n")
    lab = score_labels.read_labels(p)
    assert set(lab) == {5, 12} and len(lab[5]) == 1                        # empty and non-hold-out rows skipped
    assert lab[12][0] == {"start": 20.0, "end": 50.0, "labeler": "A"}      # reversed times are swapped


def test_score_labels_hits_and_misses(run):
    eps = episode_map(run.episodes)
    ranked = score_labels.tool_ranking(run)
    top1 = ranked[2][0]
    seg = top1.segments[0]
    labels = {2: [{"start": seg.start, "end": seg.end}],                   # exact match: hit, IoU 1
              3: [{"start": 0.5, "end": 3.0}]}                             # tiny window nobody picked: miss
    rep = score_labels.score(ranked, labels, eps)
    per = {r["episode"]: r for r in rep["per_episode"]}
    assert per[2]["hit@1"] and per[2]["iou_at_1"] == pytest.approx(1.0)
    assert not per[3]["hit@1"] and not per[3]["hit@3"]
    assert rep["aggregate"]["hit@1"]["hits"] == 1 and rep["aggregate"]["hit@1"]["n"] == 2
    # edges rule: hook and cliff within 3 s counts even when IoU is small
    hook, cliff = eps[2].beat(top1.hook_beat_id), eps[2].beat(top1.cliff_beat_id)
    m = score_labels.match(top1, 2, {"start": hook.start + 2.5, "end": cliff.end - 2.5}, eps)
    assert m["edges_match"] and m["hit"]
    allrep = score_labels.score_all(run, labels)
    assert allrep["last_window"]["aggregate"]["n_episodes"] == 2


# ------------------------------------------------------------------ cost
def test_cost_model_scales_and_prices(tmp_path):
    path = write_synthetic_run(tmp_path / "r", n_episodes=4)
    prices = {"openai/gpt-oss-120b": {"input": 0.15, "output": 0.60}}
    r = cost_model.build(path, shows=200, episodes=50, prices=prices)
    show = r["per_show_tokens"]
    assert show["analyze"]["input"] == pytest.approx(4 * 6000 * 50 / 4)   # per episode x 50 (cached rows count)
    assert show["judge"]["calls"] == pytest.approx(40)                     # scaled to judge_top_k finalists
    s = next(x for x in r["scenarios"] if x["label"] == "All Groq gpt-oss-120b (paid)")
    expect = sum(t["input"] * 0.15 + t["output"] * 0.60 for t in show.values()) / 1e6
    assert s["per_show_usd"] == pytest.approx(expect) and s["per_month_usd"] == pytest.approx(expect * 200)
    assert any(x["missing_prices"] for x in r["scenarios"])                # unknown price -> TODO, not 0
    assert "ASSUMPTIONS" in cost_model.cost_md(r)


def test_load_prices_reads_yaml_block(tmp_path):
    md = tmp_path / "t.md"
    md.write_text("# x\n\n```yaml\nprices_usd_per_mtok:\n  m1: {input: 1.0, output: 2.0}\n```\n")
    p = cost_model.load_prices(md)
    assert p["m1"]["input"] == 1.0 and "claude-opus-5-5" in p


def test_cli_end_to_end_with_fakes(tmp_path, monkeypatch, capsys):
    import hookcut.judge
    path = write_synthetic_run(tmp_path / "run", n_episodes=4)
    monkeypatch.setattr(run_eval, "_build", lambda spec, use_cache, env_file: FakeClient(spec.split(":", 1)[-1]))
    monkeypatch.setattr(hookcut.judge, "judge_candidates", fake_judge)
    rc = run_eval.main(["--run", str(path), "--n", "5", "--judge2", "groq:qwen/qwen3.8-27b",
                        "--out", str(tmp_path / "res")])
    assert rc == 0
    rep = json.loads((tmp_path / "res" / "eval.json").read_text())
    assert rep["plan"]["judge2"] == 5 and rep["agreement"]["n"] == 5
    assert (tmp_path / "run" / "eval_verdicts.json").exists() and (tmp_path / "run" / "eval_log.jsonl").exists()
    assert "| tool |" in (tmp_path / "res" / "summary.md").read_text()


def test_mine_failures_flags_synthetic_bad_picks(run):
    from eval import mine_failures
    rows = {r["id"]: r for r in mine_failures.mine(run)}
    assert "cuts_spoken_line" in rows["C0099"]["flags"]
    assert "multi_segment" in rows["C0100"]["flags"]
    md = mine_failures.failures_md(list(rows.values()), "synthetic")
    assert "C0099" in md and "Synthetic line" not in md


def test_independence_uses_models_that_actually_answered(run):
    run.config["models"]["judge"] = "groq:qwen/qwen3.8-27b -> groq:openai/gpt-oss-120b"   # fallback chain
    spec, warns = run_eval.choose_judge(run, "groq:openai/gpt-oss-120b")
    assert not warns                                   # the log shows only qwen answered
    run.run_log.append({"stage": "judge", "item": "C0001", "model": "openai/gpt-oss-120b", "cached": False})
    _, warns = run_eval.choose_judge(run, "groq:openai/gpt-oss-120b")
    assert any("not independent" in w for w in warns)
