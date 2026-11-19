"""Connect a saved editorial run to clip-level audience evidence. No model calls."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import audience, engagement
from .genome import build_genome
from .schema import RunResult

OUTCOMES = {"hold": "held_3s", "starts": "continued"}
PRIORS = ("neighbors", "genome")  # similar cuts (default) or engagement's logistic model on story features
COUNT_FIELDS = ("impressions", "held_3s", "completed", "continued")
NEIGHBORS = 5


def _counts(stats):
    return {field: getattr(stats, field) for field in COUNT_FIELDS}


def _strength(feature, objective):
    family = feature.partition(":")[0]
    # Opening cues are particularly relevant to the 3-second stop. For series
    # starts the unresolved ending, relationship and stakes also carry information.
    if family in ("hook", "opening"):
        return 4 if objective == "hold" else 2
    if family in ("cliff", "tension", "stakes", "relationship", "emotion"):
        return 2 if objective == "hold" else 4
    return 1


class AudienceModel:
    """Estimate an outcome per impression using exact cuts and similar story beats."""

    def __init__(self, features, counts, objective, content_model=None):
        self.features = {cid: set(items) for cid, items in features.items()}
        self.counts = counts
        self.outcome = OUTCOMES[objective]
        self.objective = objective
        # Optional callable(train_ids) -> {cut id: rate}. When set, it replaces the neighbor prior.
        self.content_model = content_model
        self._fits = {}

    def _content_prior(self, target, training, include_direct):
        train = frozenset(cid for cid in training if self.counts[cid].impressions
                          and (include_direct or cid != target))
        if train not in self._fits:
            self._fits[train] = self.content_model(train)
        return self._fits[train].get(target)

    def estimate(self, target, training, *, include_direct):
        own = self.counts[target]
        own_n = own.impressions if include_direct else 0
        own_hits = getattr(own, self.outcome) if include_direct else 0
        neighbors = []
        total = positives = 0
        target_features = self.features.get(target, set())
        for cid in training:
            stat = self.counts[cid]
            if cid == target or not stat.impressions:
                continue
            total += stat.impressions
            positives += getattr(stat, self.outcome)
            features = self.features.get(cid, set())
            shared = target_features & features
            if not any(_strength(f, self.objective) > 1 for f in shared):
                continue
            union = target_features | features
            sim = (sum(_strength(f, self.objective) for f in shared) /
                   sum(_strength(f, self.objective) for f in union))
            neighbors.append((sim, cid, shared))
        if not total and not own_n:
            return None, [], None
        prior = self._content_prior(target, training, include_direct) if self.content_model else None
        if prior is not None:  # same smoothing as below, with the story-feature model as the prior
            baseline = (positives + 0.5) / (total + 1)
            return engagement.hybrid_score(own_hits, own_n, prior, total), [], baseline if total else None

        # A half-success/half-failure prior keeps a single exposure from giving
        # 0% or 100%. Support caps keep one high-traffic neighbor from dominating.
        baseline = (positives + 0.5) / (total + 1)
        prior_sum, weight = 8 * baseline, 8.0
        references = []
        for similarity, cid, shared in sorted(neighbors, key=lambda item: (-item[0], item[1]))[:NEIGHBORS]:
            stat = self.counts[cid]
            w = similarity ** 2 * min(40, stat.impressions)
            rate = getattr(stat, self.outcome) / stat.impressions
            prior_sum += w * rate
            weight += w
            references.append({
                "candidate_id": cid, "shared": sorted(shared, key=lambda f: (-_strength(f, self.objective), f)),
                "impressions": stat.impressions, "rate": rate, "similarity": similarity,
            })
        prior_weight = max(1, min(20, total))
        estimate = (own_hits + prior_weight * prior_sum / weight) / (own_n + prior_weight)
        return estimate, references, baseline if total else None


def _evaluation(model, candidates):
    # Test each cut once, grouping by its full set of source episodes. All cuts
    # touching any held-out episode are removed from training, including joins.
    observed = [c for c in candidates if model.counts[c.id].impressions]
    episode_sets = {c.id: frozenset(s.episode for s in c.segments) for c in observed}
    groups = defaultdict(list)
    for cid, episodes in episode_sets.items():
        groups[episodes].append(cid)
    total = folds = 0
    model_loss = baseline_loss = 0.0
    for episodes, test_ids in groups.items():
        train = [cid for cid, used in episode_sets.items() if used.isdisjoint(episodes)]
        if not train:
            continue
        folds += 1
        for cid in test_ids:
            estimate, _, baseline = model.estimate(cid, train, include_direct=False)
            stat = model.counts[cid]
            n, k = stat.impressions, getattr(stat, model.outcome)
            total += n
            model_loss += k * (1 - estimate) ** 2 + (n - k) * estimate ** 2
            baseline_loss += k * (1 - baseline) ** 2 + (n - k) * baseline ** 2
    return {"evaluated_impressions": total, "folds": folds,
            "model_brier": model_loss / total if total else None,
            "baseline_brier": baseline_loss / total if total else None}


def _patterns(features, counts):
    patterns = {}
    for cid, tags in features.items():
        stat = counts[cid]
        if not stat.impressions:
            continue
        for tag in set(tags):
            row = patterns.setdefault(tag, {"feature": tag, "clips": 0, **dict.fromkeys(COUNT_FIELDS, 0)})
            row["clips"] += 1
            for field in COUNT_FIELDS:
                row[field] += getattr(stat, field)
    return sorted(patterns.values(), key=lambda row: (-row["impressions"], row["feature"]))


def _opening_experiment(result, rows):
    # A shared cliff ID alone is insufficient: require the same end time and
    # identical remaining segments. Only the first in-point may change.
    by_id = {c.id: c for c in result.candidates}
    rejected = {v.candidate_id for v in result.verdicts if v.verdict == "fail"}
    def tail(c):
        return (c.segments[0].episode, c.segments[0].end,
                [(s.episode, s.start, s.end) for s in c.segments[1:]])

    for row in rows:
        chosen = by_id.get(row["candidate_id"])
        if chosen is None:
            continue
        for alternate in result.candidates:
            if (alternate.id in rejected or not alternate.segments or not chosen.segments or
                    alternate.cliff_beat_id != chosen.cliff_beat_id or
                    alternate.hook_beat_id == chosen.hook_beat_id or
                    alternate.segments[0].start == chosen.segments[0].start or tail(alternate) != tail(chosen)):
                continue
            return {"candidate_id": chosen.id, "challenger_id": alternate.id,
                    "cliff_beat_id": chosen.cliff_beat_id,
                    "goal": "Compare three-second holds per impression. Check story starts per impression as a secondary outcome.",
                    "allocation": "Randomize eligible viewers 50/50 within one cohort and placement. Keep each viewer in one arm."}
    return None


def _workflow(result, source):
    jobs = []
    for stage, task in [("analyze", "Score story beats"), ("judge", "Review clips as a new viewer"),
                        ("writer", "Explain editorial picks")]:
        models = {f"{r['provider']}:{r['model']}" for r in result.run_log
                  if r.get("stage") == stage and r.get("provider") and r.get("model") and not r.get("error")}
        model_text = ", ".join(sorted(models)) or "Model details unavailable"
        if result.config.get("fixture") == "hand-authored":
            model_text = "Hand-authored example"
        jobs.append({"stage": task, "executor": model_text, "detail": "Loaded from the saved editorial run."})
    jobs.extend([
        {"stage": "Build story graph", "executor": "Local rules", "detail": "Links cuts and story elements to script beats."},
        {"stage": "Aggregate viewer activity", "executor": "Local event processing",
         "detail": {"none": "No audience events supplied.", "simulated": "Uses simulated impressions.",
                    "imported": "Uses uploaded impression records."}[source]},
        {"stage": "Rank and evaluate", "executor": "Local statistical model",
         "detail": "Ranks within each posting wave and tests transfer across held-out episodes."},
    ])
    return jobs


def _genome_model(result, genome, representative, counts, objective):
    """Fit engagement's logistic model on the training cuts' story features; score every cut."""
    features = engagement.cut_features(result, genome)
    canonical = {cid: features[cid] for cid in dict.fromkeys(representative.values())}
    field = OUTCOMES[objective]
    return lambda train: engagement.content_prior(
        canonical, {cid: (counts[cid].impressions, getattr(counts[cid], field)) for cid in train})


def build_studio(result: RunResult, *, events_csv: str | None = None, cohort: str | None = None,
                 objective: str = "starts", simulate: bool = False, prior: str = "neighbors") -> dict:
    """Build recommendations from one run. Simulation is explicit and never mixed with imports."""
    if objective not in OUTCOMES:
        raise ValueError("Choose 'hold' or 'starts' as the objective.")
    if prior not in PRIORS:
        raise ValueError("Choose 'neighbors' or 'genome' as the prior.")
    if simulate and events_csv is not None:
        raise ValueError("Choose one source of audience activity.")
    keys = audience.clip_keys(result)
    if events_csv is not None:
        events = audience.read_events_csv(events_csv, result, keys=keys)
    else:
        events = audience.demo_events(result, keys=keys) if simulate else []
    source = ("simulated" if simulate or (events and events[0].source == "synthetic")
              else "imported" if events_csv is not None else "none")
    cohorts = sorted({e.cohort for e in events})
    if cohort is not None and cohort not in cohorts:
        raise ValueError(f"Unknown cohort {cohort!r}. Available cohorts: {', '.join(cohorts)}.")
    # The same physical edit can have more than one annotated hook and candidate ID.
    # Credit an impression once even when two records describe that edit.
    aliases = defaultdict(list)
    for cid, key in keys.items():
        aliases[key].append(cid)
    activity = audience.aggregate(result, events, cohort=cohort, keys=keys)
    representative = activity.representative
    counts = activity.counts
    genome = build_genome(result)
    features = genome["features"]
    # When annotations disagree for a single physical cut, use only shared tags
    # for statistical comparisons. Full annotations remain visible in the graph.
    common_features = {group[0]: set.intersection(*(set(features[c]) for c in group))
                       for group in aliases.values()}
    content = _genome_model(result, genome, representative, counts, objective) if prior == "genome" else None
    model = AudienceModel(common_features, counts, objective, content_model=content)
    rows = []
    for promo in result.promos:
        cid = promo.candidate_id
        canonical = representative.get(cid)
        forecast, neighbors, _ = (model.estimate(canonical, counts, include_direct=True) if canonical
                                  else (None, [], None))
        s = counts.get(canonical, audience.ClipStats())
        hits = getattr(s, OUTCOMES[objective])
        if s.impressions:
            reason = f"{hits} of {s.impressions} impressions reached the objective. The estimate adjusts for sample size."
        elif content and forecast is not None:
            reason = ("This cut has no impressions in this audience. The estimate comes from a model of story "
                      "features fitted on the cuts that were shown.")
        elif neighbors:
            reason = f"This cut has no impressions in this audience. The estimate uses {len(neighbors)} similar cuts and the audience average."
        elif forecast is not None:
            reason = "Only the audience average is available for this cut."
        else:
            reason = "Awaiting audience activity. The editorial order applies."
        rows.append({"candidate_id": cid, "clip_key": keys.get(cid), "editorial_rank": promo.rank,
                     "wave": promo.post_wave, **_counts(s), "forecast": forecast, "neighbors": neighbors,
                     "features": features.get(cid, []), "reason": reason})
    rows.sort(key=lambda r: (r["wave"], -(r["forecast"] if r["forecast"] is not None else -1), r["editorial_rank"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    matrix = [{"viewer_id": viewer, "clip_key": keys[cid], "candidate_id": cid, **_counts(cell)}
              for viewer, cells in activity.viewer_cells.items()
              for cid, cell in cells.items()]
    cohort_stats = {name: {cid: _counts(by_cut.get(rep, audience.ClipStats())) for cid, rep in representative.items()}
                    for name, by_cut in activity.cohorts_by_cut.items()}
    distinct_cuts = [c for c in result.candidates if c.id == representative[c.id]]
    return {"show": result.show, "source": source, "cohorts": cohorts, "cohort": cohort, "objective": objective,
            "prior": prior, "rows": rows, "matrix": matrix, "genome": genome, "patterns": _patterns(common_features, counts),
            "cohort_stats": cohort_stats, "experiment": _opening_experiment(result, rows),
            "evaluation": _evaluation(model, distinct_cuts),
            "workflow": _workflow(result, source)}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=root / "app/fixtures/demo_run.json")
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--simulate", action="store_true", help="Generate labeled demonstration activity.")
    choices.add_argument("--events", type=Path, help="Import an impression CSV for the same cuts.")
    parser.add_argument("--cohort", help="Select an audience cohort (default: all).")
    parser.add_argument("--objective", choices=OUTCOMES, default="starts")
    parser.add_argument("--prior", choices=PRIORS, default="neighbors",
                        help="Prior for cuts with little data: similar cuts, or a story-feature model.")
    parser.add_argument("--out", type=Path, help="Write recommendations and aggregate evidence as JSON.")
    parser.add_argument("--demo-csv", type=Path, help="Also save the simulated event CSV when using --simulate.")
    args = parser.parse_args()
    if args.demo_csv and not args.simulate:
        parser.error("--demo-csv requires --simulate")
    try:
        result = RunResult.model_validate_json(args.result.read_bytes())
        report = build_studio(result, events_csv=args.events.read_text(encoding="utf-8-sig") if args.events else None,
                              cohort=args.cohort, objective=args.objective, simulate=args.simulate,
                              prior=args.prior)
        if args.demo_csv:
            args.demo_csv.parent.mkdir(parents=True, exist_ok=True)
            args.demo_csv.write_text(audience.events_csv(audience.demo_events(result)), encoding="utf-8")
        report.pop("matrix")  # Individual viewer records stay out of the editor handoff.
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cannot build the audience report: {exc}\n")
    print(f"{result.show} | {report['source']} activity | {args.cohort or 'all cohorts'} | {args.objective}")
    for row in report["rows"]:
        forecast = "awaiting data" if row["forecast"] is None else f"{row['forecast']:.1%}"
        print(f"{row['rank']:2}. {row['candidate_id']}  wave {row['wave']}  editorial #{row['editorial_rank']}"
              f"  {forecast}  ({row['impressions']} impressions)")


if __name__ == "__main__":
    main()
