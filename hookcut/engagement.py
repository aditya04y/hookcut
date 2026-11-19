"""Viewer activity for promos and episodes, a planted simulator, and an interpretable model. No model calls.

Two grains, one CSV schema each:

* promo grain (viewer x cut): ``audience.Exposure`` rows, one per settled impression:
  impression_id, viewer_id, cohort, clip_key, watch_seconds, continued, source.
  held_3s = watch_seconds >= 3; story_start = continued (opened the show from the end card).
* episode grain (viewer x episode): ``Session`` rows, one per episode offered to a viewer:
  viewer_id, cohort, episode, started, watch_fraction, completed, continued_next, exposure, source.
  started=0 is an observed skip (offered as the next episode, not played); a missing row means never
  offered. continued_next = started episode n+1 within the attribution window. exposure = the
  impression_id whose story start opened this session ("" for sessions that did not come from a promo).

Four outcomes, each an L2 logistic regression on story features with cluster-bootstrap 90% intervals:
hold (3-s hold per impression), start (story start per impression), complete (episode completed per
started session) and continue (next episode started per completed session).

Simulated activity is labelled SIMULATED everywhere. It shows that the pipeline recovers effects
planted on purpose in PLANTED; it says nothing about how Kuku's viewers behave.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import NormalDist, fmean

import numpy as np

from . import audience
from .audience import Exposure
from .genome import build_genome
from .report import RECAP_LIMIT_S, STRONG_BEAT, UNLOCK_EPISODES, episode_report, season_summary
from .schema import RunResult

LABEL = "SIMULATED"
COMPLETE_AT = 0.9       # a completed session played at least this fraction of the episode
PRIOR_STRENGTH = 20     # most equivalent impressions a content prior may add (same cap as studio)
LAMBDA = 1.0            # L2 penalty on standardised slopes (intercept unpenalised)
BOOTSTRAP = 200
FOLDS = 5
THRILLER = {"threat", "violence", "mystery"}
NULL_FAMILIES = ("stakes", "emotion", "relationship")  # genome tags with no planted effect: a false-alarm check
OUTCOMES = {  # name: (grain, label)
    "hold": ("promo", "3-second hold per impression"),
    "start": ("promo", "story start per impression"),
    "complete": ("episode", "episode completed per started session"),
    "continue": ("episode", "next episode started per completed session"),
}
FEATURES = {  # name: (label, unit). Other names are labelled from the genome or story graph.
    "opening_hook": ("Opening hook score", "points"),
    "ending_cliff": ("Ending cliffhanger score", "points"),
    "loops_open": ("Questions left unresolved", "questions"),
    "strong_beats": ("Promo-ready beats", "beats"),
    "tension": ("Average strength of strong beats", "points"),
    "mix_romance": ("Romance share of strong beats", "share"),
    "mix_thriller": ("Threat, violence or mystery share", "share"),
    "pickup_slow": ("Previous cliffhanger picked up late or never", ""),
    "spoiler_max": ("Largest spoiler in the episode", "levels"),
    "position": ("Later in the season", "of the season"),
    "paywall_next": ("Next episode is behind the paywall", ""),
    "hook_score": ("Hook beat score", "points"),
    "cliff_score": ("Cut-point cliffhanger score", "points"),
    "duration_s": ("Promo length", "s"),
    "hook_romance": ("Romance hook", ""),
    "hook_thriller": ("Threat, violence or mystery hook", ""),
    "context_required": ("Needs earlier episodes to follow", ""),
    "structure_multi": ("Joins several moments", ""),
    "tension_rises": ("Suspense rises from hook to cut", ""),
    "payoff_inside": ("Shows the answer to its own question", ""),
    "spoiler": ("Spoiler level", "levels"),
}

# One generative model for both grains. For every stage:
#   log-odds = base[cohort] + sum_f (w[f] + by_cohort[cohort][f]) * (x[f] - show mean of f) + unit noise
# so `base` is the log-odds for an average cut or episode of the show being simulated.
PLANTED = {
    "label": LABEL,
    "seed": 7,
    "viewers": 4000,
    "impressions_per_viewer": 6,    # distinct cuts per viewer, drawn uniformly: assignment ignores all scores
    "cold_start_share": 0.1,        # cuts never shown, so only the content prior can score them
    "cohorts": {  # share of viewers; resume = chance to enter mid-season; organic = starts without a promo
        "new": {"share": 0.35, "resume": 0.0, "organic": 0.05},
        "returning": {"share": 0.25, "resume": 1.0, "organic": 0.5},
        "romance-first": {"share": 0.2, "resume": 0.3, "organic": 0.1},
        "thriller-first": {"share": 0.2, "resume": 0.3, "organic": 0.1},
    },
    "promo": {
        "hold": {"base": {"new": 0.2, "returning": 0.0, "romance-first": 0.2, "thriller-first": 0.2},
                 "w": {"hook_score": 0.45},
                 "by_cohort": {"new": {"hook_romance": -0.3}, "returning": {"hook_thriller": 0.3},
                               "romance-first": {"hook_romance": 1.0, "hook_thriller": -0.4},
                               "thriller-first": {"hook_thriller": 0.8, "hook_romance": -0.5}}},
        "finish": {"base": {"new": 0.8, "returning": 1.0, "romance-first": 0.8, "thriller-first": 0.8},
                   "w": {"duration_s": -0.035, "structure_multi": -0.3},
                   "by_cohort": {"new": {"context_required": -0.7}}},
        "start": {"base": {"new": -1.0, "returning": -0.4, "romance-first": -0.9, "thriller-first": -0.9},
                  "w": {"cliff_score": 0.4, "payoff_inside": -0.7, "tension_rises": 0.3},
                  "by_cohort": {"romance-first": {"hook_romance": 0.4}, "thriller-first": {"hook_thriller": 0.4}}},
        "noise_sd": 0.25,           # per-cut log-odds the features do not explain
    },
    "episode": {
        "complete": {"base": {"new": 2.2, "returning": 2.8, "romance-first": 2.4, "thriller-first": 2.4},
                     "w": {"opening_hook": 0.3, "pickup_slow": -0.8, "position": -0.4},
                     "by_cohort": {"new": {"position": -0.6}, "romance-first": {"mix_romance": 2.0},
                                   "thriller-first": {"mix_thriller": 2.0, "tension": 0.5}}},
        "continue": {"base": {"new": 2.0, "returning": 2.8, "romance-first": 2.4, "thriller-first": 2.4},
                     "w": {"ending_cliff": 0.45, "loops_open": 0.25, "position": -0.5},
                     "by_cohort": {"thriller-first": {"ending_cliff": 0.25}}},
        "skip_ahead": 0.08,         # continued_next after dropping out mid-episode
        "noise_sd": 0.2,            # per-episode log-odds the features do not explain
    },
    "paywall": {"episode": None, "effect": -1.8},  # log-odds added to continuing INTO this episode
}


# ------------------------------------------------------------------------------ episode-grain schema
@dataclass(frozen=True)
class Session:
    viewer_id: str
    cohort: str
    episode: int
    started: bool
    watch_fraction: float
    completed: bool
    continued_next: bool
    exposure: str = ""
    source: str = "reported"


SESSION_COLUMNS = tuple(f.name for f in fields(Session))


def sessions_csv(sessions) -> str:
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(SESSION_COLUMNS)
    for s in sessions:
        writer.writerow([s.viewer_id, s.cohort, s.episode, int(s.started), s.watch_fraction, int(s.completed),
                         int(s.continued_next), s.exposure, s.source])
    return out.getvalue()


def _flag(value: str) -> bool:
    value = value.lower()
    if value not in ("0", "1", "false", "true"):
        raise ValueError("started, completed and continued_next must be 0 or 1")
    return value in ("1", "true")


def read_sessions_csv(text: str, result: RunResult) -> list[Session]:
    """Import viewer x episode rows. Repeated delivery of an identical row counts once."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿"), newline=""), strict=True)
    try:
        columns = reader.fieldnames or []
        missing = set(SESSION_COLUMNS[:-2]) - set(columns)
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate CSV column")
        rows = []
        for row in reader:
            try:
                if None in row:
                    raise ValueError("too many columns")
                c = {name: (value or "").strip() for name, value in row.items()}
                rows.append(Session(c["viewer_id"], c["cohort"], int(c["episode"]), _flag(c["started"]),
                                    float(c["watch_fraction"]), _flag(c["completed"]), _flag(c["continued_next"]),
                                    c.get("exposure", ""), (c.get("source") or "reported").lower()))
            except ValueError as exc:
                raise ValueError(f"CSV row {reader.line_num}: {exc}") from exc
        return check_sessions(rows, result)
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV near row {reader.line_num}: {exc}") from exc


def check_sessions(sessions, result: RunResult) -> list[Session]:
    """Validate rows and keep one per (viewer, episode)."""
    numbers = sorted(ep.number for ep in result.episodes)
    seen: dict[tuple[str, int], Session] = {}
    source = None
    for row, s in enumerate(sessions, start=2):
        message = ""
        if not (s.viewer_id and s.cohort):
            message = "each session needs a viewer_id and cohort"
        elif s.episode not in numbers:
            message = "unknown episode for this show"
        elif not (math.isfinite(s.watch_fraction) and 0 <= s.watch_fraction <= 1):
            message = "watch_fraction must be between 0 and 1"
        elif not s.started and (s.watch_fraction or s.completed or s.continued_next or s.exposure):
            message = "an episode that was not started has no watch time, completion, continuation or exposure"
        elif s.completed and s.watch_fraction < COMPLETE_AT:
            message = f"a completed session needs watch_fraction >= {COMPLETE_AT}"
        elif s.continued_next and s.episode == numbers[-1]:
            message = "the last episode has no next episode"
        elif s.source not in ("synthetic", "reported") or (source and s.source != source):
            message = "use one source per file (synthetic or reported)"
        elif seen.get((s.viewer_id, s.episode), s) != s:
            message = "conflicting rows for one viewer and episode"
        if message:
            raise ValueError(f"Session row {row}: {message}")
        source = s.source
        seen.setdefault((s.viewer_id, s.episode), s)
    return list(seen.values())


def episode_matrix(result: RunResult, sessions, cohort: str | None = None) -> dict[str, dict[int, Session]]:
    """Sparse viewer x episode matrix: started=False is an observed skip, a missing cell was never offered."""
    matrix: dict[str, dict[int, Session]] = {}
    for s in check_sessions(sessions, result):
        if cohort is None or s.cohort == cohort:
            matrix.setdefault(s.viewer_id, {})[s.episode] = s
    return matrix


activity_matrix = audience.activity_matrix  # the promo grain: viewer x cut ClipStats, missing = never shown
EPISODE_STATES = ["offered, not started", "dropped mid-episode", "finished, stopped", "went on to the next"]
PROMO_STATES = ["skipped before 3 s", "held 3 s", "finished the promo", "started the story"]


def session_state(s: Session) -> int:
    return 0 if not s.started else 3 if s.continued_next else 2 if s.completed else 1


def impression_state(stats: audience.ClipStats) -> int:
    return 3 if stats.continued else 2 if stats.completed else 1 if stats.held_3s else 0


# ------------------------------------------------------------------------------------------ features
def _failed(a) -> bool:
    return a.model.lower().startswith("failed") or "analysis failed" in a.summary.lower()


def fallback_episode_features(result: RunResult) -> dict[int, dict[str, float]]:
    """Per-episode features from EpisodeAnalysis alone (episodes whose analysis failed are left out)."""
    numbers = sorted(ep.number for ep in result.episodes)
    analyses = {a.episode: a for a in result.analyses if not _failed(a)}
    span = max(1, len(numbers) - 1)
    out = {}
    for i, n in enumerate(numbers):
        a = analyses.get(n)
        if a is None:
            continue
        strong = [s for s in a.beats if max(s.hook, s.cliff) >= STRONG_BEAT]
        typed = [s.hook_type for s in strong if s.hook_type]
        p = a.pickup
        slow = p.status in ("delayed", "unrelated") or (p.status == "recap_then_advance"
                                                        and (p.delay_s or 0) > RECAP_LIMIT_S)
        out[n] = {
            "opening_hook": float(a.opening_hook.score),
            "ending_cliff": float(a.ending_cliffhanger.score),
            # Strong cut points whose question this episode never answers are carried into the next one.
            "loops_open": float(sum(1 for s in a.beats if s.cliff >= STRONG_BEAT and s.open_question
                                    and not s.resolved_by)),
            "strong_beats": float(len(strong)),
            "tension": fmean(max(s.hook, s.cliff) for s in strong) if strong else 0.0,
            "mix_romance": typed.count("romance") / len(typed) if typed else 0.0,
            "mix_thriller": sum(t in THRILLER for t in typed) / len(typed) if typed else 0.0,
            "pickup_slow": float(slow and i > 0),
            "spoiler_max": float(max((s.spoiler for s in a.beats), default=0)),
            "position": i / span,
        }
    return out


def episode_features(result: RunResult, graph=None) -> tuple[dict[int, dict[str, float]], str]:
    """Fallback features plus any extra numeric ones from hookcut.story_graph, when that module exists.

    Fallback names win on a clash so the planted model always reads the same columns; graph extras get a
    "graph:" prefix and enter the model as unplanted features.
    """
    feats = fallback_episode_features(result)
    try:
        from .story_graph import episode_features as graph_features
    except ImportError:
        return feats, "EpisodeAnalysis (story_graph not available)"
    try:
        extra = graph_features(result, graph)
    except Exception as exc:  # the graph layer is optional; never block engagement on it
        return feats, f"EpisodeAnalysis (story_graph failed: {type(exc).__name__})"
    for n, values in (extra or {}).items():
        row = feats.get(int(n))
        if row is None:
            continue
        for name, value in values.items():
            if name not in row and isinstance(value, (int, float)) and math.isfinite(value):
                row[f"graph:{name}"] = float(value)
    return feats, "EpisodeAnalysis + story_graph"


def cut_features(result: RunResult, genome: dict | None = None) -> dict[str, dict[str, float]]:
    """Per-cut features from the candidate and its genome tags (binary)."""
    genome = genome or build_genome(result)
    tags_by_cut = {cid: set(tags) for cid, tags in genome["features"].items()}
    counts = Counter(t for tags in tags_by_cut.values() for t in tags)
    total = len(result.candidates)
    extra = sorted(t for t, c in counts.items()
                   if t.split(":")[0] in NULL_FAMILIES and 0.1 * total <= c <= 0.9 * total)
    out = {}
    for c in result.candidates:
        tags = tags_by_cut.get(c.id, set())
        row = {"hook_score": c.hook_score, "cliff_score": c.cliff_score, "duration_s": c.duration,
               "hook_romance": float("hook:romance" in tags),
               "hook_thriller": float(any(f"hook:{t}" in tags for t in THRILLER)),
               "context_required": float("context:required" in tags),
               "structure_multi": float("structure:multi" in tags),
               "tension_rises": float("tension:rises" in tags),
               "payoff_inside": float("payoff:inside_cut" in tags), "spoiler": float(c.spoiler)}
        row.update({t: float(t in tags) for t in extra})
        out[c.id] = row
    return out


def _means(rows) -> dict[str, float]:
    rows = list(rows)
    return {k: fmean(r[k] for r in rows) for k in rows[0]} if rows else {}


@dataclass
class _Context:
    keys: dict[str, str]          # candidate id -> clip key
    reps: dict[str, str]          # clip key -> representative candidate id (first one, as audience.aggregate)
    cut_x: dict[str, dict]        # representative id -> features
    ep_x: dict[int, dict]         # every episode -> features (failed analyses get season means)
    missing: list[int]            # episodes without usable analysis (simulated, never fitted)
    numbers: list[int]
    feature_source: str
    labels: dict[str, str]

    @property
    def cut_means(self):
        return _means(self.cut_x.values())

    @property
    def ep_means(self):
        return _means(self.ep_x.values())


def _context(result: RunResult, paywall: int | None, keys=None, graph=None) -> _Context:
    keys = keys or audience.clip_keys(result)
    reps: dict[str, str] = {}
    for c in result.candidates:
        reps.setdefault(keys[c.id], c.id)
    genome = build_genome(result)
    cuts = cut_features(result, genome)
    feats, source = episode_features(result, graph)
    numbers = sorted(ep.number for ep in result.episodes)
    names = list(dict.fromkeys(k for row in feats.values() for k in row))
    season = {k: fmean(r[k] for r in feats.values() if k in r) for k in names}
    ep_x = {}
    for i, n in enumerate(numbers):
        row = {k: feats.get(n, {}).get(k, season[k]) for k in names}
        if "position" in row:
            row["position"] = i / max(1, len(numbers) - 1)
        row["paywall_next"] = float(paywall is not None and i + 1 < len(numbers) and numbers[i + 1] == paywall)
        ep_x[n] = row
    labels = {k: v["label"] for k, v in genome["nodes"].items() if v.get("kind") == "feature"}
    return _Context(keys, reps, {cid: cuts[cid] for cid in reps.values()}, ep_x,
                    sorted(set(numbers) - set(feats)), numbers, source, labels)


# -------------------------------------------------------------------------------------------- simulator
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def planted_config(planted=None, *, seed=None, viewers=None, paywall=None) -> dict:
    """A copy of PLANTED (or `planted`) with CLI overrides; paywall=0 switches the paywall off."""
    p = copy.deepcopy(planted or PLANTED)
    if seed is not None:
        p["seed"] = seed
    if viewers is not None:
        p["viewers"] = viewers
    if paywall is not None:
        p["paywall"]["episode"] = paywall or None
    return p


def _stage(planted, grain, stage, cohort, x, means) -> float:
    cfg = planted[grain][stage]
    weights = Counter(cfg.get("w", {}))
    weights.update(cfg.get("by_cohort", {}).get(cohort, {}))
    return cfg["base"][cohort] + sum(w * (x[f] - means[f]) for f, w in weights.items() if f in x)


def planted_probability(planted, outcome, cohort, x, means, noise=(0.0, 0.0, 0.0)) -> float:
    """The planted chance of an outcome for one cut or episode (noise=0 gives the systematic part)."""
    if outcome in ("hold", "start"):
        hold, finish, start = (_sigmoid(_stage(planted, "promo", s, cohort, x, means) + e)
                               for s, e in zip(("hold", "finish", "start"), noise))
        return float(hold if outcome == "hold" else hold * finish * start)
    pay = planted["paywall"]["effect"] * x.get("paywall_next", 0.0)
    if outcome == "complete":
        return float(_sigmoid(_stage(planted, "episode", "complete", cohort, x, means) + noise[0]))
    if outcome == "continue":
        return float(_sigmoid(_stage(planted, "episode", "continue", cohort, x, means) + noise[1] + pay))
    return float(_sigmoid(_logit(planted["episode"]["skip_ahead"]) + pay))  # "skip": continue without finishing


def simulate(result: RunResult, planted=None, *, keys=None, graph=None) -> tuple[list[Exposure], list[Session]]:
    """Draw promo impressions and episode sessions from the planted model. Every row is source=synthetic.

    Each viewer sees up to `impressions_per_viewer` distinct cuts chosen uniformly at random. The first story
    start opens the show (the session carries that impression_id as `exposure`); viewers with no story start
    may still arrive organically. Entry is episode 1, or a random episode for viewers who resume mid-season.
    """
    P = planted_config(planted)
    ctx = _context(result, P["paywall"]["episode"], keys, graph)
    rng = np.random.default_rng(P["seed"])
    launched = sorted(ctx.reps)
    if len(launched) >= 2:  # hold back a few cuts, including one shortlisted promo, to exercise cold start
        shortlist = sorted({ctx.keys[p.candidate_id] for p in sorted(result.promos, key=lambda p: p.rank)[1:]
                            if p.candidate_id in ctx.keys})
        held = [shortlist[int(rng.integers(len(shortlist)))]] if shortlist else []
        rest = [k for k in launched if k not in held]
        size = min(len(launched) - 1, max(1, round(P["cold_start_share"] * len(launched)))) - len(held)
        held += [rest[i] for i in rng.permutation(len(rest))[:max(0, size)]]
        launched = [k for k in launched if k not in held]
    cohorts = list(P["cohorts"])
    shares = np.array([P["cohorts"][c]["share"] for c in cohorts], float)
    cut_means, ep_means = ctx.cut_means, ctx.ep_means
    cut_noise = {k: rng.normal(0, P["promo"]["noise_sd"], 3) for k in launched}
    ep_noise = {n: rng.normal(0, P["episode"]["noise_sd"], 2) for n in ctx.numbers}
    promo_p = {(c, k): [_sigmoid(_stage(P, "promo", s, c, ctx.cut_x[ctx.reps[k]], cut_means) + e)
                        for s, e in zip(("hold", "finish", "start"), cut_noise[k])]
               for c in cohorts for k in launched}
    ep_p = {(c, n): [planted_probability(P, o, c, ctx.ep_x[n], ep_means, ep_noise[n])
                     for o in ("complete", "continue", "skip")] for c in cohorts for n in ctx.numbers}
    durations = {k: next(c.duration for c in result.candidates if c.id == ctx.reps[k]) for k in launched}
    impressions, sessions = [], []
    for v in range(P["viewers"]):
        cohort = cohorts[rng.choice(len(cohorts), p=shares / shares.sum())]
        viewer, exposure = f"sim-v{v + 1:05d}", ""
        for j in rng.choice(len(launched), size=min(P["impressions_per_viewer"], len(launched)), replace=False):
            key = launched[j]
            duration, (hold, finish, start) = durations[key], promo_p[cohort, key]
            if rng.random() >= hold:
                watched, started = round(rng.uniform(0, min(2.95, duration * 0.8)), 3), False
            elif rng.random() >= finish:
                watched, started = round(rng.uniform(3, max(3, duration - 0.05)), 3), False
            else:
                watched, started = duration, bool(rng.random() < start)
            iid = f"sim-{P['seed']}-{v + 1}-{j}"
            impressions.append(Exposure(iid, viewer, cohort, key, watched, started, "synthetic"))
            exposure = exposure or (iid if started else "")
        if not exposure and rng.random() >= P["cohorts"][cohort]["organic"]:
            continue
        i = int(rng.integers(len(ctx.numbers))) if rng.random() < P["cohorts"][cohort]["resume"] else 0
        while True:
            n, last = ctx.numbers[i], i + 1 == len(ctx.numbers)
            complete, cont, skip = ep_p[cohort, n]
            done = bool(rng.random() < complete)
            go = not last and bool(rng.random() < (cont if done else skip))
            fraction = 1.0 if done else round(rng.uniform(0.02, COMPLETE_AT - 0.01), 3)
            sessions.append(Session(viewer, cohort, n, True, fraction, done, go, exposure, "synthetic"))
            exposure = ""
            if last:
                break
            if not go:  # the player offered the next episode and the viewer did not start it
                sessions.append(Session(viewer, cohort, ctx.numbers[i + 1], False, 0.0, False, False, "", "synthetic"))
                break
            i += 1
    return impressions, sessions


# ------------------------------------------------------------------------------------------------ model
class Logit:
    """L2-penalised binomial logistic regression, fitted by Newton's method on aggregated (x, n, k) rows.

    Columns are standardised with trial-weighted mean and SD (or a given `scaling`, so bootstrap refits
    share one penalty); `slopes` are per natural unit of each feature.
    """

    def __init__(self, X, n, k, lam=LAMBDA, scaling=None, start=None):
        if scaling is None:
            w = n / n.sum()
            mean = w @ X
            scale = np.sqrt(w @ (X - mean) ** 2)
            scaling = mean, np.where(scale > 1e-12, scale, 1.0)
        self.mean, self.scale = scaling
        A = np.column_stack([np.ones(len(X)), (X - self.mean) / self.scale])
        beta = np.zeros(A.shape[1]) if start is None else start.copy()
        if start is None:
            beta[0] = _logit((k.sum() + 0.5) / (n.sum() + 1))
        penalty = np.full(A.shape[1], lam)
        penalty[0] = 0.0
        for _ in range(100):
            p = _sigmoid(A @ beta)
            grad = A.T @ (k - n * p) - penalty * beta
            hess = (A.T * (n * p * (1 - p))) @ A + np.diag(penalty) + 1e-9 * np.eye(len(beta))
            step = np.linalg.solve(hess, grad)
            step *= min(1.0, 4.0 / max(np.abs(step).max(), 1e-12))  # damp the first steps from a far start
            beta += step
            if np.abs(step).max() < 1e-8:
                break
        self.beta = beta

    @property
    def slopes(self):
        return self.beta[1:] / self.scale

    def predict(self, X):
        return _sigmoid(self.beta[0] + ((np.atleast_2d(X) - self.mean) / self.scale) @ self.beta[1:])


@dataclass
class Table:
    """Aggregated binomial rows for one outcome: one row per (cut or episode, cohort)."""
    outcome: str
    names: list[str]
    X: np.ndarray
    n: np.ndarray
    k: np.ndarray
    cohort: np.ndarray
    unit: list            # candidate id or episode number (the bootstrap and holdout cluster)

    def subset(self, mask):
        mask = np.asarray(mask, bool)
        return Table(self.outcome, self.names, self.X[mask], self.n[mask], self.k[mask], self.cohort[mask],
                     [u for u, m in zip(self.unit, mask) if m])


def _table(outcome, rows, names) -> Table:
    """rows: (features dict, n, k, cohort, unit)."""
    rows = [r for r in rows if r[1] > 0]
    X = np.array([[r[0][f] for f in names] for r in rows], float).reshape(len(rows), len(names))
    return Table(outcome, names, X, np.array([r[1] for r in rows], float), np.array([r[2] for r in rows], float),
                 np.array([r[3] for r in rows], dtype=object), [r[4] for r in rows])


def _core(names) -> list[str]:
    """The named features every show has. Other columns (story-graph numbers, genome tags) are screened."""
    return [f for f in names if f in FEATURES]


def _design(table: Table, cohorts: list[str] | None, features: list[str]):
    """Feature matrix for `features`, plus cohort intercepts (first cohort is the baseline) when pooled."""
    X = table.X[:, [table.names.index(f) for f in features]]
    if not cohorts:
        return list(features), X
    dummies = [f"cohort:{c}" for c in cohorts[1:]]
    D = np.array([[float(c == d[7:]) for d in dummies] for c in table.cohort]).reshape(len(table.cohort), len(dummies))
    return list(features) + dummies, np.column_stack([X, D])


def _columns(X, n, names) -> list[int]:
    """Drop constant columns and any column almost collinear (|r| > 0.95) with an earlier one."""
    w = n / n.sum()
    Z = X - w @ X
    sd = np.sqrt(w @ Z ** 2)
    keep = []
    for j in range(len(names)):
        if sd[j] < 1e-9:
            continue
        if any(abs(w @ (Z[:, i] * Z[:, j])) / (sd[i] * sd[j]) > 0.95 for i in keep):
            continue
        keep.append(j)
    return keep


class OutcomeModel:
    """One fit (pooled with cohort intercepts, or one cohort) with a cluster bootstrap over cuts or episodes."""

    def __init__(self, table: Table, cohorts: list[str] | None, features: list[str], *, bootstrap: int, seed: int,
                 lam=LAMBDA):
        self.table, self.cohorts = table, cohorts
        names, X = _design(table, cohorts, features)
        self.cols = _columns(X, table.n, names)
        self.names = [names[j] for j in self.cols]
        self.X = X[:, self.cols]
        self.fit = Logit(self.X, table.n, table.k, lam)
        units = sorted(set(table.unit), key=str)
        index = np.array([units.index(u) for u in table.unit])
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(bootstrap):
            weight = np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units))[index]
            keep = weight > 0
            refit = Logit(self.X[keep], table.n[keep] * weight[keep], table.k[keep] * weight[keep], lam,
                          scaling=(self.fit.mean, self.fit.scale), start=self.fit.beta)
            boots.append(refit.slopes)
        self.boots = np.array(boots).reshape(bootstrap, len(self.names))
        self.units = len(units)

    def vector(self, x: dict, cohort: str | None = None) -> np.ndarray:
        return np.array([float(cohort == f[7:]) if f.startswith("cohort:") else x.get(f, 0.0) for f in self.names])

    def predict(self, x: dict, cohort: str | None = None) -> float:
        return float(self.fit.predict(self.vector(x, cohort))[0])

    def binary(self, j: int) -> bool:
        return set(np.unique(self.X[:, j])) <= {0.0, 1.0}

    def swing(self, j: int) -> float:
        """Binary features move 0 -> 1; continuous ones move 2 SD, which makes the two comparable."""
        return 1.0 if self.binary(j) else 2 * float(self.fit.scale[j])

    def effects(self, labels: dict[str, str], only: str | None = None) -> list[dict]:
        rate = float(self.table.k.sum() / self.table.n.sum())

        def points(b):
            return round(100 * float(_sigmoid(_logit(rate) + b) - rate), 2)

        rows = []
        for j, name in enumerate(self.names):
            if name.startswith("cohort:") or (only and name != only):
                continue
            step = self.swing(j)
            b = float(self.fit.slopes[j]) * step
            lo, hi = (np.percentile(self.boots[:, j] * step, [5, 95]) if len(self.boots) else (b, b))
            label, unit = FEATURES.get(name, (labels.get(name, name.split(":", 1)[-1].replace("_", " ")), ""))
            rows.append({"feature": name, "label": label.capitalize(),
                         "change": "yes vs no" if self.binary(j) else f"+{step:.2g} {unit}".strip(),
                         "log_odds": round(b, 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                         "points": points(b), "points_lo": points(lo), "points_hi": points(hi),
                         "n": int(self.table.n.sum()), "units": self.units,
                         # Cuts or episodes where the feature is non-zero. With one (a single paywalled
                         # episode, say) its effect cannot be told apart from that unit's own noise.
                         "support": len({u for u, v in zip(self.table.unit, self.X[:, j]) if v})})
        return sorted(rows, key=lambda r: -abs(r["log_odds"]))


# ------------------------------------------------------------------------------------------- tables
def _episode_counts(sessions) -> dict[tuple[int, str], list[int]]:
    """(episode, cohort) -> [started, completed, continued after completing, continued without completing]."""
    agg: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for s in sessions:
        if s.started:
            a = agg[s.episode, s.cohort]
            a[0] += 1
            a[1] += s.completed
            a[2] += s.completed and s.continued_next
            a[3] += (not s.completed) and s.continued_next
    return agg


def _tables(ctx: _Context, activity, sessions) -> dict[str, Table]:
    tables = {}
    if activity is not None:
        names = list(next(iter(ctx.cut_x.values())))
        for outcome, field in (("hold", "held_3s"), ("start", "continued")):
            rows = [(ctx.cut_x[cid], s.impressions, getattr(s, field), cohort, cid)
                    for cohort, stats in sorted(activity.cohorts_by_cut.items()) for cid, s in stats.items()]
            if rows:
                tables[outcome] = _table(outcome, rows, names)
    if sessions:
        agg = _episode_counts(sessions)
        fitted = [n for n in ctx.numbers if n not in ctx.missing]
        names = [f for f in next(iter(ctx.ep_x.values())) if f != "paywall_next"]
        rows = [(ctx.ep_x[n], a[0], a[1], c, n) for (n, c), a in sorted(agg.items()) if n in fitted]
        tables["complete"] = _table("complete", rows, names)
        rows = [(ctx.ep_x[n], a[1], a[2], c, n) for (n, c), a in sorted(agg.items())
                if n in fitted and n != ctx.numbers[-1]]
        tables["continue"] = _table("continue", rows, names + ["paywall_next"])
    return {k: t for k, t in tables.items() if len(t.n) >= 2 and t.n.sum() > 0}


def _fit_all(tables, bootstrap, seed):
    """Per outcome: the pooled core model (cohort intercepts), one core model per cohort (every cohort x
    feature interaction), and each extra feature screened by adding it alone to the pooled core model.
    Fifty episodes cannot support fifty regressors at once; one at a time keeps each estimate adjusted
    for the core and the core stable."""
    models = {}
    for outcome, table in tables.items():
        cohorts, core = sorted(set(table.cohort)), _core(table.names)
        pooled = OutcomeModel(table, cohorts, core, bootstrap=bootstrap, seed=seed)
        by_cohort = {}
        for c in cohorts:
            part = table.subset(table.cohort == c)
            if len(set(part.unit)) >= 2:
                by_cohort[c] = OutcomeModel(part, None, core, bootstrap=bootstrap, seed=seed)
        screened = {}
        for extra in (f for f in table.names if f not in core):
            model = OutcomeModel(table, cohorts, core + [extra], bootstrap=max(10, bootstrap // 4), seed=seed)
            if extra in model.names:  # dropped when constant or collinear with the core
                screened[extra] = model
        models[outcome] = (pooled, by_cohort, screened)
    return models


# ------------------------------------------------------------------------------------------ validation
def _ranks(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    for value in np.unique(a):
        ranks[a == value] = ranks[a == value].mean()
    return ranks


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return round(float(np.corrcoef(_ranks(a), _ranks(b))[0, 1]), 3)


def _planted_slopes(model: OutcomeModel, planted, ctx: _Context):
    """Slopes the planted model implies on the same rows: weighted least squares of the noiseless planted
    log-odds on the model's own columns. For a one-stage outcome this returns the planted weights exactly;
    for composites (start, continue across cohorts) it is the best logistic summary of the truth."""
    table = model.table
    means = ctx.cut_means if OUTCOMES[table.outcome][0] == "promo" else ctx.ep_means
    names = table.names
    y = np.array([_logit(planted_probability(planted, table.outcome, c, dict(zip(names, row)), means))
                  for row, c in zip(table.X, table.cohort)])
    A = np.column_stack([np.ones(len(y)), model.X])
    sw = np.sqrt(table.n)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return coef[1:]


def recovery(models, planted, ctx: _Context) -> dict:
    """Do estimated effects match the planted ones? Checks the pipeline, not any audience."""
    out = {"label": LABEL}
    for outcome, (pooled, by_cohort, screened) in models.items():
        pairs = []
        fits = [("pooled", pooled, None), *((c, m, None) for c, m in by_cohort.items()),
                *(("screened", m, f) for f, m in screened.items())]
        for cohort, model, only in fits:
            truth = _planted_slopes(model, planted, ctx)
            for j, name in enumerate(model.names):
                if name.startswith("cohort:") or (only and name != only):
                    continue
                step = model.swing(j)
                lo, hi = np.percentile(model.boots[:, j] * step, [5, 95]) if len(model.boots) else (0, 0)
                pairs.append({"cohort": cohort, "feature": name, "planted": round(float(truth[j] * step), 3),
                              "estimated": round(float(model.fit.slopes[j] * step), 3),
                              "lo": round(float(lo), 3), "hi": round(float(hi), 3)})
        core = [p for p in pairs if p["cohort"] != "screened"]
        real = [p for p in core if abs(p["planted"]) > 0.05]
        null = [p for p in pairs if abs(p["planted"]) <= 0.05]
        out[outcome] = {
            "planted_effects": len(real),
            "sign_agreement": round(fmean(np.sign(p["planted"]) == np.sign(p["estimated"]) for p in real), 3)
            if real else None,
            # Over every feature (the many tied nulls dilute it) and over the planted effects only.
            "spearman": _spearman([p["planted"] for p in core], [p["estimated"] for p in core]),
            "spearman_planted": _spearman([p["planted"] for p in real], [p["estimated"] for p in real]),
            "ci_coverage": round(fmean(p["lo"] <= p["planted"] <= p["hi"] for p in core), 3) if core else None,
            "null_features": len(null),   # core features with no planted effect, plus every screened extra
            "null_false_alarms": sum(1 for p in null if p["lo"] > 0 or p["hi"] < 0),
            "screened": sum(1 for p in pairs if p["cohort"] == "screened"),
            "pairs": pairs,
        }
    return out


def _brier(p, n, k):
    return float(((k * (1 - p) ** 2 + (n - k) * p ** 2).sum()) / n.sum())


def _auc(p, n, k):
    """Weighted Mann-Whitney AUC over aggregated rows; ties count one half."""
    uniq, inv = np.unique(np.round(p, 12), return_inverse=True)
    pos, neg = np.bincount(inv, k, len(uniq)), np.bincount(inv, n - k, len(uniq))
    below = np.cumsum(neg) - neg
    if not pos.sum() or not neg.sum():
        return None
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def holdout(tables, result: RunResult, folds=FOLDS, lam=LAMBDA) -> dict:
    """Train on some episodes, test on the others (interleaved folds). A cut is tested in the fold of its
    first episode and never trained on with any episode of that fold, so joins cannot leak."""
    numbers = sorted(ep.number for ep in result.episodes)
    k_folds = max(2, min(folds, len(numbers)))
    fold = {n: i % k_folds for i, n in enumerate(numbers)}
    cut_eps = {c.id: {s.episode for s in c.segments} or {numbers[0]} for c in result.candidates}
    out = {"label": LABEL, "folds": k_folds}
    for outcome, table in tables.items():
        promo = OUTCOMES[outcome][0] == "promo"
        preds = {"model": [], "base_rate": [], "cohort_base_rate": []}
        used = 0
        for f in range(k_folds):
            if promo:
                test = np.array([fold[min(cut_eps[u])] == f for u in table.unit])
                train = np.array([all(fold[e] != f for e in cut_eps[u]) for u in table.unit])
            else:
                test = np.array([fold[u] == f for u in table.unit])
                train = ~test
            if not test.any() or train.sum() < 2:
                continue
            tr, te = table.subset(train), table.subset(test)
            cohorts, core = sorted(set(tr.cohort)), _core(tr.names)
            names, X = _design(tr, cohorts, core)
            cols = _columns(X, tr.n, names)
            fit = Logit(X[:, cols], tr.n, tr.k, lam)
            _, Xt = _design(te, cohorts, core)
            base = tr.k.sum() / tr.n.sum()
            by_cohort = {c: tr.k[tr.cohort == c].sum() / tr.n[tr.cohort == c].sum() for c in cohorts}
            preds["model"].append(fit.predict(Xt[:, cols]))
            preds["base_rate"].append(np.full(len(te.n), base))
            preds["cohort_base_rate"].append(np.array([by_cohort.get(c, base) for c in te.cohort]))
            preds.setdefault("_n", []).append(te.n)
            preds.setdefault("_k", []).append(te.k)
            used += 1
        if not used:
            out[outcome] = None
            continue
        n, k = np.concatenate(preds.pop("_n")), np.concatenate(preds.pop("_k"))
        out[outcome] = {"trials": int(n.sum()), "folds_used": used,
                        **{f"{name}_brier": round(_brier(np.concatenate(p), n, k), 4) for name, p in preds.items()},
                        **{f"{name}_auc": (round(a, 3) if (a := _auc(np.concatenate(p), n, k)) is not None else None)
                           for name, p in preds.items()}}
    return out


# -------------------------------------------------------------------------------------------- decisions
def hybrid_score(hits: int, impressions: int, prior: float, total: int) -> float:
    """Smoothed observed rate: the cut's own counts plus up to PRIOR_STRENGTH impressions at the prior."""
    m = max(1, min(PRIOR_STRENGTH, total))
    return (hits + m * prior) / (impressions + m)


def content_prior(features: dict[str, dict[str, float]], trials: dict[str, tuple[int, int]],
                  lam=LAMBDA) -> dict[str, float]:
    """Rate for every cut in `features` from a logistic fit on the cuts in `trials` ({id: (n, k)}, n > 0).
    New cuts with zero impressions get a score from their story features alone."""
    ids = [cid for cid, (n, _) in trials.items() if n > 0 and cid in features]
    if len(ids) < 2:
        return {}
    names = _core(features[ids[0]])
    X = np.array([[features[cid][f] for f in names] for cid in ids])
    n = np.array([trials[cid][0] for cid in ids], float)
    k = np.array([trials[cid][1] for cid in ids], float)
    cols = _columns(X, n, names)
    fit = Logit(X[:, cols], n, k, lam)
    every = list(features)
    p = fit.predict(np.array([[features[cid][f] for f in names] for cid in every])[:, cols])
    return dict(zip(every, map(float, p)))


def promo_rankings(result: RunResult, ctx: _Context, activity) -> dict:
    """Shortlisted promos re-ranked within posting waves for every audience, per objective."""
    audiences = {"all": activity.counts, **dict(sorted(activity.cohorts_by_cut.items()))}
    out = {"label": LABEL, "prior_strength": PRIOR_STRENGTH}
    for objective, field in (("start", "continued"), ("hold", "held_3s")):
        ranked = {}
        for name, stats in audiences.items():
            trials = {cid: (s.impressions, getattr(s, field)) for cid, s in stats.items()}
            prior = content_prior(ctx.cut_x, trials)
            total = sum(n for n, _ in trials.values())
            rows = []
            for promo in result.promos:
                cid = ctx.reps.get(ctx.keys.get(promo.candidate_id))
                if cid is None:
                    continue
                n, k = trials.get(cid, (0, 0))
                p = prior.get(cid)
                score = hybrid_score(k, n, p, total) if p is not None else ((k + 0.5) / (n + 1) if n else None)
                rows.append({"candidate_id": promo.candidate_id, "editorial_rank": promo.rank, "wave": promo.post_wave,
                             "impressions": n, "hits": k, "prior": None if p is None else round(p, 4),
                             "score": None if score is None else round(score, 4), "cold_start": n == 0})
            rows.sort(key=lambda r: (r["wave"], -(r["score"] if r["score"] is not None else -1), r["editorial_rank"]))
            for rank, row in enumerate(rows, 1):
                row["rank"] = rank
            ranked[name] = rows
        base = {r["candidate_id"]: r["rank"] for r in ranked["all"]}
        for name, rows in ranked.items():
            for row in rows:
                row["change_vs_all"] = base[row["candidate_id"]] - row["rank"]
        out[objective] = ranked
    out["changes"] = [{"cohort": name, "candidate_id": r["candidate_id"], "rank_all": base_rank, "rank": r["rank"]}
                      for name, rows in out["start"].items() if name != "all"
                      for r in rows if (base_rank := r["rank"] + r["change_vs_all"]) != r["rank"]]
    out["changes"].sort(key=lambda c: (-abs(c["rank_all"] - c["rank"]), c["cohort"], c["rank"]))
    return out


def _loop_kind(question: str) -> str:
    from . import genome
    cues = getattr(genome, "_CLIFF_CUES", [])  # the genome's open-question mechanisms, when available
    return next((kind for kind, cue in cues if cue.search(question or "")), "outcome")


def episode_insights(result: RunResult, ctx: _Context, sessions, models) -> dict:
    agg = _episode_counts(sessions)
    cohorts = sorted({c for _, c in agg})
    numbers = ctx.numbers
    total = defaultdict(lambda: [0, 0, 0, 0])
    for (n, _), a in agg.items():
        total[n] = [x + y for x, y in zip(total[n], a)]
    skip_n = sum(a[0] - a[1] for (n, _), a in agg.items() if n != numbers[-1])
    skip = sum(a[3] for a in agg.values()) / skip_n if skip_n else 0.0
    complete, cont = models.get("complete", (None,))[0], models.get("continue", (None,))[0]
    share = Counter({c: sum(a[0] for (_, cc), a in agg.items() if cc == c) for c in cohorts})

    def predicted(n, cohort, paywall=True):
        """Model chance a viewer who starts episode n starts n+1, and its two stages."""
        x = dict(ctx.ep_x[n], **({} if paywall else {"paywall_next": 0.0}))
        p_done, p_cont = complete.predict(x, cohort), cont.predict(x, cohort)
        return p_done * p_cont + (1 - p_done) * skip, p_done, p_cont

    def mixed(n, paywall=True):
        parts = [(share[c], predicted(n, c, paywall)) for c in cohorts]
        weight = sum(w for w, _ in parts) or 1
        return [sum(w * v[i] for w, v in parts) / weight for i in range(3)]

    retention = {}
    for cohort in [None, *cohorts]:
        observed, model_r, curve = 1.0, 1.0, []
        for i, n in enumerate(numbers):
            a = total[n] if cohort is None else agg.get((n, cohort), [0, 0, 0, 0])
            curve.append({"episode": n, "observed": None if observed is None else round(observed, 4),
                          "predicted": round(model_r, 4) if complete and cont else None, "sessions": a[0]})
            if i + 1 < len(numbers):
                observed = observed * (a[2] + a[3]) / a[0] if observed is not None and a[0] else None
                if complete and cont:
                    model_r *= (mixed(n) if cohort is None else predicted(n, cohort))[0]
        retention[cohort or "all"] = curve
    out = {"label": LABEL, "skip_ahead_rate": round(skip, 4), "retention": retention}

    analyses = {a.episode: a for a in result.analyses}
    groups = defaultdict(lambda: [0, 0, set()])
    for n in numbers[:-1]:
        x, a = ctx.ep_x[n], analyses.get(n)
        if n in ctx.missing or a is None:
            continue
        loops = int(x["loops_open"])
        for key in (("ending cliffhanger score", str(int(x["ending_cliff"]))),
                    ("questions left unresolved", "0-1" if loops <= 1 else "2" if loops == 2 else "3+"),
                    ("ending question kind", _loop_kind(a.ending_open_question))):
            g = groups[key]
            g[0] += total[n][1]
            g[1] += total[n][2]
            g[2].add(n)
    out["endings"] = [{"group": g, "value": v, "episodes": len(e), "completed": c, "continued": k,
                       "rate": round(k / c, 4) if c else None}
                      for (g, v), (c, k, e) in sorted(groups.items())]
    if not (complete and cont):
        out["dropoff"], out["paywall"] = [], None
        return out

    drops = []
    for n in numbers[:-1]:
        p_next, p_done, p_cont = mixed(n)
        a = total[n]
        stage, model = (("inside the episode", complete) if 1 - p_done >= p_done * (1 - p_cont)
                        else ("after the ending", cont))
        x = model.vector(ctx.ep_x[n])
        pull = [(float(model.fit.slopes[j] * (x[j] - model.fit.mean[j])), name)
                for j, name in enumerate(model.names) if not name.startswith("cohort:")]
        drops.append({"episode": n, "predicted_drop": round(1 - p_next, 4),
                      "observed_drop": round(1 - (a[2] + a[3]) / a[0], 4) if a[0] else None,
                      "sessions": a[0], "where": stage,
                      "drivers": [FEATURES.get(f, (f,))[0] for v, f in sorted(pull)[:2] if v < 0]})
    drops.sort(key=lambda d: -d["predicted_drop"])
    out["dropoff"] = drops[:5]

    lo, hi = UNLOCK_EPISODES
    reach, candidates = 1.0, []
    for n in numbers[:-1]:
        p_next, p_done, _ = mixed(n)
        if lo <= n <= hi and n not in ctx.missing:
            p_after = mixed(n, paywall=False)[2]
            candidates.append({"last_free_episode": n, "paywall_episode": numbers[numbers.index(n) + 1],
                               "continue_after_ending": round(p_after, 4), "reach_end": round(reach * p_done, 4),
                               "unlock_demand": round(reach * p_done * p_after, 4)})
        reach *= p_next
    candidates.sort(key=lambda c: (-c["continue_after_ending"], c["last_free_episode"]))
    rows = result.episode_report or episode_report(result.episodes, result.analyses)
    editor = [u["episode"] for u in season_summary(rows)["unlock_candidates"]]
    ours = [c["last_free_episode"] for c in candidates[:3]]
    out["paywall"] = {"window": [lo, hi], "candidates": candidates[:3], "season_summary_unlock": editor,
                      "agreement": sorted(set(ours) & set(editor)),
                      "note": "Ranked by the predicted chance that a viewer who finishes the last free episode wants "
                              "the next one. Earlier paywalls reach more viewers (reach_end); pricing that trade-off "
                              "needs a paywall-position test."}
    return out


def sample_size(base: float, lift: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Viewers per arm for a two-sided two-proportion z-test to detect base -> base + lift."""
    z = NormalDist().inv_cdf
    other = base + lift
    mean = (base + other) / 2
    root = z(1 - alpha / 2) * math.sqrt(2 * mean * (1 - mean)) + z(power) * math.sqrt(
        base * (1 - base) + other * (1 - other))
    return math.ceil(root ** 2 / lift ** 2)


def ab_design(result: RunResult, hold_rate: float | None, start_rate: float | None) -> dict:
    from .studio import _opening_experiment
    pair = _opening_experiment(result, [{"candidate_id": p.candidate_id}
                                        for p in sorted(result.promos, key=lambda p: p.rank)])
    sizes = []
    for metric, rate, lifts in (("3-second hold", hold_rate, (0.02, 0.03, 0.05)),
                                ("story start", start_rate, (0.01, 0.02, 0.03))):
        if rate is not None and 0 < rate < 1:
            sizes += [{"metric": metric, "base_rate": round(rate, 4), "lift_points": round(100 * d, 1),
                       "viewers_per_arm": sample_size(rate, d)} for d in lifts if rate + d < 1]
    return {
        "pair": pair,
        "arms": "Two openings (different first in-point) with an identical ending and the rest of the cut unchanged.",
        "unit": "Viewer. Assign by a salted hash of viewer_id (50/50) so one viewer always sees one arm.",
        "strata": "Randomise within cohort x placement (feed, reels, retargeting); same bid, cap and schedule.",
        "windows": {"3-second hold": "within the impression", "story start": "24 h after the first impression",
                    "episode 2 start": "7 days (sessions table, matched by exposure id)",
                    "episode 5 reach": "7 days"},
        "analysis": "Primary: 3-second hold per viewer's first impression; secondary: story starts. Two-proportion "
                    "z-test, alpha 0.05 two-sided, power 0.8; fix the sample size up front and do not stop on a peek.",
        "sample_size": sizes,
    }


# ------------------------------------------------------------------------------------------ assemble
def _heatmaps(result, ctx, activity, sessions, viewers=40) -> dict:
    """Small samples of both matrices with anonymous viewer labels. -1 marks never shown / never offered."""
    out = {}
    if activity is not None:
        ranked = sorted(result.promos, key=lambda p: p.rank)
        cuts = [c for c in dict.fromkeys(ctx.reps.get(ctx.keys.get(p.candidate_id)) for p in ranked) if c][:20]
        rows = [cells for _, cells in sorted(activity.viewer_cells.items()) if set(cells) & set(cuts)][:viewers]
        out["promo"] = {"cuts": cuts, "viewers": [f"V{i + 1:02d}" for i in range(len(rows))],
                        "cells": [[impression_state(cells[c]) if c in cells else -1 for c in cuts] for cells in rows],
                        "states": PROMO_STATES}
    if sessions:
        by_viewer = defaultdict(dict)
        for s in sessions:
            by_viewer[s.viewer_id][s.episode] = s
        rows = [cells for _, cells in sorted(by_viewer.items())][:viewers]
        out["episode"] = {"episodes": ctx.numbers, "viewers": [f"V{i + 1:02d}" for i in range(len(rows))],
                          "cells": [[session_state(cells[n]) if n in cells else -1 for n in ctx.numbers]
                                    for cells in rows], "states": EPISODE_STATES}
    return out


def _matrix_sizes(ctx, impressions, sessions, activity) -> dict:
    out = {"label": LABEL}
    if activity is not None:
        viewers = len({e.viewer_id for e in impressions})
        cells = sum(len(c) for c in activity.viewer_cells.values())
        cuts = len(ctx.reps)
        out["promo"] = {"viewers": viewers, "cuts": cuts, "cuts_shown": sum(1 for s in activity.counts.values()
                                                                            if s.impressions),
                        "impressions": len(impressions), "cells": cells,
                        "density": round(cells / (viewers * cuts), 4) if viewers and cuts else 0,
                        "observed_skips": sum(1 for e in impressions if e.watch_seconds < 3),
                        "never_shown_cells": viewers * cuts - cells}
    if sessions:
        viewers = len({s.viewer_id for s in sessions})
        starts = {e.impression_id for e in impressions if e.continued}
        linked = [s.exposure for s in sessions if s.exposure]
        out["episode"] = {"viewers": viewers, "episodes": len(ctx.numbers), "cells": len(sessions),
                          "started": sum(s.started for s in sessions),
                          "observed_not_started": sum(not s.started for s in sessions),
                          "never_offered_cells": viewers * len(ctx.numbers) - len(sessions),
                          "density": round(len(sessions) / (viewers * len(ctx.numbers)), 4),
                          "from_promo": len(linked), "unmatched_exposures": sum(1 for x in linked if x not in starts)}
    return out


def analyze(result: RunResult, impressions, sessions, *, source: str, planted=None, paywall=None,
            bootstrap=BOOTSTRAP, seed=0, keys=None, graph=None) -> dict:
    """Matrices, models, validations and decisions for one show's activity (simulated or reported)."""
    simulated = source == "simulated"
    ctx = _context(result, paywall, keys, graph)
    activity = audience.aggregate(result, impressions, keys=ctx.keys) if impressions else None
    sessions = check_sessions(sessions, result) if sessions else []
    tables = _tables(ctx, activity, sessions)
    models = _fit_all(tables, bootstrap, seed)
    label = LABEL if simulated else "REPORTED"
    report = {
        "label": label,
        "note": (("SIMULATED activity drawn from PLANTED in hookcut/engagement.py. " if planted else
                  "SIMULATED activity (the uploaded rows are marked synthetic). ") +
                 "It checks that the pipeline recovers planted effects; it is not evidence about Kuku's viewers."
                 if simulated else "Reported activity from the uploaded CSVs."),
        "show": result.show, "source": source, "feature_source": ctx.feature_source,
        "planted": planted if simulated else None, "paywall_episode": paywall,
        "matrices": _matrix_sizes(ctx, impressions, sessions, activity),
        "models": {o: {"label": label, "outcome": OUTCOMES[o][1], "grain": OUTCOMES[o][0],
                       "base_rate": round(float(m.table.k.sum() / m.table.n.sum()), 4),
                       "effects": m.effects(ctx.labels),
                       "by_cohort": {c: cm.effects(ctx.labels) for c, cm in by.items()},
                       "screened": sorted((e for f, sm in screened.items() for e in sm.effects(ctx.labels, f)),
                                          key=lambda e: -abs(e["log_odds"]))}
                   for o, (m, by, screened) in models.items()},
        "recovery": recovery(models, planted, ctx) if simulated and planted else None,
        "holdout": holdout(tables, result),
        "promo_ranking": promo_rankings(result, ctx, activity) if activity is not None else None,
        "episodes": episode_insights(result, ctx, sessions, models) if sessions else None,
        "episode_features": {str(n): {k: round(v, 4) for k, v in x.items()} for n, x in ctx.ep_x.items()},
        "heatmaps": _heatmaps(result, ctx, activity, sessions),
    }
    rate = lambda o: report["models"][o]["base_rate"] if o in report["models"] else None  # noqa: E731
    report["ab_test"] = ab_design(result, rate("hold"), rate("start"))
    for section in ("holdout", "promo_ranking", "episodes", "recovery", "matrices"):
        if isinstance(report[section], dict):
            report[section]["label"] = label
    return report


def load_activity(result: RunResult, *, simulate_activity=None, events_csv=None, sessions_csv=None, planted=None,
                  seed=None, viewers=None, paywall=None, keys=None, graph=None) -> dict:
    """Simulated or imported rows, never mixed. simulate_activity=None simulates only when no CSV is given;
    every simulated output is labelled SIMULATED."""
    given = events_csv is not None or sessions_csv is not None
    if simulate_activity and given:
        raise ValueError("Choose simulated or reported activity, not both.")
    keys = keys or audience.clip_keys(result)
    if simulate_activity or (simulate_activity is None and not given):
        planted = planted_config(planted, seed=seed, viewers=viewers, paywall=paywall)
        impressions, sessions = simulate(result, planted, keys=keys, graph=graph)
        return {"impressions": impressions, "sessions": sessions, "source": "simulated", "planted": planted,
                "paywall": planted["paywall"]["episode"], "keys": keys}
    impressions = audience.read_events_csv(events_csv, result, keys=keys) if events_csv else []
    sessions = read_sessions_csv(sessions_csv, result) if sessions_csv else []
    if not impressions and not sessions:
        raise ValueError("No activity: simulate, or supply an impression or session CSV with rows.")
    # Rows marked synthetic stay SIMULATED after a round trip through CSV (as in studio).
    synthetic = any(r.source == "synthetic" for r in [*impressions[:1], *sessions[:1]])
    return {"impressions": impressions, "sessions": sessions, "source": "simulated" if synthetic else "reported",
            "planted": None, "paywall": paywall, "keys": keys}


def build_engagement(result: RunResult, *, bootstrap=BOOTSTRAP, graph=None, **activity) -> dict:
    """The engagement report for one run (see load_activity for the activity options)."""
    rows = load_activity(result, graph=graph, **activity)
    return analyze(result, rows["impressions"], rows["sessions"], source=rows["source"], planted=rows["planted"],
                   paywall=rows["paywall"], bootstrap=bootstrap, keys=rows["keys"], graph=graph)


def write_engagement(report: dict, out_dir, impressions=(), sessions=(), sample_rows=1500) -> dict[str, str]:
    """engagement.json, plus small CSV samples when the activity is simulated (reported rows stay out)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"engagement": out / "engagement.json"}
    if report["source"] == "simulated":
        report = dict(report, samples={"promo": "simulated_promo_impressions_sample.csv",
                                       "episode": "simulated_episode_sessions_sample.csv", "rows": sample_rows})
        paths["promo_sample"] = out / report["samples"]["promo"]
        paths["episode_sample"] = out / report["samples"]["episode"]
        paths["promo_sample"].write_text(audience.events_csv(list(impressions)[:sample_rows]), encoding="utf-8")
        paths["episode_sample"].write_text(sessions_csv(list(sessions)[:sample_rows]), encoding="utf-8")
    else:
        report = {k: v for k, v in report.items() if k != "heatmaps"}
    paths["engagement"].write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                                   encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def _config_defaults() -> dict:
    try:
        from .config import load_config
        return load_config().get("engagement") or {}
    except Exception:  # config is optional for this offline tool
        return {}


def _print(report: dict) -> None:
    print(f"[{report['label']}] {report['show']} | features: {report['feature_source']}")
    for grain, size in report["matrices"].items():
        if isinstance(size, dict):
            print(f"  {grain} matrix: " + ", ".join(f"{k}={v}" for k, v in size.items()))
    for outcome, model in report["models"].items():
        print(f"  {outcome} ({model['outcome']}, base {model['base_rate']:.1%}):")
        for e in model["effects"][:5]:
            print(f"    {e['label']:<48} {e['change']:>16}  {e['log_odds']:+.2f} log-odds "
                  f"[{e['lo']:+.2f}, {e['hi']:+.2f}]  {e['points']:+.1f} pts")
    for outcome, r in (report["recovery"] or {}).items():
        if isinstance(r, dict):
            print(f"  recovery {outcome}: sign {r['sign_agreement']} ({r['planted_effects']} effects) spearman "
                  f"{r['spearman']} (planted only {r['spearman_planted']}) "
                  f"coverage {r['ci_coverage']} false alarms {r['null_false_alarms']}/{r['null_features']}")
    for outcome, h in report["holdout"].items():
        if isinstance(h, dict):
            print(f"  holdout {outcome}: Brier {h['model_brier']} vs base {h['base_rate_brier']} "
                  f"(cohort base {h['cohort_base_rate_brier']}); AUC {h['model_auc']} vs {h['cohort_base_rate_auc']}")
    if report.get("episodes") and report["episodes"].get("paywall"):
        drops = report["episodes"]["dropoff"]
        print("  drop-off: " + ", ".join(f"E{d['episode']} {d['predicted_drop']:.0%}" for d in drops))
        pw = report["episodes"]["paywall"]
        print("  paywall after: " + ", ".join(f"E{c['last_free_episode']}" for c in pw["candidates"])
              + f" | season_summary: {pw['season_summary_unlock']}")


def main(argv=None):
    defaults = _config_defaults()
    parser = argparse.ArgumentParser(description="Activity matrices, model and decisions for a saved run.")
    parser.add_argument("--run", type=Path, required=True, help="Run folder with result.json (or the file itself).")
    parser.add_argument("--out", type=Path, help="Folder for engagement.json (default: the run folder).")
    parser.add_argument("--simulate", action="store_true", help="Draw SIMULATED activity from PLANTED.")
    parser.add_argument("--events", type=Path, help="Promo impression CSV (audience.Exposure columns).")
    parser.add_argument("--sessions", type=Path, help="Episode session CSV (Session columns).")
    parser.add_argument("--paywall", type=int, default=defaults.get("paywall_episode"),
                        help="First paywalled episode (0 = none).")
    parser.add_argument("--viewers", type=int, default=defaults.get("viewers"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed"))
    parser.add_argument("--bootstrap", type=int, default=defaults.get("bootstrap", BOOTSTRAP))
    args = parser.parse_args(argv)
    if args.simulate == bool(args.events or args.sessions):
        parser.error("use --simulate, or --events and/or --sessions")
    path = args.run / "result.json" if args.run.is_dir() else args.run
    try:
        result = RunResult.model_validate_json(path.read_bytes())
        rows = load_activity(
            result, simulate_activity=args.simulate,
            events_csv=args.events.read_text(encoding="utf-8-sig") if args.events else None,
            sessions_csv=args.sessions.read_text(encoding="utf-8-sig") if args.sessions else None,
            seed=args.seed, viewers=args.viewers, paywall=args.paywall)
        report = analyze(result, rows["impressions"], rows["sessions"], source=rows["source"],
                         planted=rows["planted"], paywall=rows["paywall"], bootstrap=args.bootstrap, keys=rows["keys"])
        paths = write_engagement(report, args.out or path.parent, rows["impressions"], rows["sessions"])
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cannot build the engagement report: {exc}\n")
    _print(report)
    print("  wrote " + ", ".join(paths.values()))


if __name__ == "__main__":
    main()
