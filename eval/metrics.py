"""Small, checkable statistics (numpy only): bootstrap CIs, Cohen's kappa, Spearman, interval IoU."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

B, CI, SEED = 2000, 0.90, 0


def _arr(xs: Sequence) -> np.ndarray:
    return np.asarray([float(x) for x in xs if x is not None], dtype=float)


def bootstrap_ci(xs: Sequence, b: int = B, ci: float = CI, seed: int = SEED) -> Optional[tuple[float, float]]:
    """Percentile bootstrap CI of the mean. Works for 0/1 data (a rate) and scores alike."""
    x = _arr(xs)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(b, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def bootstrap_diff_ci(xs: Sequence, ys: Sequence, b: int = B, ci: float = CI,
                      seed: int = SEED) -> Optional[tuple[float, float]]:
    """CI of mean(xs) - mean(ys), resampling each (independent) set separately."""
    x, y = _arr(xs), _arr(ys)
    if len(x) == 0 or len(y) == 0:
        return None
    rng = np.random.default_rng(seed)
    d = (x[rng.integers(0, len(x), size=(b, len(x)))].mean(axis=1)
         - y[rng.integers(0, len(y), size=(b, len(y)))].mean(axis=1))
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def summarize(xs: Sequence) -> dict:
    """{'n', 'mean', 'ci90'} for a list that may contain None (missing = left out, n says how many count)."""
    x = _arr(xs)
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci90": bootstrap_ci(x)}


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    """Binary Cohen's kappa. None when undefined (both raters constant and identical: expected agreement 1)."""
    if len(a) != len(b) or not a:
        return None
    x, y = np.asarray(a, bool), np.asarray(b, bool)
    po = float((x == y).mean())
    pa, pb = x.mean(), y.mean()
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    return None if pe >= 1.0 - 1e-12 else (po - pe) / (1 - pe)


def rankdata(xs: Sequence[float]) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank: the usual Spearman convention."""
    x = np.asarray(xs, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def agreement(v1: list[dict], v2: list[dict]) -> dict:
    """Agreement between two judges on the same clips. v1/v2: verdict dicts keyed by position (same clips)."""
    pairs = [(x, y) for x, y in zip(v1, v2) if x and y]
    n = len(pairs)
    if not n:
        return {"n": 0}
    same = [x["verdict"] == y["verdict"] for x, y in pairs]
    return {
        "n": n,
        "verdict_agreement": float(np.mean(same)),
        "kappa_pass": cohen_kappa([x["verdict"] == "pass" for x, _ in pairs],
                                  [y["verdict"] == "pass" for _, y in pairs]),
        "spearman_keep_watching": spearman([x["keep_watching"] for x, _ in pairs],
                                           [y["keep_watching"] for _, y in pairs]),
        "mean_abs_diff_standalone": float(np.mean([abs(x["standalone"] - y["standalone"]) for x, y in pairs])),
        "pass_rate_1": float(np.mean([x["verdict"] == "pass" for x, _ in pairs])),
        "pass_rate_2": float(np.mean([y["verdict"] == "pass" for _, y in pairs])),
    }


# ------------------------------------------------------------------ intervals
def _merge(iv: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted((min(a, b), max(a, b)) for a, b in iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _length(iv: Sequence[tuple[float, float]]) -> float:
    return sum(e - s for s, e in _merge(iv))


def interval_iou(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    """Temporal IoU of two sets of intervals (a multi-segment clip vs a labelled window)."""
    inter = 0.0
    for s1, e1 in _merge(a):
        for s2, e2 in _merge(b):
            inter += max(0.0, min(e1, e2) - max(s1, s2))
    union = _length(list(a) + list(b))
    return inter / union if union > 0 else 0.0
