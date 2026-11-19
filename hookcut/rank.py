"""Turn judged candidates into the ranked promo list. Plain code, no model calls.

1. Final score = script-side prelim score (min-max normalised) + the cold viewer's evidence.
   A cold-viewer "fail" never enters the list; "borderline" may, with a flag.
2. Diversity: MMR over embeddings of (open question + hook line), plus hard caps (episodes, time overlap).
3. Posting order in three waves (cold audiences -> escalation -> big reveals for warm audiences).
4. Variants of each final promo: other openings and length tiers for the same cliffhanger.
5. Optional, with the show's story graph (graph=None ranks exactly as without it), three rules:
   a. protect the twist: a clip containing an identity reveal or relationship reversal (story alert with an evidence
      beat inside the clip) goes to wave 3, unless the alert is in the early season (that is the premise wave 1 sells);
   b. feuds: core conflict pairs on opposite sides (story_graph.feud_pairs) on screen get a "feud: A vs B" flag, and
      each of the top 3 feuds missing from the list gets its best cold-viewer-passed clip swapped in for the weakest
      pick when the caps allow (a diversity rule, not a quota);
   c. leads open better cold: a small flat bonus when the hook beat's speaker or subject is a top-3 promo-pull character.
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from . import candidates as cg
from .schema import Beat, Candidate, ColdViewerVerdict, Episode, EpisodeAnalysis, Promo, Segment
from .textfmt import ep_tc

# Final score. The cold viewer (an independent model that sees only the clip) can add up to 5 points,
# the script-side prelim score at most 2: the stranger's reaction is the closest thing we have to an audience.
FINAL_WEIGHTS = {
    "prelim": 2.0,           # prelim score, min-max normalised to 0-1 over all candidates
    "standalone": 0.4,       # per point (0-5): a stranger can follow it
    "keep_watching": 0.4,    # per point (0-5): a stranger wants the next episode
    "hook_lands": 1.0,       # the stranger says the first 3 s would stop the scroll
    "borderline": -0.5,      # verdict "borderline": allowed in, flagged, pushed down a little
    "lead_hook": 0.2,        # story graph only: the hook beat's speaker or subject is a top-3 promo-pull character.
                             # Flat, once per clip (capped by construction): half of one cold-viewer point, less than
                             # the borderline penalty, so it reorders near-ties but cannot outweigh a verdict
}
MMR_LAMBDA = 0.75            # 1.0 = pure score order, 0.0 = pure novelty
MAX_TIME_OVERLAP = 0.30      # two picks may share at most 30% of the shorter clip
EARLY_SEASON_FRAC = 0.30     # wave 1 draws from the first ~30% of episodes
WAVE1_MAX_SPOILER = 1
WAVE1_MIN_STANDALONE = 4
WAVE3_MIN_SPOILER = 2
MAX_ALT_OPENINGS = 3
LONG_CUT_S = (90.0, 120.0)   # optional tier, outside the brief's 20-60 s
EMBED_MODEL = "minishlab/potion-base-8M"
# story-graph rules (docstring item 5)
TOP_PULL = 3                 # "leads": the characters with the 3 highest promo-pull shares
TOP_FEUDS = 3                # the list should cover the 3 strongest feuds when passing clips exist
REVEAL_ALERTS = ("identity_reveal", "relationship_reversal")
REVEAL_FLAG = "contains a reveal: keep for warm audiences"
PREMISE_FLAG = "early-season reveal: the premise, not held back"

_model = None                # model2vec StaticModel, loaded once per process
_model_failed = False


# ---------------------------------------------------------------------------- similarity
def _embedder():
    global _model, _model_failed
    if _model is None and not _model_failed:
        try:
            from model2vec import StaticModel
            _model = StaticModel.from_pretrained(EMBED_MODEL)
        except Exception:   # offline / no cache / package missing -> token Jaccard
            _model_failed = True
    return _model


_STOP = set("a an the of to and or is are was be in on at for with who what why how does do did will his her "
            "their they he she it this that i you we me my your".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOP}


def similarity_matrix(texts: list[str], backend: str = "model2vec") -> tuple[np.ndarray, str]:
    """Pairwise similarity in [0, 1]: cosine of model2vec embeddings, or token Jaccard as a fallback."""
    model = _embedder() if backend == "model2vec" and texts else None
    if model is not None:
        v = np.asarray(model.encode(texts), dtype=float)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return np.clip(v @ v.T, 0.0, 1.0), "model2vec"
    toks = [_tokens(t) for t in texts]
    n = len(texts)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            u = toks[i] | toks[j]
            m[i, j] = m[j, i] = len(toks[i] & toks[j]) / len(u) if u else 0.0
    return m, "jaccard"


# ------------------------------------------------------------------------------- helpers
def beat_text(b: Beat, limit: int = 200) -> str:
    if b.kind in cg.SPEECH:
        return f"{b.speaker or 'VOICE'}: {b.text}"
    return b.text if len(b.text) <= limit else b.text[: limit - 1] + "…"


def _episodes_of(c: Candidate) -> set[int]:
    return {s.episode for s in c.segments}


def _overlap_frac(a: Candidate, b: Candidate) -> float:
    shorter = min(a.duration, b.duration)
    return cg.time_overlap(a.segments, b.segments) / shorter if shorter > 0 else 0.0


def final_score(c: Candidate, v: Optional[ColdViewerVerdict], prelim_norm: float, lead_hook: bool = False) -> float:
    w = FINAL_WEIGHTS
    s = w["prelim"] * prelim_norm
    if v is not None:
        s += w["standalone"] * v.standalone + w["keep_watching"] * v.keep_watching
        s += w["hook_lands"] * (1.0 if v.hook_lands else 0.0)
        s += w["borderline"] if v.verdict == "borderline" else 0.0
    s += w["lead_hook"] if lead_hook else 0.0
    return round(s, 4)


def mmr_select(items: list[Candidate], scores: dict[str, float], sim: np.ndarray, n: int,
               max_per_episode: int, lam: float = MMR_LAMBDA) -> tuple[list[int], dict[int, float]]:
    """Greedy maximal marginal relevance under hard caps. Returns picked indices (in pick order) and
    each pick's highest similarity to an earlier pick."""
    top = max(scores.values(), default=0.0) or 1.0
    rel = [scores[c.id] / top for c in items]
    picked: list[int] = []
    max_sim: dict[int, float] = {}
    per_ep: dict[int, int] = {}
    while len(picked) < n:
        best, best_val = None, -math.inf
        for i, c in enumerate(items):
            if i in picked or any(per_ep.get(e, 0) >= max_per_episode for e in _episodes_of(c)):
                continue
            if any(_overlap_frac(c, items[j]) > MAX_TIME_OVERLAP for j in picked):
                continue
            redundancy = max((sim[i, j] for j in picked), default=0.0)
            val = lam * rel[i] - (1 - lam) * redundancy
            if val > best_val + 1e-12:
                best, best_val = i, val
        if best is None:
            break
        max_sim[best] = float(max((sim[best, j] for j in picked), default=0.0))
        picked.append(best)
        for e in _episodes_of(items[best]):
            per_ep[e] = per_ep.get(e, 0) + 1
    return picked, max_sim


def assign_wave(c: Candidate, v: Optional[ColdViewerVerdict], early: set[int], reveal: bool = False) -> int:
    if c.spoiler >= WAVE3_MIN_SPOILER or reveal:
        return 3
    if (c.segments[0].episode in early and c.spoiler <= WAVE1_MAX_SPOILER and v is not None
            and v.verdict != "fail" and v.standalone >= WAVE1_MIN_STANDALONE):
        return 1
    return 2


def early_episodes(episodes: list[Episode]) -> set[int]:
    nums = sorted(e.number for e in episodes)
    return set(nums[: max(1, math.ceil(EARLY_SEASON_FRAC * len(nums)))]) if nums else set()


# ------------------------------------------------------------------------- story-graph rules
def story_context(graph, episodes: list[Episode], pool: list[Candidate]) -> dict:
    """Per candidate, what the story-graph rules need. graph: story_graph.StoryGraph or its JSON dict.
    A beat counts as inside a clip when it is complete inside a segment (the rule genome.py uses)."""
    from . import story_graph as sg     # lazy: ranking without a graph never touches the genome layer
    g = graph if isinstance(graph, sg.StoryGraph) else sg.StoryGraph.model_validate(graph)
    eps = {e.number: e for e in episodes}
    on = sg.cast_by_beat(episodes, g)
    names = {c.id: c.name for c in g.characters}
    pull = {c.id: c.promo_pull for c in g.characters}
    leads = {c.id for c in sorted(g.characters, key=lambda c: (-c.promo_pull, c.id))[:TOP_PULL] if c.promo_pull > 0}

    def label(e) -> str:     # the bigger promo draw first: "Mara vs Rosalind"
        a, b = sorted((e.a, e.b), key=lambda x: (-pull.get(x, 0.0), x))
        return f"{names.get(a, a)} vs {names.get(b, b)}"

    feuds = [(e, label(e)) for e in sg.feud_pairs(g, TOP_FEUDS)]
    alerts = [a for a in g.alerts if a.type in REVEAL_ALERTS and a.beat_ids]
    early = early_episodes(episodes)
    cuts = {}
    for c in pool:
        inside = {b.id for b in sg.beats_inside(eps, c.segments)}
        present = {x for bid in inside for x in on.get(bid, [])}
        hits = [a for a in alerts if inside & set(a.beat_ids)]
        cuts[c.id] = {"lead_hook": bool(leads & set(on.get(c.hook_beat_id, []))),
                      "feuds": [name for e, name in feuds if e.a in present and e.b in present],
                      "twist": [a.id for a in hits if a.ep not in early],
                      "premise": [a.id for a in hits if a.ep in early]}
    return {"cuts": cuts, "feuds": [name for _, name in feuds], "leads": sorted(leads)}


def _fits(c: Candidate, others: list[Candidate], max_per_episode: int) -> bool:
    per_ep: dict[int, int] = {}
    for o in others:
        for e in _episodes_of(o):
            per_ep[e] = per_ep.get(e, 0) + 1
    return (all(per_ep.get(e, 0) < max_per_episode for e in _episodes_of(c))
            and all(_overlap_frac(c, o) <= MAX_TIME_OVERLAP for o in others))


def cover_feuds(picked: list[int], pool: list[Candidate], scores: dict[str, float], by_id: dict, ctx: dict,
                sim: np.ndarray, max_per_episode: int) -> tuple[list[int], dict[int, float]]:
    """Rule 5b: each top feud missing from the picks gets its best cold-viewer-passed clip, swapped in for the
    lowest-scoring pick that is not another feud's only cover, if the episode and overlap caps allow. No passing
    clip, or no room under the caps: the feud stays uncovered. Returns picks and their max similarity to earlier picks."""
    picked = list(picked)
    feuds_of = lambda i: set(ctx["cuts"][pool[i].id]["feuds"])
    for feud in ctx["feuds"]:
        if any(feud in feuds_of(i) for i in picked):
            continue
        sole = {i for i in picked for f in feuds_of(i) if sum(f in feuds_of(j) for j in picked) == 1}
        out = [i for i in picked if i not in sole]
        if not out:
            continue
        weakest = min(out, key=lambda i: (scores[pool[i].id], pool[i].id))
        rest = [pool[i] for i in picked if i != weakest]
        for j in sorted(range(len(pool)), key=lambda j: (-scores[pool[j].id], pool[j].id)):
            v = by_id.get(pool[j].id)
            if (j not in picked and feud in feuds_of(j) and v is not None and v.verdict == "pass"
                    and _fits(pool[j], rest, max_per_episode)):
                picked[picked.index(weakest)] = j
                break
    max_sim = {i: float(max((sim[i, j] for j in picked[:k]), default=0.0)) for k, i in enumerate(picked)}
    return picked, max_sim


# ------------------------------------------------------------------------------ variants
def _seg_dict(ep: int, start: float, end: float, ep_obj: Episode, last_id: str) -> dict:
    return Segment(episode=ep, start=start, end=end, first_beat_id=cg.first_beat_from(ep_obj, start).id,
                   last_beat_id=last_id).model_dump()


def build_variants(c: Candidate, episodes: dict[int, Episode], idx, beats, cfg: dict) -> list[dict]:
    """Same cliffhanger, different cut: other hook openings, the shortest and longest valid cut and,
    if cfg.long_cut, a 90-120 s long cut. Every variant obeys the same cut rules as a candidate."""
    first, last = c.segments[0], c.segments[-1]
    ep_first, ep_last = episodes[first.episode], episodes[last.episode]
    cliff = beats[c.cliff_beat_id]
    resolver = cg._resolver(idx, beats, cliff.id)
    max_lead = cfg.get("hook_window_s", 3.0) - 1.0
    middle = c.segments[1:-1]
    min_first = cg.MULTI_MIN_SEGMENT_S if len(c.segments) > 1 else 0.0

    # end options for the last segment: as chosen, or tight on the cliff beat (no shot extension)
    ends = {(last.end, last.last_beat_id)}
    tight = cg.close_at(ep_last, cliff, resolver, extend=False)
    if tight:
        ends.add((tight[0], tight[1]))

    hooks = [b for b in ep_first.beats if cg.hook_of(idx, b.id) >= cfg.get("hook_min", 3)
             and (len(c.segments) > 1 or b.start < cliff.start - cg.TOL)]
    options = []   # (segments, hook beat)
    for h in hooks:
        opening = cg.open_at(ep_first, h, max_lead)
        if not opening:
            continue
        for end, last_id in sorted(ends):
            if len(c.segments) == 1:
                segs = [_seg_dict(first.episode, opening[0], end, ep_first, last_id)]
            else:
                segs = ([_seg_dict(first.episode, opening[0], first.end, ep_first, first.last_beat_id)]
                        + [s.model_dump() for s in middle]
                        + [dict(last.model_dump(), end=end, last_beat_id=last_id)])
            seg_objs = [Segment(**s) for s in segs]
            if seg_objs[0].end - seg_objs[0].start < max(min_first, 0.5):
                continue
            if cg.answer_inside(resolver, seg_objs):
                continue
            options.append((segs, h, cg.seg_length(seg_objs)))

    main_key = tuple((s.episode, s.start, s.end) for s in c.segments)
    seen = {main_key}
    out: list[dict] = []

    def add(label: str, segs: list[dict], h: Beat, dur: float, note: str) -> None:
        key = tuple((s["episode"], s["start"], s["end"]) for s in segs)
        if key in seen:
            return
        seen.add(key)
        out.append({"label": label, "segments": segs, "duration": dur, "hook_beat_id": h.id, "note": note})

    lo, hi = cfg.get("min_s", 20), cfg.get("max_s", 60)
    valid = [o for o in options if lo <= o[2] <= hi]
    alts = [o for o in valid if o[1].id != c.hook_beat_id and o[0][-1]["end"] == last.end]
    alts.sort(key=lambda o: (-cg.hook_of(idx, o[1].id), abs(o[2] - c.duration)))
    used_hooks: set[str] = set()
    for segs, h, dur in alts:
        if len(used_hooks) >= MAX_ALT_OPENINGS or h.id in used_hooks:
            continue
        used_hooks.add(h.id)
        hs = idx.get(h.id)
        kind = (hs.hook_type if hs and hs.hook_type else h.kind)
        add(f"alt opening: {kind}", segs, h, dur,
            f"opens on {h.kind} at {ep_tc(h.episode, h.start)} (hook {cg.hook_of(idx, h.id)}/5)")
    if valid:
        segs, h, dur = min(valid, key=lambda o: (o[2], o[1].id))
        add("shortest cut", segs, h, dur, f"{dur:.1f} s, same cliffhanger")
        segs, h, dur = max(valid, key=lambda o: (o[2], o[1].id))
        add("longest cut", segs, h, dur, f"{dur:.1f} s, same cliffhanger")
    if cfg.get("long_cut"):
        longs = [o for o in options if LONG_CUT_S[0] <= o[2] <= LONG_CUT_S[1]]
        if longs:
            segs, h, dur = max(longs, key=lambda o: (cg.hook_of(idx, o[1].id), -abs(o[2] - 105), o[1].id))
            add("long cut (outside brief)", segs, h, dur,
                f"{dur:.1f} s: outside the brief's {lo}-{hi} s; competitors' top spend often runs 90-120 s")
    return out


# -------------------------------------------------------------------------------- main
def rank_promos(candidates: list[Candidate], verdicts: list[ColdViewerVerdict], episodes: list[Episode],
                analyses: list[EpisodeAnalysis], promo_cfg: dict, graph=None) -> list[Promo]:
    """graph: the show's story graph (story_graph.StoryGraph or its JSON dict) turns on the rules in the module
    docstring, item 5; None gives exactly the ranking without them."""
    cfg = promo_cfg or {}
    if not candidates:
        return []
    by_id = {v.candidate_id: v for v in verdicts}
    eps = {e.number: e for e in episodes}
    idx = cg.score_index(analyses)
    beats = cg.beat_index(episodes)

    prelims = [c.prelim_score for c in candidates]
    lo, hi = min(prelims), max(prelims)
    norm = {c.id: (c.prelim_score - lo) / (hi - lo) if hi > lo else 1.0 for c in candidates}

    pool = [c for c in candidates if (by_id.get(c.id) is None or by_id[c.id].verdict != "fail")
            and c.hook_beat_id in beats and c.cliff_beat_id in beats]
    if not pool:
        return []
    ctx = story_context(graph, episodes, pool) if graph is not None else None
    story = (lambda c: ctx["cuts"][c.id]) if ctx else (lambda c: {"lead_hook": False, "feuds": [], "twist": [],
                                                                   "premise": []})
    scores = {c.id: final_score(c, by_id.get(c.id), norm[c.id], story(c)["lead_hook"]) for c in pool}
    pool.sort(key=lambda c: (-scores[c.id], c.id))

    def question(c: Candidate) -> str:
        v = by_id.get(c.id)
        return v.open_question if v and v.verdict != "fail" and v.open_question else c.open_question

    texts = [f"{question(c)} {beats[c.hook_beat_id].text}" for c in pool]
    sim, _ = similarity_matrix(texts, cfg.get("embeddings", "model2vec"))   # "jaccard" skips the model
    picked, max_sim = mmr_select(pool, scores, sim, cfg.get("final_top_n", 20), cfg.get("max_per_episode_in_top", 2))
    if ctx and ctx["feuds"]:
        picked, max_sim = cover_feuds(picked, pool, scores, by_id, ctx, sim, cfg.get("max_per_episode_in_top", 2))

    early = early_episodes(episodes)
    rows = []
    for i in picked:
        c = pool[i]
        v = by_id.get(c.id)
        rows.append((assign_wave(c, v, early, reveal=bool(story(c)["twist"])), -scores[c.id], c.id, c, v, max_sim[i]))
    rows.sort(key=lambda r: r[:3])

    promos = []
    for rank, (wave, _, _, c, v, msim) in enumerate(rows, start=1):
        hook, cliff = beats[c.hook_beat_id], beats[c.cliff_beat_id]
        flags = []
        if c.needs_context:
            flags.append("needs context: " + "; ".join(c.needs_context))
        if c.spoiler >= WAVE3_MIN_SPOILER:
            flags.append(f"spoiler (level {c.spoiler}/3): hold for warm audiences")
        if c.kind == "multi":
            flags.append("multi-segment: check continuity")
        if hook.kind not in cg.SPEECH:
            flags.append("visual-only hook: check footage")
        if v is None:
            flags.append("not cold-viewer tested")
        elif v.verdict == "borderline":
            flags.append("borderline cold-viewer")
        if v is not None and v.confusions:
            flags.append("cold viewer confused: " + "; ".join(v.confusions[:2]))
        st = story(c)
        if st["twist"]:
            flags.append(REVEAL_FLAG)
        elif st["premise"]:
            flags.append(PREMISE_FLAG)
        flags += [f"feud: {f}" for f in st["feuds"]]
        sc = {"final": scores[c.id], "prelim": c.prelim_score, "prelim_norm": round(norm[c.id], 4),
              "hook": c.hook_score, "cliff": c.cliff_score, "spoiler": float(c.spoiler),
              "duration_fit": round(cg.duration_fit(c.duration, cfg), 3), "max_similarity": round(msim, 3)}
        if v is not None:
            sc.update(standalone=float(v.standalone), keep_watching=float(v.keep_watching),
                      hook_lands=1.0 if v.hook_lands else 0.0)
        if ctx:
            sc["lead_hook"] = 1.0 if st["lead_hook"] else 0.0
        promos.append(Promo(
            rank=rank, candidate_id=c.id, kind=c.kind, segments=c.segments, duration=c.duration,
            hook_beat_id=c.hook_beat_id, cliff_beat_id=c.cliff_beat_id,
            hook_time=ep_tc(hook.episode, hook.start), hook_text=beat_text(hook),
            cliffhanger_time=ep_tc(cliff.episode, cliff.start), cliffhanger_text=beat_text(cliff),
            open_question=question(c), why_it_works="", post_wave=wave, flags=flags, scores=sc,
            cold_viewer=v, variants=build_variants(c, eps, idx, beats, cfg)))
    return promos
