"""Script-editor view: how strong each episode's opening hook and ending cliffhanger are, and whether the
next episode picks the cliffhanger up. Plain code over the episode analyses."""
from __future__ import annotations

from typing import Optional

from .schema import Episode, EpisodeAnalysis
from .textfmt import tc

WEAK_SCORE = 3              # an opening hook / ending cliffhanger below this is flagged weak
PICKUP_LIMIT_S = 30.0       # the previous cliffhanger should be picked up within this
RECAP_LIMIT_S = 5.0         # a recap longer than this costs a scroller's patience
STRONG_BEAT = 3             # beats at or above this (hook or cliff) count as promo-ready
UNLOCK_EPISODES = (5, 15)   # assumption: free episodes run out around here (the paywall / unlock point)
UNLOCK_TOP = 3
WEAKEST_N = 5


def _failed(a: Optional[EpisodeAnalysis]) -> bool:
    if a is None:
        return True
    tag = f"{a.model} {a.summary}".lower()
    return "analysis failed" in tag or a.model.lower().startswith("failed")


def _beat_info(ep: Episode, beat_id: Optional[str]) -> tuple[str, str]:
    if not beat_id:
        return "", ""
    try:
        b = ep.beat(beat_id)
    except KeyError:
        return "", ""
    text = f"{b.speaker}: {b.text}" if b.kind in ("dialogue", "vo") else b.text
    return tc(b.start), text if len(text) <= 160 else text[:159] + "…"


def episode_report(episodes: list[Episode], analyses: list[EpisodeAnalysis]) -> list[dict]:
    by_num = {a.episode: a for a in analyses if a is not None}
    first_num = min((e.number for e in episodes), default=0)
    rows = []
    for ep in sorted(episodes, key=lambda e: e.number):
        a = by_num.get(ep.number)
        row = {"episode": ep.number, "duration_s": round(ep.duration, 1),
               "opening_hook_score": None, "opening_hook_time": "", "opening_hook_text": "",
               "ending_cliff_score": None, "ending_cliff_time": "", "ending_cliff_text": "",
               "ending_open_question": "", "pickup_status": "", "pickup_delay_s": None,
               "promo_ready_beats": 0, "flags": []}
        if _failed(a):
            row["flags"].append("analysis failed")
            rows.append(row)
            continue
        row["opening_hook_score"] = a.opening_hook.score
        row["opening_hook_time"], row["opening_hook_text"] = _beat_info(ep, a.opening_hook.beat_id)
        row["ending_cliff_score"] = a.ending_cliffhanger.score
        row["ending_cliff_time"], row["ending_cliff_text"] = _beat_info(ep, a.ending_cliffhanger.beat_id)
        row["ending_open_question"] = a.ending_open_question
        row["promo_ready_beats"] = sum(1 for s in a.beats if max(s.hook, s.cliff) >= STRONG_BEAT)

        p = a.pickup
        delay = p.delay_s
        if delay is None and p.first_new_beat_id:
            try:
                delay = ep.beat(p.first_new_beat_id).start
            except KeyError:
                delay = None
        first = ep.number == first_num
        row["pickup_status"] = "not_applicable" if first else p.status
        row["pickup_delay_s"] = None if first or delay is None else round(delay, 1)

        if a.opening_hook.score < WEAK_SCORE:
            row["flags"].append(f"weak opening (<{WEAK_SCORE})")
        if a.ending_cliffhanger.score < WEAK_SCORE:
            row["flags"].append(f"weak ending (<{WEAK_SCORE})")
        if not first:
            if p.status == "unrelated" or (row["pickup_delay_s"] or 0) > PICKUP_LIMIT_S:
                row["flags"].append(f"previous cliffhanger not picked up in first {PICKUP_LIMIT_S:.0f} s")
            if p.status == "recap_then_advance" and (row["pickup_delay_s"] or 0) > RECAP_LIMIT_S:
                row["flags"].append(f"recap longer than {RECAP_LIMIT_S:.0f} s")
        rows.append(row)
    return rows


def season_summary(rows: list[dict]) -> dict:
    """Season-level view for the script editor, with the business hook: the strongest cliffhangers around
    the point where free episodes run out are the natural unlock (paywall) episodes."""
    ok = [r for r in rows if r.get("opening_hook_score") is not None]

    def avg(key: str) -> Optional[float]:
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def weakest(key: str) -> list[dict]:
        return [{"episode": r["episode"], "score": r[key]}
                for r in sorted(ok, key=lambda r: (r[key], r["episode"]))[:WEAKEST_N]]

    lo, hi = UNLOCK_EPISODES
    window = [r for r in ok if lo <= r["episode"] <= hi]
    unlock = sorted(window, key=lambda r: (-r["ending_cliff_score"], r["episode"]))[:UNLOCK_TOP]
    flag_counts: dict[str, int] = {}
    for r in rows:
        for f in r.get("flags", []):
            flag_counts[f] = flag_counts.get(f, 0) + 1
    return {
        "episodes": len(rows), "analyzed": len(ok),
        "avg_opening_hook": avg("opening_hook_score"), "avg_ending_cliff": avg("ending_cliff_score"),
        "weakest_openings": weakest("opening_hook_score"), "weakest_endings": weakest("ending_cliff_score"),
        "unlock_candidates": [{"episode": r["episode"], "score": r["ending_cliff_score"],
                               "open_question": r["ending_open_question"]} for r in unlock],
        "unlock_assumption": (f"Assumes free episodes run out around episodes {lo}-{hi} (typical microdrama "
                              "paywall); the strongest cliffhangers there are candidate unlock points. Kuku earns "
                              "almost all revenue from subscriptions, so this ending sells the subscription."),
        "flag_counts": flag_counts,
    }
