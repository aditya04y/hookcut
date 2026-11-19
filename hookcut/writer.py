"""Stage 6: one LLM call writes the one-line "why it works" for every ranked promo.

The model sees only facts already chosen upstream (hook line, cliffhanger line, open question, what the cold
viewer understood), so it cannot invent plot. If the call fails, a plain template line is used instead.
"""
from __future__ import annotations

import json

from .analyze import load_prompt
from .llm.base import LLMClient
from .runlog import RunLog
from .schema import Episode, Promo

MAX_WORDS = 25


def _facts(p: Promo) -> dict:
    return {"id": p.candidate_id,
            "hook": f"{p.hook_time} {p.hook_text}",
            "cliffhanger": f"{p.cliffhanger_time} {p.cliffhanger_text}",
            "open_question": p.open_question,
            "cold_viewer_premise": p.cold_viewer.premise if p.cold_viewer else ""}


def template_line(p: Promo) -> str:
    """Deterministic fallback, grounded in the same facts."""
    hook = " ".join(p.hook_text.split()[:8])
    q = p.open_question or "what happens next"
    line = f'Opens on "{hook}" and cuts before the answer to: {q}'
    words = line.split()
    return " ".join(words[:MAX_WORDS]) + ("…" if len(words) > MAX_WORDS else "")


def _check(raw, ids: list[str]) -> dict[str, str]:
    lines = raw.get("lines") if isinstance(raw, dict) else None
    if not isinstance(lines, list):
        raise ValueError('reply must be {"lines": [{"id": ..., "why": ...}, ...]}')
    got = {str(x.get("id")): str(x.get("why") or "").strip() for x in lines if isinstance(x, dict)}
    missing = [i for i in ids if not got.get(i)]
    long = [i for i in ids if got.get(i) and len(got[i].split()) > MAX_WORDS]
    problems = []
    if missing:
        problems.append(f"missing or empty lines for ids: {', '.join(missing)}")
    if long:
        problems.append(f"lines over {MAX_WORDS} words for ids: {', '.join(long)}")
    if problems:
        raise ValueError("; ".join(problems))
    return {i: got[i] for i in ids}


def write_why_lines(promos: list[Promo], episodes: list[Episode], client: LLMClient, *, log: RunLog) -> list[Promo]:
    """`episodes` is part of the stage contract; the facts we need are already on each Promo."""
    if not promos:
        return []
    system, version = load_prompt("writer_system")
    ids = [p.candidate_id for p in promos]
    user = "PROMOS:\n" + "\n".join(json.dumps(_facts(p), ensure_ascii=False) for p in promos)
    try:
        res = client.complete_json(system, user, validate=lambda d: _check(d, ids), cache_tag=version)
        lines = _check(res.data, ids)
        log.record("writer", "all", res)
    except Exception as e:
        log.record("writer", "all", error=f"{type(e).__name__}: {e} (template lines used)")
        lines = {p.candidate_id: template_line(p) for p in promos}
    return [p.model_copy(update={"why_it_works": lines[p.candidate_id]}) for p in promos]
