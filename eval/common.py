"""Small helpers shared by the eval scripts: loading runs, cut checks, Candidate -> Promo, model families."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from hookcut.schema import Beat, Candidate, Episode, Promo, RunResult
from hookcut.textfmt import ep_tc

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
EPS = 0.05   # timestamps in the scripts have 0.1 s resolution; anything inside 50 ms counts as "on" a boundary


# ------------------------------------------------------------------ runs
def resolve_run(path: str | Path) -> Path:
    """Accept outputs/<run>/result.json or the folder outputs/<run>."""
    p = Path(path)
    return p / "result.json" if p.is_dir() else p


def load_run(path: str | Path) -> RunResult:
    return RunResult.model_validate_json(resolve_run(path).read_text())


def run_name(path: str | Path) -> str:
    return resolve_run(path).parent.name


def results_dir(name: str, base: Optional[Path] = None) -> Path:
    d = (base or RESULTS) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def episode_map(episodes: Iterable[Episode]) -> dict[int, Episode]:
    return {e.number: e for e in episodes}


# ------------------------------------------------------------------ cuts
def straddles(b: Beat, t: float) -> bool:
    """True if a cut at time t would land inside beat b."""
    return b.start < t - EPS and b.end > t + EPS


def is_safe_cut(ep: Episode, t: float) -> bool:
    """A cut is safe if it does not split any dialogue / vo / action / sfx beat. Camera beats are shot
    directions running on their own track; cutting inside one is fine."""
    return not any(straddles(b, t) for b in ep.beats if b.kind != "camera")


SPEECH = ("dialogue", "vo")
STORY = ("dialogue", "vo", "action", "sfx", "other")   # everything except camera directions


def cut_beats(ep: Episode, start: float, end: float, kinds: tuple[str, ...] = SPEECH) -> list[str]:
    """Ids of beats (spoken lines by default) that a clip [start, end] cuts in half. Camera, dialogue and
    action run on overlapping tracks in these scripts, so a cut on a shot boundary can still chop a line."""
    return [b.id for b in ep.beats if b.kind in kinds and (straddles(b, start) or straddles(b, end))]


def beats_in(ep: Episode, start: float, end: float) -> list[Beat]:
    return [b for b in ep.beats if b.start >= start - EPS and b.end <= end + EPS]


def candidate_cut_beats(c: Candidate, eps: dict[int, Episode], kinds: tuple[str, ...] = SPEECH) -> list[str]:
    out: list[str] = []
    for s in c.segments:
        if s.episode in eps:
            out += cut_beats(eps[s.episode], s.start, s.end, kinds)
    return out


# ------------------------------------------------------------------ candidate -> promo
def _beat(eps: dict[int, Episode], beat_id: str) -> Optional[Beat]:
    try:
        return eps[int(beat_id[1:3])].beat(beat_id)
    except (KeyError, ValueError, IndexError):
        return None


def as_promo(c: Candidate, eps: dict[int, Episode], rank: int = 1) -> Promo:
    """Wrap a Candidate as a Promo so `validate_promos` can check baselines the same way as the tool's picks."""
    hook, cliff = _beat(eps, c.hook_beat_id), _beat(eps, c.cliff_beat_id)
    return Promo(
        rank=rank, candidate_id=c.id, kind=c.kind, segments=c.segments, duration=c.duration,
        hook_time=ep_tc(hook.episode, hook.start) if hook else "", hook_text=hook.text if hook else "",
        cliffhanger_time=ep_tc(cliff.episode, cliff.start) if cliff else "",   # beat start, as validate expects
        cliffhanger_text=cliff.text if cliff else "", open_question=c.open_question,
        why_it_works="(baseline)", post_wave=1, flags=list(c.notes))


# ------------------------------------------------------------------ model families
_FAMILIES = (("gpt-oss", "openai"), ("gpt-", "openai"), ("o4", "openai"), ("qwen", "qwen"), ("gemini", "google"),
             ("gemma", "google"), ("claude", "anthropic"), ("llama", "meta"), ("mistral", "mistral"),
             ("mixtral", "mistral"), ("deepseek", "deepseek"), ("nemotron", "nvidia"), ("kimi", "moonshot"))


def family(spec: str) -> str:
    """'groq:openai/gpt-oss-120b' -> 'openai'; 'gemini:gemini-3.8-flash' -> 'google'. Unknown -> the model name."""
    s = (spec or "").lower()
    model = s.split(":", 1)[-1]
    for needle, fam in _FAMILIES:
        if needle in model:
            return fam
    if s.startswith("anthropic:"):
        return "anthropic"
    return model or "?"


def spec_families(spec: str) -> set[str]:
    """A FallbackClient spec 'a -> b -> c' may have used any of its models."""
    return {family(p.strip()) for p in (spec or "").split("->") if p.strip()}
