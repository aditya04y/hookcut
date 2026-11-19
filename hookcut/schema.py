"""Data contracts shared by every HookCut stage.

All times are seconds (float) from the start of the episode. Beat ids look like
"E07.B042" (episode 7, beat 42 in time order) and are stable for a given script.
Workers: change these models only through the lead (see docs/process/team_board.md).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

BeatKind = Literal["camera", "action", "dialogue", "vo", "sfx", "other"]
HookType = Literal[
    "violence",       # slap, shove, choke, weapon
    "reveal",         # identity / secret / pregnancy / betrayal revealed
    "shocking_line",  # a line that stops you cold
    "humiliation",    # public shaming, rejection, being thrown out
    "threat",         # danger, ultimatum, blackmail
    "romance",        # kiss, charged closeness, jealousy
    "mystery",        # unexplained arrival, hidden object, who-is-that
    "other",
]


# ---------------------------------------------------------------- parsed script
class Beat(BaseModel):
    id: str
    episode: int
    scene: int
    kind: BeatKind
    speaker: Optional[str] = None   # dialogue / vo only, e.g. "LADY MARA"
    delivery: Optional[str] = None  # parenthetical, e.g. "Cold, Commanding"
    text: str
    start: float
    end: float


class Scene(BaseModel):
    number: int
    heading: str = ""
    start: float
    end: float


class Episode(BaseModel):
    number: int
    source_file: str
    source_id: Optional[str] = None
    logline: str = ""
    continuity_notes: str = ""
    scenes: list[Scene]
    beats: list[Beat]
    duration: float
    warnings: list[str] = []

    def beat(self, beat_id: str) -> Beat:
        for b in self.beats:
            if b.id == beat_id:
                return b
        raise KeyError(beat_id)


# ---------------------------------------------------------- LLM episode analysis
class BeatScore(BaseModel):
    """Scores for one notable beat. Beats the model leaves out count as 0/0."""
    beat_id: str
    hook: int = Field(0, ge=0, le=5)    # stop-the-scroll power if this beat is in a clip's first 3 s
    hook_type: Optional[HookType] = None
    cliff: int = Field(0, ge=0, le=5)   # how badly a viewer needs more if we cut right after it
    open_question: Optional[str] = None  # the question left hanging if we cut here
    resolved_by: Optional[str] = None    # beat id in THIS episode that answers it, if any
    needs_context: list[str] = []        # what a newcomer must already know to follow it
    spoiler: int = Field(0, ge=0, le=3)  # 0 none .. 3 gives away a major twist
    reason: str = ""


class EpisodeEdge(BaseModel):
    """The episode's own opening hook or ending cliffhanger (script-editor view)."""
    beat_id: Optional[str] = None
    score: int = Field(0, ge=0, le=5)
    note: str = ""


class Pickup(BaseModel):
    """How this episode's opening picks up the previous episode's cliffhanger."""
    status: Literal["immediate", "recap_then_advance", "delayed", "unrelated", "not_applicable"]
    first_new_beat_id: Optional[str] = None  # first beat that moves the story past the recap
    delay_s: Optional[float] = None          # filled by code from first_new_beat_id
    note: str = ""


class Character(BaseModel):
    name: str
    aliases: list[str] = []
    role: str = ""


class EpisodeAnalysis(BaseModel):
    episode: int
    summary: str
    characters: list[Character] = []
    beats: list[BeatScore]
    opening_hook: EpisodeEdge
    ending_cliffhanger: EpisodeEdge
    ending_open_question: str = ""
    pickup: Pickup
    model: str = ""
    prompt_version: str = ""
    warnings: list[str] = []      # what code corrected in the model's reply (wrong ids, out-of-window edges, ...)


# ------------------------------------------------------------ promo candidates
class Segment(BaseModel):
    episode: int
    start: float
    end: float
    first_beat_id: str
    last_beat_id: str


class Candidate(BaseModel):
    id: str                                  # "C0001"
    kind: Literal["single", "multi"]         # one continuous stretch, or joined moments
    segments: list[Segment]
    duration: float                          # sum of segment lengths
    hook_beat_id: str
    cliff_beat_id: str
    hook_score: float
    cliff_score: float
    hook_type: Optional[HookType] = None
    open_question: str = ""
    spoiler: int = 0
    needs_context: list[str] = []
    prelim_score: float = 0.0
    notes: list[str] = []


class ColdViewerVerdict(BaseModel):
    """A model that sees ONLY the clip, like a stranger scrolling past."""
    candidate_id: str
    premise: str                    # what the stranger thinks is going on
    conflict: str
    open_question: str              # what they would want answered next
    confusions: list[str] = []      # "who is the man with the cane?"
    hook_lands: bool                # would the first 3 seconds stop them?
    standalone: int = Field(ge=0, le=5)
    keep_watching: int = Field(ge=0, le=5)
    verdict: Literal["pass", "borderline", "fail"]
    model: str = ""


class Promo(BaseModel):
    rank: int
    candidate_id: str
    kind: Literal["single", "multi"]
    segments: list[Segment]
    duration: float
    hook_beat_id: str = ""
    cliff_beat_id: str = ""
    hook_time: str                  # "E01 00:39.10"
    hook_text: str
    cliffhanger_time: str
    cliffhanger_text: str
    open_question: str
    why_it_works: str
    post_wave: Literal[1, 2, 3]     # 1 cold audiences, 2 escalation, 3 big reveals / retargeting
    flags: list[str] = []
    scores: dict[str, float] = {}
    cold_viewer: Optional[ColdViewerVerdict] = None
    variants: list[dict] = []       # alternative openings / length tiers of the same moment:
                                    # {"label", "segments": [Segment-like dicts], "duration", "hook_beat_id", "note"}


# ------------------------------------------------------------------- run output
class RunResult(BaseModel):
    show: str
    created_at: str
    config: dict = {}
    episodes: list[Episode]
    analyses: list[EpisodeAnalysis] = []
    candidates: list[Candidate] = []
    verdicts: list[ColdViewerVerdict] = []
    promos: list[Promo] = []
    episode_report: list[dict] = []
    run_log: list[dict] = []
