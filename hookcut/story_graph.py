"""Content genome, show level: who is in the story, how they are bound, which questions keep viewers watching.

Built from a saved run (RunResult), in two layers that end in the same story_graph.json:
1. Deterministic, code only. Entity resolution merges character names across episodes (honorifics stripped,
   aliases from the per-episode analyses, a gender guard against wrong alias claims). Characters are then
   projected through the scenes they share (character <-> scene bipartite graph -> character <-> character edges
   with co-presence, dominance and hostility statistics and an additive conflict score), profiled (lines, screen
   time, "promo pull"), and checked for story alerts (relationship reversals, identity reveals, long open loops).
   Open loops come from scored beats (open_question / resolved_by) and each episode's ending question.
2. Optional show-level LLM pass: ONE call over compact per-episode digests (batches of ~13 episodes when the prompt
   would pass ~25K tokens) merges aliases, assigns archetypes, labels relationships and their turns, reports
   identity reveals, links each ending question to the episode that answers it, and tags tropes/emotions from
   fixed lists. Code checks every id and episode number (repairs or drops with a warning); if the call fails, the
   deterministic layer stands.

The relationship layer follows the author's tender-intelligence
bidder graph: contractor <-> tender projection becomes character <-> scene, the Jaccard co-bidding
index becomes co-presence, win asymmetry becomes dominance (who delivers the strong lines), price spreads become the
hostility/warmth ratio, the syndicate risk score becomes conflict_intensity (core conflict pairs), bidder profiles
become character profiles, forensic alerts become story alerts, and the name resolver becomes _Cast.

CLI: python -m hookcut.story_graph --run outputs/run_full --out outputs/run_full [--model gemini:...] [--no-llm]
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Literal, Optional, get_args

from pydantic import BaseModel

from .llm.base import LLMClient
from .runlog import RunLog
from .schema import EpisodeAnalysis, HookType, RunResult

PROMPT_FILE = Path(__file__).parent / "prompts" / "story_graph_system.md"
DEFAULT_MODEL = "gemini:gemini-3.5-flash-lite"
MAX_PROMPT_TOKENS = 25_000   # above this (chars / 4) the show goes to the model in batches
BATCH_EPISODES = 13
STRONG = 4                   # a hook or cliff score of 4+ is a "strong" beat
MIN_SHARED_SCENES = 2        # co-presence edges below this are dropped unless the story names the bond
CORE_CONFLICT_MIN = 55.0     # conflict_intensity at or above this marks a core conflict pair
TOP_TROPES = 8               # trope_* feature columns: the show's most frequent tropes

LoopKind = Literal["identity", "romance", "revenge", "danger", "mystery", "status", "justice"]
RelKind = Literal["family", "romance", "rivalry", "alliance", "power", ""]
AlertType = Literal["relationship_reversal", "identity_reveal", "loop_opened", "loop_closed"]
LOOP_KINDS = get_args(LoopKind)
REL_KINDS = tuple(k for k in get_args(RelKind) if k)
HOOK_TYPES = get_args(HookType)
PICKUPS = ("immediate", "recap_then_advance", "delayed", "unrelated", "not_applicable")
ARCHETYPES = ("wronged heroine", "hidden-identity protector", "romantic lead", "scheming sister", "scheming rival",
              "rival lover", "betrayer", "complicit parent", "cruel matriarch", "villain", "loyal ally",
              "authority figure", "mentor", "other")
EMOTIONS = ("shock", "humiliation", "anger", "fear", "grief", "love", "jealousy", "hope", "triumph", "suspense")
TROPES = {  # trope -> cue in the analysis text (summaries, reasons, questions); the LLM pass replaces these tags
    "hidden identity": r"disguis\w*|hidden identity|true identity|real identity|unmask\w*|impost[eo]r\w*"
                       r"|secret(?:ly)? (?:the )?(?:duke|heir|prince|king|lord)",
    "betrayal at the altar": r"altar|jilt\w*|former groom|groom (?:leaves|chooses|marries)|wedding (?:is )?called off",
    "public humiliation": r"humiliat\w*|publicly (?:sham|mock|disgrac)\w*|mock\w*|ridicul\w*",
    "contract marriage": r"contract marriage|marriage (?:contract|of convenience)|arranged marriage|fake marriage"
                         r"|marr\w* (?:a|the) (?:beggar|stranger)",
    "forced marriage": r"forced (?:to|into) marr\w*|forced marriage|married off|must marry",
    "false accusation": r"accus\w*|framed|wrongly|witch\w*|false charge",
    "rescue": r"rescu\w*|saves? (?:her|him|them)|comes to (?:her|his) aid|intervenes",
    "revenge": r"reveng\w*|vengeance|turns? the tables|payback|retribution",
    "secret pregnancy": r"pregnan\w*|with child",
    "class divide": r"beggar\w*|peasant\w*|commoner\w*|lowly|penniless|beneath (?:her|him|them)",
    "evil stepfamily": r"step-?(?:mother|sister|father|family|daughter)\w*",
    "jealous rival": r"jealous\w*|envious|envy|rival\w*",
    "kidnapping": r"kidnap\w*|abduct\w*|held captive|hostage\w*|ransom",
    "poisoning": r"poison\w*",
    "inheritance battle": r"inherit\w*|succession|rightful (?:heir|duke|lord)|claim to the (?:title|estate|throne)",
    "fake death": r"fak\w* (?:his|her) (?:own )?death|presumed dead|believed (?:to be )?dead|back from the dead",
}
_TROPE_CUES = {name: re.compile(rf"\b(?:{p})", re.I) for name, p in TROPES.items()}
_HOOK_EMOTION = {"humiliation": "humiliation", "threat": "fear", "violence": "anger", "reveal": "shock",
                 "shocking_line": "shock", "romance": "love", "mystery": "suspense"}
_EMOTION_CUES = {name: re.compile(rf"\b(?:{p})", re.I) for name, p in {
    "grief": r"grie\w*|mourn\w*|funeral|tears|sob\w*|weep\w*", "jealousy": r"jealous\w*|envy|envious",
    "anger": r"anger|angry|furious|fury|rage|enraged|outrage\w*", "hope": r"hope\w*|reunit\w*|new life|freedom",
    "triumph": r"triumph\w*|turns? the tables|vindicat\w*|stands up to|defies|victory",
    "love": r"love\w*|kiss\w*|tender\w*|embrace\w*|romanc\w*", "fear": r"fear\w*|terrif\w*|danger\w*|threat\w*",
    "shock": r"shock\w*|stun\w*|revelation", "humiliation": r"humiliat\w*|mock\w*|sham\w*|disgrac\w*"}.items()}
_LOOP_CUES = [(kind, re.compile(rf"\b(?:{p})", re.I)) for kind, p in (
    ("identity", r"who (?:is|was|really)|identity|real name|really (?:is|the)|disguis\w*|recogni[sz]\w*|unmask\w*"),
    ("justice", r"guilty|innocen\w*|trial|accus\w*|prove|arrest\w*|punish\w*|expos\w*|justice|verdict|sentence"),
    ("danger", r"surviv\w*|kill\w*|die|dies|death|escape\w*|rescue\w*|danger\w*|hurt|harm\w*|attack\w*|in time|safe"),
    ("revenge", r"reveng\w*|get back at|pay for|retaliat\w*|turn the tables"),
    ("romance", r"love\w*|marr\w*|kiss\w*|wedding|husband|wife|feel\w*|jealous\w*|heart"),
    ("status", r"title|status|inherit\w*|accept\w*|position|heir|duchess|duke|estate|disown\w*|reputation"))]
_LABEL_KIND = {
    **dict.fromkeys("sister brother sibling stepsister stepbrother half-sister half-brother mother father parent "
                    "stepmother stepfather daughter son stepdaughter stepson aunt uncle cousin grandmother "
                    "grandfather mother-in-law father-in-law sister-in-law brother-in-law".split(), "family"),
    **dict.fromkeys("husband wife spouse fiancé fiancée fiance fiancee betrothed groom bride lover suitor ex "
                    "ex-fiancé ex-fiance beloved partner sweetheart".split(), "romance"),
    **dict.fromkeys("rival enemy nemesis tormentor accuser".split(), "rivalry"),
    **dict.fromkeys("friend ally confidant companion protector savior saviour".split(), "alliance"),
    **dict.fromkeys("maid servant employer master mistress guard captor steward butler".split(), "power"),
}
_LABEL_GROUP = {**dict.fromkeys("husband wife spouse".split(), "spouse"),
                **dict.fromkeys("bride groom betrothed fiancé fiancée fiance fiancee".split(), "betrothed"),
                **dict.fromkeys("sister brother sibling".split(), "sibling"),
                **dict.fromkeys("mother father parent daughter son".split(), "parent")}
_HOSTILE_WORDS = set("enemy rival nemesis betrayer traitor ex estranged accuser captor tormentor jilted former "
                     "adversary blackmailer".split())
_WARM_WORDS = set("husband wife spouse lover ally friend betrothed fiancé fiancée protector beloved savior "
                  "partner".split())
HOSTILE_HOOKS = {"violence", "humiliation", "threat"}
WARM_HOOKS = {"romance"}

# Entity resolution vocabulary (the story version of the bidder resolver's prefix/suffix lists).
_HONORIFICS = {"lady": "f", "miss": "f", "mrs": "f", "madam": "f", "dame": "f", "dowager": "f",
               "lord": "m", "sir": "m", "mr": "m", "master": "m", "the": ""}
_FEMALE = set("lady miss mrs madam dame dowager duchess princess queen woman girl bride sister stepsister mother "
              "stepmother daughter wife maid matriarch widow heroine noblewoman aunt grandmother fiancee fiancée "
              "housekeeper".split())
_MALE = set("lord sir mr master duke prince king man boy groom brother stepbrother father stepfather son husband "
            "nobleman uncle grandfather fiance fiancé".split())
_ROLE_WORDS = set("father mother sister brother son daughter wife husband maid guard guards judge constable priest "
                  "lord lady matriarch housekeeper noblewoman nobleman informant officer captain doctor nurse "
                  "servant butler duke duchess king queen prince princess stranger man woman girl boy official "
                  "thug villager elder groom bride figure veteran beggar soldier healer widow heir mistress master "
                  "child baby crowd voice narrator driver accused sentry footman coachman messenger herald minister "
                  "bishop clerk merchant physician attendant courtier noble aristocrat onlooker".split())
_ADDRESS = {"your grace", "my lady", "my lord", "your majesty", "your highness", "sir", "madam", "miss", "darling",
            "dear", "my dear"}
_STOP = set("in of the with and a an her his my your our their to at on for from by new old young grace majesty "
            "highness dear".split())   # a key with one of these is a description, not a name
# Titles and role words change hands over time (a new holder takes a title; the previous holder was someone else),
# like a company name reused by a new firm in the bidders resolver: an analysis alias made only of these words is a
# time-scoped ROLE (CharacterNode.roles), never an identity alias. Speaker labels and listed names stay identities.
_TITLES = set("duke duchess lord lady king queen prince princess emperor empress earl countess baron baroness marquess "
              "viscount dowager heir heiress grace majesty highness ladyship lordship".split())   # held by one at a time
_ROLE_VOCAB = _ROLE_WORDS | _TITLES | set(_LABEL_KIND) | set("imposter impostor prisoner captive witch".split())


# ------------------------------------------------------------------ models (tables inside story_graph.json)
class RoleSpan(BaseModel):
    """A title or role the analyses gave a character for a while ("duchess", "disgraced bride", "maid")."""
    role: str                           # canonical form of the analysis alias
    first_ep: int
    last_ep: int
    episodes: list[int] = []            # claim episodes where the script backs it up
    evidence_beat: str                  # first beat naming the role in a scene the character is in


class CharacterNode(BaseModel):
    id: str
    name: str
    aliases: list[str] = []
    archetype: str = ""                 # one of ARCHETYPES ("" = unknown)
    role: str = ""                      # most common per-episode role descriptions ("; "-joined, up to 2)
    roles: list[RoleSpan] = []          # titles / role descriptors, time-scoped, only with script evidence
    gender: str = ""                    # f / m / "" from titles and role words (used to reject wrong alias claims)
    first_ep: int = 0
    last_ep: int = 0
    episodes_present: list[int] = []
    # profile (the bidder_profiles analogue)
    lines: int = 0                      # dialogue / VO beats spoken
    screen_time_s: float = 0.0          # union of beat time where they speak or are named in an action/camera line
    scenes: int = 0
    hook_lines: int = 0                 # strong-hook beats (hook 4+) they deliver
    cliff_beats: int = 0                # strong-cliff beats (cliff 4+) they are in
    clips: int = 0                      # candidate clips they appear in
    promos: int = 0                     # final promos they appear in
    promo_pull: float = 0.0             # share of the show's strong beats (hook or cliff 4+) that involve them
    top_costars: list[str] = []         # ids, most shared scenes first


class AliasRow(BaseModel):
    alias: str                          # as written: a speaker label ("LADY MARA") or an analysis name/alias
    canonical: str                      # character id
    first_ep: int
    last_ep: int
    episodes: list[int] = []
    count: int = 0
    sources: list[str] = []             # speaker label / analysis name / analysis alias


class RelationChange(BaseModel):
    ep: int
    to: str
    beat_id: Optional[str] = None


class RelationEdge(BaseModel):
    a: str
    b: str
    kind: RelKind = ""                  # "" = co-presence only, no bond named
    label: str = ""                     # what a is to b ("sister", "betrothed")
    first_ep: int = 0
    changes: list[RelationChange] = []
    basis: str = ""                     # analysis roles / llm / co-presence
    shared_scenes: int = 0
    shared_episodes: int = 0
    jaccard: float = 0.0                # shared scenes / (scenes of a + scenes of b - shared)
    overlap_s: float = 0.0              # sum over shared scenes of min(a's, b's screen time there)
    a_strong: int = 0                   # strong beats in shared scenes delivered by a
    b_strong: int = 0
    asymmetry: float = 0.0              # |a_strong - b_strong| / (a_strong + b_strong)
    dominant: str = ""                  # who delivers more of them
    hostile_beats: int = 0              # scored beats in shared scenes typed violence / humiliation / threat
    warm_beats: int = 0                 # ... typed romance
    hostility: float = 0.0              # hostile / (hostile + warm)
    conflict_intensity: float = 0.0     # 0-100, sum of score_parts
    score_parts: dict[str, float] = {}
    core: bool = False                  # conflict_intensity >= CORE_CONFLICT_MIN


class OpenLoop(BaseModel):
    id: str
    question: str
    kind: LoopKind = "mystery"
    scope: Literal["scene", "episode", "thread"]   # a beat's question / the episode's ending question / a long arc
    opened_ep: int
    opened_beat: Optional[str] = None
    answered_ep: Optional[int] = None
    answered_beat: Optional[str] = None
    status: Literal["answered", "open"] = "open"
    basis: str = ""                     # analysis (resolved_by) / pickup (proxy: next episode picks it up) / llm
    strength: int = 0                   # cliff score (0-5) of the beat that opens it


class StoryAlert(BaseModel):
    id: str
    type: AlertType
    severity: Literal["high", "medium", "low"]
    confidence: float
    ep: int
    beat_ids: list[str] = []
    characters: list[str] = []
    summary: str = ""
    evidence: dict = {}


class EpisodeNode(BaseModel):
    ep: int
    tension: float = 0.0                # 0-1, from beat scores (see FEATURES["tension"])
    emotions: list[str] = []
    tropes: list[str] = []
    characters: list[str] = []
    opening_hook: int = 0
    ending_cliff: int = 0
    pickup: str = ""
    loops_opened: list[str] = []
    loops_closed: list[str] = []
    loops_carried: list[str] = []       # cross-episode loops still open at the episode's end


class StoryGraph(BaseModel):
    show: str
    model: str = "deterministic"
    prompt_version: str = ""
    characters: list[CharacterNode] = []
    aliases: list[AliasRow] = []
    relationships: list[RelationEdge] = []
    alerts: list[StoryAlert] = []
    loops: list[OpenLoop] = []
    episodes: list[EpisodeNode] = []
    show_tropes: list[str] = []
    notes: list[str] = []
    warnings: list[str] = []


# ------------------------------------------------------------------ text helpers
def _key(text) -> str:
    return " ".join(re.sub(r"[^\w\s'-]", " ", str(text or "").replace("’", "'").casefold()).split())


def canonical_name(label) -> tuple[str, str]:
    """'LADY MARA' -> ('mara', 'f'): case and punctuation normalised, leading honorifics stripped."""
    words, gender = _key(label).split(), ""
    while len(words) > 1 and words[0] in _HONORIFICS:
        gender = gender or _HONORIFICS[words.pop(0)]
    return " ".join(words), gender


def _gender(*texts: str) -> str:
    words = [w for t in texts for w in re.findall(r"[a-zé]+", _key(t))]
    f, m = sum(w in _FEMALE for w in words), sum(w in _MALE for w in words)
    return "f" if f > m else "m" if m > f else ""


def _compatible(g1: str, g2: str) -> bool:
    return not g1 or not g2 or g1 == g2


def _descriptor(key: str) -> bool:
    """'mysterious hooded man', 'old countess', 'father abel': a role, not a proper name."""
    return key in _ADDRESS or any(w in _ROLE_WORDS for w in key.split())


def _role_alias(key: str, labels: set[str], name_words: set[str]) -> bool:
    """An analysis alias that is a role, not a name: a bare role word ('maid', 'husband'), any phrase with a title
    ('duchess of westmere', 'her grace', 'dead duke'), or a description nobody in the show is labelled by
    ('disgraced bride', 'elder woman'). Never a role: an alias that names someone ('hattie the young maid'), or a
    title-free description the show itself uses as a speaker label or listed name ('the hooded stranger',
    'the former groom', 'cart driver'): those are disguises and hidden identities. Words after 'of' name a house or
    place ('duchess of vale'), not the person, so they do not count as a name."""
    words = [w.removesuffix("'s") for w in key.split()]
    before_of = [w.removesuffix("'s") for w in key.split(" of ")[0].split()]
    if not words or any(w in name_words for w in before_of) or not any(w in _ROLE_VOCAB for w in words):
        return False
    return len(words) == 1 or any(w in _TITLES for w in words) or key not in labels


def _role_head(role: str) -> str:
    """The word the script would use for the role: 'duchess of westmere' -> duchess, "beggar's wife" -> wife."""
    words = [w.removesuffix("'s") for w in role.split(" of ")[0].split()]
    return next((w for w in reversed(words) if w in _ROLE_VOCAB), words[-1] if words else role)


def _slug(text) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _key(text)).strip("-")


def _unique(base: str, taken: set[str]) -> str:
    base, n = base or "character", 2
    out = base
    while out in taken:
        out, n = f"{base}-{n}", n + 1
    taken.add(out)
    return out


def _fix_beat(x, ep: int, beat_ids: set[str]) -> tuple[Optional[str], str]:
    """Accept 'E04.B010', 'B10' or '10' for a beat of episode `ep`; return (id or None, warning)."""
    if x is None or str(x).strip().lower() in ("", "null", "none"):
        return None, ""
    s = str(x).strip()
    m = re.fullmatch(r"(?:E(\d+)\.?)?B?(\d+)", s, re.I)
    if m and (m.group(1) is None or int(m.group(1)) == ep):
        fixed = f"E{ep:02d}.B{int(m.group(2)):03d}"
        if fixed in beat_ids:
            return fixed, ("" if fixed == s else f"repaired beat id {s!r} -> {fixed}")
    return None, f"dropped beat id {s!r} (not a beat of E{ep:02d})"


def _usable(analyses: list[EpisodeAnalysis]) -> list[EpisodeAnalysis]:
    return [a for a in analyses if not a.summary.startswith("analysis failed")]


def _loop_kind(text: str) -> str:
    return next((kind for kind, cue in _LOOP_CUES if cue.search(text or "")), "mystery")


def _norm(x) -> str:
    return " ".join(str(x or "").strip().lower().replace("_", " ").split())


# ------------------------------------------------------------------ entity resolution
class _Cast:
    """Per-episode character lists and speaker labels -> canonical characters (a union-find over ids).

    1. Analysis names cluster on their canonical form ("Lady Mara" = "Mara") when titles do not contradict
       each other's gender and the two were not listed side by side in one episode; a full name joins its unique
       given name ("Mara Vale" -> "Mara"), never a role ("Father Abel" is not "Father").
    2. An alias that is another cluster's name merges the two when exactly one character claims it ("THE FORMER
       GROOM" -> Edmund). If that alias was used only in episodes before the claimant is first named, the merge is
       an identity reveal, not a plain alias. Alias claims that contradict a character's gender are dropped.
    3. Speaker labels resolve per episode (that episode's list first, then names, then alias votes); a label nobody
       claims becomes its own speaker-only character, which the LLM pass may merge later.
    4. A title or role alias ('Duchess', 'disgraced bride', 'the maid'; see _role_alias) is not an identity: it
       still resolves that episode's speaker labels, but never votes across episodes, never merges characters and
       never enters the alias table. It becomes a time-scoped role (CharacterNode.roles) when the script backs it.
    """

    def __init__(self, result: RunResult):
        self.parent: dict[str, str] = {}
        self.info: dict[str, dict] = {}
        self.claims: dict[tuple[int, str], Counter] = defaultdict(Counter)   # (ep, key) -> votes (that ep's list)
        self.by_name: dict[str, set[str]] = defaultdict(set)                  # key -> ids named so
        self.by_alias: dict[str, Counter] = defaultdict(Counter)              # key -> votes across episodes
        self.labels: dict[tuple[int, str], str] = {}                          # (ep, speaker label) -> id
        self.mentions: list[tuple[int, str, str, str]] = []                   # (ep, label, id, source) for the alias table
        self.role_claims: list[tuple[int, str, str]] = []                     # (ep, id, alias): titles / roles, see _role_alias
        self.reveals: list[dict] = []
        self.dropped: list[tuple[int, str, str]] = []                         # gender-contradicting alias claims
        self._cluster(result)
        self._alias_merges()
        for ep in result.episodes:
            for label in dict.fromkeys(b.speaker for b in ep.beats if b.speaker):
                self.labels[(ep.number, label)] = self.resolve(ep.number, label) or self._speaker_node(label)

    # union-find
    def root(self, cid: str) -> str:
        while self.parent[cid] != cid:
            self.parent[cid] = self.parent[self.parent[cid]]
            cid = self.parent[cid]
        return cid

    def roots(self) -> list[str]:
        return sorted({self.root(c) for c in self.parent})

    def members(self, root: str) -> list[str]:
        return [c for c in self.parent if self.root(c) == root]

    def merged(self, root: str, field_: str):
        vals = [self.info[c][field_] for c in self.members(root)]
        return sum(vals, Counter()) if isinstance(vals[0], Counter) else set().union(*vals)

    def gender(self, root: str) -> str:
        g = Counter(self.info[c]["gender"] for c in self.members(root) if self.info[c]["gender"])
        return g.most_common(1)[0][0] if len(g) == 1 else ""

    def apart(self, a: str, b: str) -> Optional[str]:
        """Why two characters must stay separate, or None."""
        shared = self.merged(self.root(a), "listed") & self.merged(self.root(b), "listed")
        if len(shared) >= 2:   # one shared episode can be the reveal episode itself
            return f"listed as different people in {len(shared)} episodes"
        if not _compatible(self.gender(self.root(a)), self.gender(self.root(b))):
            return "titles or roles give different genders"
        return None

    def merge(self, keep: str, gone: str) -> None:
        ka, kb = self.root(keep), self.root(gone)
        if ka != kb:
            self.parent[kb] = ka

    def _add(self, cid: str, names: Counter, roles: Counter, listed: set[int], gender: str, proper: bool) -> None:
        self.parent[cid] = cid
        self.info[cid] = {"names": names, "aliases": Counter(), "roles": roles, "listed": set(listed),
                          "gender": gender, "proper": proper}

    def _cluster(self, result: RunResult) -> None:
        clusters, index = [], defaultdict(list)
        for a in _usable(result.analyses):
            for ch in a.characters:
                key, g = canonical_name(ch.name)
                if not key or key in _ADDRESS:
                    continue
                g = g or _gender(ch.name, ch.role)
                home = next((c for c in index[key] if a.episode not in c["eps"] and _compatible(c["g"], g)), None)
                if home is None:
                    home = {"keys": {key}, "eps": set(), "g": g, "entries": [], "alive": True}
                    clusters.append(home)
                    index[key].append(home)
                home["entries"].append((a.episode, ch))
                home["eps"].add(a.episode)
                home["g"] = home["g"] or g
        for c in clusters:       # a full name joins its unique given name
            key = sorted(c["keys"])[0]
            if len(key.split()) < 2 or _descriptor(key):
                continue
            targets = [t for t in index.get(key.split()[0], []) if t is not c and t["alive"]
                       and not t["eps"] & c["eps"] and _compatible(t["g"], c["g"])]
            if len(targets) == 1:
                t = targets[0]
                t["entries"] += c["entries"]
                t["eps"] |= c["eps"]
                t["keys"] |= c["keys"]
                t["g"] = t["g"] or c["g"]
                c["alive"] = False
        # what the show uses as someone's label (speaker labels, listed names) and the words of its proper names
        labels = {canonical_name(b.speaker)[0] for e in result.episodes for b in e.beats if b.speaker}
        labels |= {canonical_name(ch.name)[0] for c in clusters for _, ch in c["entries"]}
        name_words = {w for k in labels if not _descriptor(k) for w in k.split() if w not in _STOP}
        taken: set[str] = set()
        for c in (c for c in clusters if c["alive"]):
            names = Counter(ch.name.strip() for _, ch in c["entries"])
            proper = [n for n, _ in names.most_common() if not _descriptor(canonical_name(n)[0])]
            cid = _unique(_slug(proper[0] if proper else names.most_common(1)[0][0]), taken)
            roles = Counter(ch.role.strip() for _, ch in c["entries"] if ch.role.strip())
            self._add(cid, names, roles, c["eps"], _gender(*names.elements(), *roles.elements()), bool(proper))
            own = {canonical_name(n)[0] for n in names}     # "CONSTABLE" for the Constable is the name, not a role
            for ep, ch in c["entries"]:
                key = canonical_name(ch.name)[0]
                self.claims[(ep, key)][cid] += 1
                self.by_name[key].add(cid)
                if len(key.split()) > 1 and not _descriptor(key):
                    self.by_name[key.split()[0]].add(cid)
                self.mentions.append((ep, ch.name.strip(), cid, "analysis name"))
                for alias in ch.aliases:
                    akey, ag = canonical_name(alias)
                    if not akey or akey in _ADDRESS:
                        continue
                    if not _compatible(ag or _gender(alias), self.info[cid]["gender"]):
                        self.dropped.append((ep, cid, alias))
                        continue
                    self.info[cid]["aliases"][alias.strip()] += 1    # what the analyses said (the LLM digest shows it)
                    self.claims[(ep, akey)][cid] += 1                # resolves THIS episode's speaker labels only
                    if akey not in own and _role_alias(akey, labels, name_words):
                        self.role_claims.append((ep, cid, alias.strip()))
                        continue
                    self.mentions.append((ep, alias.strip(), cid, "analysis alias"))
                    if akey not in _ROLE_WORDS:
                        self.by_alias[akey][cid] += 1
        self._taken = taken

    def _alias_merges(self) -> None:
        for key, votes in list(self.by_alias.items()):
            owners = {self.root(c) for c in self.by_name.get(key, ())}
            claimants = {self.root(c) for c in votes} - owners
            if len(owners) != 1 or len(claimants) != 1:
                continue
            (owner,), (claimant,) = owners, claimants
            if self.apart(owner, claimant):
                continue
            keep, gone = ((owner, claimant) if self.info[owner]["proper"] and not self.info[claimant]["proper"]
                          else (claimant, owner))
            named = [min(self.info[c]["listed"]) for c in self.members(keep)
                     if self.info[c]["proper"] and self.info[c]["listed"]]
            before = self.merged(gone, "listed")
            if named and before and max(before) < min(named):   # known only by a descriptor until then
                self.reveals.append({"character": keep, "was": gone, "ep": min(named), "basis": "analysis alias"})
            self.merge(keep, gone)

    def resolve(self, ep: int, label: str) -> Optional[str]:
        key, g = canonical_name(label)
        g = g or _gender(label)          # "LORD HALE" never resolves to Lady Hale
        for votes in (self.claims.get((ep, key)), Counter(dict.fromkeys(self.by_name.get(key, ()), 1)),
                      self.by_alias.get(key)):
            roots = Counter()
            for c, n in (votes or {}).items():
                if _compatible(g, self.gender(self.root(c))):
                    roots[self.root(c)] += n
            if roots:
                top = roots.most_common(2)
                if len(top) == 1 or top[1][1] < top[0][1]:
                    return top[0][0]
        return None

    def _speaker_node(self, label: str) -> str:
        key, g = canonical_name(label)
        cid = _unique(_slug(key or label), self._taken)
        self._add(cid, Counter({label.strip().title(): 1}), Counter(), set(), g or _gender(label), not _descriptor(key))
        self.by_name[key].add(cid)
        return cid


# ------------------------------------------------------------------ the deterministic draft
@dataclass
class _Draft:
    result: RunResult
    cast: _Cast
    loops: list[OpenLoop]
    bonds: list[dict]                       # {a, b, ep, label, kind, basis, change, beat_id}
    semantics: dict[int, dict]              # ep -> {tropes, emotions}
    names: dict[str, str] = field(default_factory=dict)        # LLM overrides, by id
    archetypes: dict[str, str] = field(default_factory=dict)
    reveals: list[dict] = field(default_factory=list)          # LLM-reported reveals
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str = "deterministic"
    version: str = ""

    @property
    def episode_numbers(self) -> list[int]:
        return sorted(e.number for e in self.result.episodes)

    @property
    def beat_ids(self) -> set[str]:
        return {b.id for e in self.result.episodes for b in e.beats}


_POSSESSIVE = re.compile(r"\b(?P<name>[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2})'s\s+(?:[\w-]+,?\s+){0,3}?(?P<label>(?i:"
                         + "|".join(sorted(map(re.escape, _LABEL_KIND), key=len, reverse=True)) + r"))\b")


def _role_bonds(result: RunResult, cast: _Cast) -> list[dict]:
    """'Mara's spiteful sister' in Ivy's role -> ivy is mara's sister (first stated in that episode)."""
    out = []
    for ep, label, cid, source in cast.mentions:
        if source != "analysis name":
            continue
        a = next((x for x in _usable(result.analyses) if x.episode == ep), None)
        for ch in (c for c in a.characters if c.name.strip() == label):
            for m in _POSSESSIVE.finditer(ch.role.replace("’", "'")):
                other = cast.resolve(ep, m["name"])
                if other and cast.root(other) != cast.root(cid):
                    word = m["label"].lower()
                    out.append({"a": cid, "b": other, "ep": ep, "label": word, "kind": _LABEL_KIND[word],
                                "basis": "analysis roles", "change": False, "beat_id": None})
    return out


def _loops(result: RunResult) -> list[OpenLoop]:
    analyses = sorted(_usable(result.analyses), key=lambda a: a.episode)
    nxt = {a.episode: b for a, b in zip(analyses, analyses[1:])}
    out = []
    for a in analyses:
        for s in a.beats:
            if not s.open_question or (s.beat_id == a.ending_cliffhanger.beat_id and not s.resolved_by):
                continue   # the ending question covers the ending beat
            out.append(OpenLoop(id=f"L-{s.beat_id}", question=s.open_question, kind=_loop_kind(s.open_question),
                                scope="scene", opened_ep=a.episode, opened_beat=s.beat_id, strength=s.cliff,
                                answered_ep=a.episode if s.resolved_by else None, answered_beat=s.resolved_by,
                                status="answered" if s.resolved_by else "open",
                                basis="analysis" if s.resolved_by else ""))
        if a.ending_open_question.strip():
            loop = OpenLoop(id=f"L-E{a.episode:02d}-end", question=a.ending_open_question.strip(),
                            kind=_loop_kind(a.ending_open_question), scope="episode", opened_ep=a.episode,
                            opened_beat=a.ending_cliffhanger.beat_id, strength=a.ending_cliffhanger.score)
            b = nxt.get(a.episode)
            if b and b.pickup.status in ("immediate", "recap_then_advance", "delayed"):
                loop.answered_ep, loop.answered_beat = b.episode, b.pickup.first_new_beat_id
                loop.status, loop.basis = "answered", "pickup"
            out.append(loop)
    return out


def _analysis_text(a: EpisodeAnalysis) -> str:
    return " ".join([a.summary, a.ending_open_question] + [f"{s.reason} {s.open_question or ''}" for s in a.beats])


def _semantics(a: EpisodeAnalysis) -> dict:
    text = _analysis_text(a)
    score = Counter()
    for s in a.beats:
        if s.hook_type in _HOOK_EMOTION:
            score[_HOOK_EMOTION[s.hook_type]] += max(s.hook, s.cliff)
    for name, cue in _EMOTION_CUES.items():
        score[name] += 3 * bool(cue.search(text))
    emotions = sorted((e for e in EMOTIONS if score[e] > 0), key=lambda e: (-score[e], EMOTIONS.index(e)))[:3]
    return {"tropes": [t for t, cue in _TROPE_CUES.items() if cue.search(text)][:3], "emotions": emotions}


def _tension(a: EpisodeAnalysis) -> float:
    """0.5 x mean of the top-3 cliff scores + 0.2 x top-3 hook scores + 0.3 x the ending cliffhanger, all /5."""
    top3 = lambda xs: sum(sorted(xs, reverse=True)[:3]) / 3 / 5
    t = 0.5 * top3([s.cliff for s in a.beats]) + 0.2 * top3([s.hook for s in a.beats]) \
        + 0.3 * a.ending_cliffhanger.score / 5
    return round(min(1.0, t), 3)


def _draft(result: RunResult) -> _Draft:
    cast = _Cast(result)
    d = _Draft(result=result, cast=cast, loops=_loops(result), bonds=_role_bonds(result, cast),
               semantics={a.episode: _semantics(a) for a in _usable(result.analyses)})
    if cast.dropped:
        ep, cid, alias = cast.dropped[0]
        d.warnings.append(f"ignored {len(cast.dropped)} alias claim(s) whose title contradicts the character's gender "
                          f"(first: E{ep:02d} {cid} <- {alias!r})")
    return d


# ------------------------------------------------------------------ show-level LLM pass
REPLY_KEYS = ("characters", "relationships", "reveals", "ending_questions", "threads", "episodes", "notes")


def system_prompt() -> tuple[str, str]:
    text = PROMPT_FILE.read_text()
    for name, values in (("ARCHETYPES", ARCHETYPES), ("REL_KINDS", REL_KINDS), ("LOOP_KINDS", LOOP_KINDS),
                         ("TROPES", tuple(TROPES)), ("EMOTIONS", EMOTIONS)):
        text = text.replace("{{" + name + "}}", ", ".join(values))
    return text, f"story_graph_system@{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def _cut(text, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _cast_by_ep(d: _Draft) -> dict[int, list[str]]:
    out: dict[int, set[str]] = defaultdict(set)
    for cid, info in d.cast.info.items():
        for ep in info["listed"]:
            out[ep].add(d.cast.root(cid))
    for (ep, _), cid in d.cast.labels.items():
        out[ep].add(d.cast.root(cid))
    return {ep: sorted(v) for ep, v in out.items()}


def show_digest(d: _Draft, eps: list[int], earlier: list[OpenLoop]) -> str:
    """The compact text the show-level pass reads: no screenplay lines, only the analyses' own summaries."""
    cast, by_ep = d.cast, _cast_by_ep(d)
    analyses = {a.episode: a for a in d.result.analyses}
    count = Counter(c for ep in eps for c in by_ep.get(ep, []))
    in_batch = sorted(count, key=lambda c: (-count[c], c))           # main characters first
    lines = [f"SHOW: {d.result.show} | {len(d.episode_numbers)} episodes in total | this message: "
             f"E{eps[0]:02d}-E{eps[-1]:02d}", "", "CHARACTERS (id | name | episodes | gender | also called | roles)"]
    for cid in in_batch:
        seen = sorted(ep for ep, ids in by_ep.items() if cid in ids)
        names = cast.merged(cid, "names")
        aka = [n for n, _ in (names + cast.merged(cid, "aliases")).most_common() if _key(n) != _key(_name(d, cid))][:4]
        roles = [r for r, _ in cast.merged(cid, "roles").most_common(2)]
        lines.append(f"{cid} | {_name(d, cid)} | E{seen[0]:02d}-E{seen[-1]:02d} ({len(seen)} eps) | "
                     f"{cast.gender(cid) or '?'} | {', '.join(_cut(x, 30) for x in aka) or '-'} | "
                     f"{'; '.join(_cut(r, 60) for r in roles) or 'speaker only'}")
    lines += ["", "EPISODES"]
    for ep in eps:
        a = analyses.get(ep)
        if a is None or a.summary.startswith("analysis failed"):
            lines.append(f"E{ep:02d} | analysis failed")
            continue
        p = a.pickup
        lines.append(f"E{ep:02d} | cast: {', '.join(by_ep.get(ep, [])) or '-'} | pickup: {p.status}"
                     + (f" - {_cut(p.note, 90)}" if p.note else ""))
        lines.append(f"  summary: {_cut(a.summary, 300)}")
        hooks = sorted((s for s in a.beats if s.hook >= 3), key=lambda s: -s.hook)[:2]
        cliffs = sorted((s for s in a.beats if s.cliff >= 3), key=lambda s: -s.cliff)[:2]
        key_beats = sorted({s.beat_id: s for s in hooks + cliffs}.values(), key=lambda s: s.beat_id)
        if key_beats:
            lines.append("  beats: " + " | ".join(
                f"{s.beat_id} hook {s.hook}{' ' + s.hook_type if s.hook_type else ''} / cliff {s.cliff}: "
                f"{_cut(s.reason, 80)}" for s in key_beats))
        if a.ending_open_question.strip():
            lines.append(f"  ending [L-E{ep:02d}-end]: {_cut(a.ending_open_question, 160)}")
    if earlier:
        lines += ["", "EARLIER OPEN QUESTIONS (answer them in ending_questions if an episode here does)"]
        lines += [f"{l.id} (E{l.opened_ep:02d}): {_cut(l.question, 160)}" for l in earlier[-25:]]
    return "\n".join(lines)


def _check_usable(data) -> None:
    if not isinstance(data, dict) or not any(isinstance(data.get(k), list) for k in REPLY_KEYS):
        raise ValueError("reply must be one JSON object with lists: " + ", ".join(REPLY_KEYS))


def _vocab(values, allowed, cap: int) -> tuple[list[str], list[str]]:
    kept, bad = [], []
    for v in values if isinstance(values, list) else []:
        n = _norm(v)
        (kept if n in allowed and n not in kept else bad).append(n)
    return kept[:cap], [b for b in bad if b and b not in kept]


def _apply(d: _Draft, data: dict, batch: list[int], linkable: set[str]) -> None:
    """Fold one show-level reply into the draft. Every id and episode number is checked; bad ones are repaired
    (beat 'B4' -> 'E03.B004', a merged-away id -> its survivor) or dropped, each with a warning."""
    cast, warn, beat_ids, valid = d.cast, d.warnings.append, d.beat_ids, set(d.episode_numbers)

    def cid(x) -> Optional[str]:
        s = _slug(x)
        return cast.root(s) if s in cast.parent else None

    def epn(x) -> Optional[int]:
        try:
            n = int(x)
        except (TypeError, ValueError):
            return None
        return n if n in valid else None

    def items(key: str) -> list[dict]:
        raw = data.get(key) or []
        return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []

    strings = Counter()                        # same_as entries that are alias strings, not ids
    for c in items("characters"):
        me = cid(c.get("id"))
        if not me:
            warn(f"characters: unknown id {c.get('id')!r} dropped")
            continue
        for other in c.get("same_as") or []:
            o = cid(other)
            if o is None:   # an alias string: accepted only when code already resolves it to this character
                strings["same character" if cast.resolve(-1, str(other)) == me else "other"] += 1
            elif o != me and (why := cast.apart(me, o)):
                warn(f"kept {me} and {o} apart: {why}")
            elif o != me:
                cast.merge(me, o)
        if 0 < len(name := str(c.get("name") or "").strip()) <= 60:
            d.names[cast.root(me)] = name
        arch = _norm(c.get("archetype"))
        if arch in ARCHETYPES:
            d.archetypes[cast.root(me)] = arch
        elif arch:
            warn(f"characters: {me} archetype {arch!r} is not in the list; dropped")
    if strings:
        warn(f"characters: same_as held {sum(strings.values())} alias strings instead of ids "
             f"({strings['same character']} already resolved to that character by code, {strings['other']} "
             "naming another character or nobody); only ids can merge characters")

    for r in items("relationships"):
        a, b = cid(r.get("a")), cid(r.get("b"))
        if not a or not b or a == b:
            warn(f"relationships: {r.get('a')!r}-{r.get('b')!r} does not name two known characters; dropped")
            continue
        label = _norm(r.get("label"))[:40]
        kind = _norm(r.get("kind"))
        if kind not in REL_KINDS:
            kind = _LABEL_KIND.get(label.split()[-1] if label else "", "")
            if not kind:
                warn(f"relationships: {a}-{b} kind {r.get('kind')!r} unknown; dropped")
                continue
        first = epn(r.get("first_ep"))
        if first is None and r.get("first_ep") is not None:
            warn(f"relationships: {a}-{b} first_ep {r.get('first_ep')!r} does not exist; set from co-presence")
        d.bonds.append({"a": a, "b": b, "ep": first, "label": label or kind, "kind": kind, "basis": "llm",
                        "change": False, "beat_id": None})
        for ch in r.get("changes") or []:
            ch = ch if isinstance(ch, dict) else {}
            ep, to = epn(ch.get("ep")), _norm(ch.get("to"))[:40]
            if ep is None or not to:
                warn(f"relationships: {a}-{b} change {ch!r} has no valid episode or label; dropped")
                continue
            beat, why = _fix_beat(ch.get("beat_id"), ep, beat_ids)
            if why:
                warn(f"relationships: {a}-{b} change E{ep:02d}: {why}")
            d.bonds.append({"a": a, "b": b, "ep": ep, "label": to, "kind": _LABEL_KIND.get(to.split()[-1], kind),
                            "basis": "llm", "change": True, "beat_id": beat})

    for r in items("reveals"):
        who, ep = cid(r.get("character")), epn(r.get("ep"))
        if not who or ep is None:
            warn(f"reveals: {r.get('character')!r} E{r.get('ep')!r} is not a known character and episode; dropped")
            continue
        beat, why = _fix_beat(r.get("beat_id"), ep, beat_ids)
        if why:
            warn(f"reveals: {who} E{ep:02d}: {why}")
        d.reveals.append({"character": who, "ep": ep, "beat_id": beat, "what": _cut(r.get("what"), 80),
                          "basis": "llm"})

    loops = {l.id: l for l in d.loops}
    for q in items("ending_questions"):
        m = re.search(r"E?(\d+)", str(q.get("loop") or ""), re.I)
        lid = f"L-E{int(m.group(1)):02d}-end" if m else ""
        loop = loops.get(lid)
        if loop is None or lid not in linkable:
            warn(f"ending_questions: loop {q.get('loop')!r} is not one of this message's loops; dropped")
            continue
        if _norm(q.get("kind")) in LOOP_KINDS:
            loop.kind = _norm(q.get("kind"))
        answer = q.get("answered_ep")
        if answer in (None, "", "null"):
            loop.answered_ep = loop.answered_beat = None
            loop.status, loop.basis = "open", "llm"
            continue
        ep = epn(answer)
        if ep is None or ep <= loop.opened_ep:
            warn(f"ending_questions: {lid} answered_ep {answer!r} is not a later episode; kept "
                 f"{loop.basis or 'open'}")
            continue
        beat, why = _fix_beat(q.get("answered_beat"), ep, beat_ids)
        if why:
            warn(f"ending_questions: {lid}: {why}")
        loop.answered_ep, loop.answered_beat, loop.status, loop.basis = ep, beat, "answered", "llm"

    threads = {_key(l.question) for l in d.loops if l.scope == "thread"}
    for t in items("threads")[:8]:
        question, opened = _cut(t.get("question"), 200), epn(t.get("opened_ep"))
        if not question or opened is None or _key(question) in threads:
            if not question or opened is None:
                warn(f"threads: {t.get('opened_ep')!r} is not a valid opening episode; dropped")
            continue
        answer = t.get("answered_ep")
        ans = epn(answer) if answer not in (None, "", "null") else None
        if answer not in (None, "", "null") and (ans is None or ans <= opened):
            warn(f"threads: answered_ep {answer!r} is not after E{opened:02d}; kept open")
            ans = None
        kind = _norm(t.get("kind"))
        threads.add(_key(question))
        d.loops.append(OpenLoop(id=f"T{len(threads):02d}", question=question, scope="thread", opened_ep=opened,
                                kind=kind if kind in LOOP_KINDS else _loop_kind(question),
                                answered_ep=ans, status="answered" if ans else "open", basis="llm"))

    for e in items("episodes"):
        ep = epn(e.get("ep"))
        if ep is None:
            warn(f"episodes: episode {e.get('ep')!r} does not exist; dropped")
            continue
        tropes, bad_t = _vocab(e.get("tropes"), TROPES, 3)
        emotions, bad_e = _vocab(e.get("emotions"), EMOTIONS, 3)
        if bad_t or bad_e:
            warn(f"episodes: E{ep:02d} dropped off-list tags {bad_t + bad_e}")
        sem = d.semantics.setdefault(ep, {"tropes": [], "emotions": []})
        if tropes or e.get("tropes") == []:
            sem["tropes"] = tropes
        if emotions:
            sem["emotions"] = emotions
    d.notes += [_cut(n, 200) for n in (data.get("notes") or [])[:3] if isinstance(n, str) and n.strip()]


def _llm_pass(d: _Draft, client: LLMClient, log: Optional[RunLog], max_prompt_tokens: int, batch_size: int) -> None:
    system, version = system_prompt()
    eps = d.episode_numbers
    whole = show_digest(d, eps, [])
    batches = ([eps] if (len(system) + len(whole)) / 4 <= max_prompt_tokens
               else [eps[i:i + batch_size] for i in range(0, len(eps), batch_size)])
    for i, batch in enumerate(batches, 1):
        earlier = [l for l in d.loops if l.scope == "episode" and l.status == "open" and l.opened_ep < batch[0]]
        user = whole if len(batches) == 1 else show_digest(d, batch, earlier)
        linkable = {l.id for l in d.loops if l.scope == "episode" and l.opened_ep in batch} | {l.id for l in earlier}
        item = "show" if len(batches) == 1 else f"show {i}/{len(batches)}"
        try:
            res = client.complete_json(system, user, validate=_check_usable, cache_tag=version)
            _check_usable(res.data)   # cached replies skip the validate callback
        except Exception as e:
            d.warnings.append(f"show-level pass {item} failed ({type(e).__name__}); E{batch[0]:02d}-E{batch[-1]:02d} "
                              "keep the deterministic layer")
            if log:
                log.record("genome", item, error=f"{type(e).__name__}: {e}")
            continue
        before = len(d.warnings)
        _apply(d, res.data, batch, linkable)
        if log:
            log.record("genome", item, res, warnings=d.warnings[before:])
        d.model, d.version = f"{res.provider}:{res.model}", version


# ------------------------------------------------------------------ assembly: projection, profiles, alerts
def _name(d: _Draft, root: str) -> str:
    for c in [root] + d.cast.members(root):
        if c in d.names:
            return d.names[c]
    names = d.cast.merged(root, "names")
    proper = [n for n, _ in names.most_common() if not _descriptor(canonical_name(n)[0])]
    return proper[0] if proper else names.most_common(1)[0][0]


_ARCHETYPE_CUES = [(name, re.compile(rf"\b(?:{p})", re.I)) for name, p in (
    ("wronged heroine", r"bride|heroine|wronged|disowned|disgraced|outcast|scorned|abandoned|jilted|accused"),
    ("hidden-identity protector", r"disguis\w*|hidden|mysterious|stranger|scarred|beggar|veteran|soldier|rightful"),
    ("scheming sister", r"(?:cruel|spiteful|scheming|jealous|wicked|envious)\W+(?:\w+\W+)?sister"),
    ("betrayer", r"traitor\w*|unfaithful|faithless|ex\b|former groom"),
    ("complicit parent", r"(?:cold|cruel|harsh|unsympathetic|complicit|greedy)\W+(?:\w+\W+)?(?:father|mother|parent)"),
    ("cruel matriarch", r"matriarch|stepmother|dowager"),
    ("villain", r"villain|antagonist|schem\w*|conspirator|usurper|impost[eo]r|blackmail\w*|thug"),
    ("loyal ally", r"loyal|ally|friend|maid|servant|protector|confidant"),
    ("authority figure", r"judge|constable|guard|officer|official|priest|magistrate|captain|police"))]


def _guess_archetype(roles: Counter) -> str:
    score = Counter()
    for role, n in roles.items():
        for name, cue in _ARCHETYPE_CUES:
            score[name] += n * bool(cue.search(role))
    best = [name for name, _ in _ARCHETYPE_CUES if score[name] and score[name] == max(score.values())]
    return best[0] if best else ""


def _mention_index(graph: StoryGraph) -> tuple[Optional[re.Pattern], dict[str, str]]:
    """Proper names only (analysis names and speaker labels, never descriptors such as 'lady in green' or
    forms of address such as 'her grace'); a given name comes only from a full analysis name."""
    owners: dict[str, set[str]] = defaultdict(set)
    for row in graph.aliases:
        key = canonical_name(row.alias)[0]
        if (not {"analysis name", "speaker label"} & set(row.sources) or len(key) < 3 or _descriptor(key)
                or set(key.split()) & _STOP):
            continue
        owners[key].add(row.canonical)
        if len(key.split()) > 1 and "analysis name" in row.sources:
            owners[key.split()[0]].add(row.canonical)
    keys = {k: next(iter(v)) for k, v in owners.items() if len(v) == 1}
    if not keys:
        return None, {}
    alt = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    return re.compile(rf"(?<![\w'])(?:{alt})(?![\w-])", re.I), keys


def beat_cast(result: RunResult, graph: StoryGraph) -> dict[str, list[str]]:
    """beat id -> character ids on screen: the speaker (via the alias table, per episode) first, then characters
    named in an action or camera line (proper names only, in order of mention)."""
    return cast_by_beat(result.episodes, graph)


def cast_by_beat(episodes, graph: StoryGraph) -> dict[str, list[str]]:
    """beat_cast for a list of parsed episodes (rank.py calls it before a RunResult exists)."""
    speaker: dict[tuple[str, int], str] = {}
    for row in sorted(graph.aliases, key=lambda r: "speaker label" not in r.sources):
        for ep in row.episodes:
            speaker.setdefault((_key(row.alias), ep), row.canonical)
    pattern, keys = _mention_index(graph)
    out = {}
    for ep in episodes:
        for b in ep.beats:
            ids = []
            if b.speaker and (who := speaker.get((_key(b.speaker), ep.number))):
                ids.append(who)
            if b.kind in ("action", "camera") and pattern is not None:
                for m in pattern.finditer(b.text.replace("’", "'")):
                    who = keys.get(_key(m.group(0)))
                    if who and who not in ids:
                        ids.append(who)
            out[b.id] = ids
    return out


def beats_inside(episodes: dict, segments) -> list:
    """Complete beats inside the segments (the same rule genome.py and the editors use)."""
    return [b for s in segments for b in getattr(episodes.get(s.episode), "beats", [])
            if s.start <= b.start < s.end and b.end <= s.end]


def _union(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if e > end:
            total += e - max(s, end)
            end = e
    return total


def _polarity(label: str) -> int:
    words = set(re.findall(r"[a-zé-]+", label))
    return -1 if words & _HOSTILE_WORDS else 1 if words & _WARM_WORDS else 0


def _assemble(d: _Draft) -> StoryGraph:
    result, cast = d.result, d.cast
    episodes = {e.number: e for e in result.episodes}
    analyses = {a.episode: a for a in _usable(result.analyses)}
    scores = {s.beat_id: s for a in analyses.values() for s in a.beats}
    roots = cast.roots()

    # characters and the alias -> canonical table (title / role aliases go to CharacterNode.roles instead)
    role_aliases: dict[str, set[str]] = defaultdict(set)
    for _, cid, alias in cast.role_claims:
        role_aliases[cast.root(cid)].add(alias)
    chars = {}
    for r in roots:
        roles = cast.merged(r, "roles")
        names = cast.merged(r, "names")
        aliases = [n for n, _ in (names + cast.merged(r, "aliases")).most_common() if _key(n) != _key(_name(d, r))
                   and (n in names or n not in role_aliases[r])]
        chars[r] = CharacterNode(id=r, name=_name(d, r), aliases=list(dict.fromkeys(aliases))[:12],
                                 role="; ".join(x for x, _ in roles.most_common(2)), gender=cast.gender(r),
                                 archetype=next((d.archetypes[c] for c in [r] + cast.members(r) if c in d.archetypes),
                                                _guess_archetype(roles)))
    rows: dict[tuple[str, str], dict] = {}
    for (ep, label), cid in cast.labels.items():
        rows.setdefault((_key(label), cast.root(cid)), {"alias": label, "eps": Counter(), "src": set()})
        row = rows[(_key(label), cast.root(cid))]
        row["eps"][ep] += sum(1 for b in episodes[ep].beats if b.speaker == label)
        row["src"].add("speaker label")
    for ep, label, cid, source in cast.mentions:
        row = rows.setdefault((_key(label), cast.root(cid)), {"alias": label, "eps": Counter(), "src": set()})
        row["eps"][ep] += 1
        row["src"].add(source)
    alias_rows = [AliasRow(alias=v["alias"], canonical=root, first_ep=min(v["eps"]), last_ep=max(v["eps"]),
                           episodes=sorted(v["eps"]), count=sum(v["eps"].values()), sources=sorted(v["src"]))
                  for (_, root), v in rows.items()]
    alias_rows.sort(key=lambda r: (r.canonical, -r.count, r.alias))
    graph = StoryGraph(show=result.show, model=d.model, prompt_version=d.version, characters=list(chars.values()),
                       aliases=alias_rows)

    # presence: the character <-> scene bipartite graph
    on = beat_cast(result, graph)
    deliverer = {bid: ids[0] for bid, ids in on.items() if ids}
    scene_time: dict[tuple[int, int], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    strong = {s.beat_id for s in scores.values() if max(s.hook, s.cliff) >= STRONG and s.beat_id in on}
    for ep in result.episodes:
        for b in ep.beats:
            for cid in on[b.id]:
                scene_time[(ep.number, b.scene)][cid].append((b.start, b.end))
    eps_of = {r: set(cast.merged(r, "listed")) for r in roots}   # listed by an analysis counts as present
    scenes_of: dict[str, set] = defaultdict(set)
    for (ep_no, scene), present in scene_time.items():
        for cid in present:
            eps_of[cid].add(ep_no)
            scenes_of[cid].add((ep_no, scene))
    clips_of, promos_of = Counter(), Counter()
    for items, counter in ((result.candidates, clips_of), (result.promos, promos_of)):
        for item in items:
            counter.update({cid for b in beats_inside(episodes, item.segments) for cid in on.get(b.id, [])})

    # character <-> character projection with the bidder-style statistics
    blank = lambda: {"scenes": 0, "eps": set(), "overlap": 0.0, "strong": Counter(), "hostile": 0, "warm": 0}
    pairs: dict[frozenset, dict] = defaultdict(blank)
    for (ep_no, scene), present in scene_time.items():
        scored = [b for b in episodes[ep_no].beats if b.scene == scene and b.id in scores and b.id in deliverer]
        for a, b in combinations(sorted(present), 2):
            p = pairs[frozenset((a, b))]
            p["scenes"] += 1
            p["eps"].add(ep_no)
            p["overlap"] += min(_union(present[a]), _union(present[b]))
            theirs = [x for x in scored if deliverer[x.id] in (a, b)]   # beats one of the two delivers
            p["strong"].update(deliverer[x.id] for x in theirs if x.id in strong)
            p["hostile"] += sum(scores[x.id].hook_type in HOSTILE_HOOKS for x in theirs)
            p["warm"] += sum(scores[x.id].hook_type in WARM_HOOKS for x in theirs)

    # named bonds: LLM events win over role-text events for the same pair
    events: dict[frozenset, list[dict]] = defaultdict(list)
    for ev in d.bonds:
        a, b = cast.root(ev["a"]), cast.root(ev["b"])
        if a != b:
            events[frozenset((a, b))].append({**ev, "a": a, "b": b})
    edges = []
    for pair in set(events) | {k for k, v in pairs.items() if v["scenes"] >= MIN_SHARED_SCENES}:
        evs = events.get(pair, [])
        llm = [e for e in evs if e["basis"] == "llm"]
        use = sorted(llm or evs, key=lambda e: (e["change"], e["ep"] or 0))
        a, b = (use[0]["a"], use[0]["b"]) if use else tuple(sorted(pair))
        p = pairs.get(pair) or blank()
        e = RelationEdge(a=a, b=b, basis=("llm" if llm else "analysis roles") if use else "co-presence")
        if use:
            first = use[0]
            e.kind, e.label = first["kind"], first["label"]
            current = _LABEL_GROUP.get(first["label"], first["label"])
            for ev in use[1:]:
                group = _LABEL_GROUP.get(ev["label"], ev["label"])
                if (ev["change"] or not llm) and group != current:   # a "change" to the same label is no turn
                    e.changes.append(RelationChange(ep=ev["ep"], to=ev["label"], beat_id=ev["beat_id"]))
                    current = group
        firsts = [x["ep"] for x in evs if x["ep"]] + ([min(p["eps"])] if p["eps"] else [])
        e.first_ep = min(firsts) if firsts else 0
        sa, sb = len(scenes_of[a]), len(scenes_of[b])
        e.shared_scenes, e.shared_episodes, e.overlap_s = p["scenes"], len(p["eps"]), round(p["overlap"], 1)
        e.jaccard = round(p["scenes"] / max(1, sa + sb - p["scenes"]), 3)
        e.a_strong, e.b_strong = p["strong"][a], p["strong"][b]
        total = e.a_strong + e.b_strong
        e.asymmetry = round(abs(e.a_strong - e.b_strong) / total, 3) if total else 0.0
        e.dominant = (a if e.a_strong > e.b_strong else b) if e.a_strong != e.b_strong else ""
        e.hostile_beats, e.warm_beats = p["hostile"], p["warm"]
        typed = e.hostile_beats + e.warm_beats
        e.hostility = round(e.hostile_beats / typed, 3) if typed else 0.0
        # the syndicate_risk_score analogue: additive, every part capped and visible; dominance and hostility
        # only count in full once there is enough evidence (like the bidders' "10+ pair wins" gate), and
        # hostility is net of warmth (an even hostile/warm mix, typical of the romantic leads, scores 0)
        e.score_parts = {"volume": round(25.0 * min(1.0, e.shared_scenes / 12), 1),
                         "co_presence": round(20.0 * min(1.0, e.jaccard / 0.5), 1),
                         "dominance": round(20.0 * e.asymmetry * min(1.0, total / 8), 1),
                         "hostility": round(35.0 * max(0.0, 2 * e.hostility - 1) * min(1.0, typed / 10), 1)}
        e.conflict_intensity = round(min(100.0, sum(e.score_parts.values())), 1)
        e.core = e.conflict_intensity >= CORE_CONFLICT_MIN
        edges.append(e)
    edges.sort(key=lambda e: (-e.conflict_intensity, -e.shared_scenes, e.a, e.b))
    graph.relationships = edges

    # profiles (the bidder_profiles analogue)
    costars: dict[str, Counter] = defaultdict(Counter)
    for e in edges:
        costars[e.a][e.b] = costars[e.b][e.a] = e.shared_scenes
    for c in graph.characters:
        present = sorted(eps_of[c.id])
        c.episodes_present = present
        c.first_ep, c.last_ep = (present[0], present[-1]) if present else (0, 0)
        mine = [b for ep in result.episodes for b in ep.beats if c.id in on[b.id]]
        c.lines = sum(1 for b in mine if b.kind in ("dialogue", "vo") and deliverer.get(b.id) == c.id)
        c.screen_time_s = round(sum(_union(v[c.id]) for v in scene_time.values() if c.id in v), 1)
        c.scenes = len(scenes_of[c.id])
        c.hook_lines = sum(1 for bid in strong if scores[bid].hook >= STRONG and deliverer.get(bid) == c.id)
        c.cliff_beats = sum(1 for bid in strong if scores[bid].cliff >= STRONG and c.id in on[bid])
        c.clips, c.promos = clips_of[c.id], promos_of[c.id]
        c.promo_pull = round(sum(1 for bid in strong if c.id in on[bid]) / len(strong), 3) if strong else 0.0
        c.top_costars = [x for x, _ in costars[c.id].most_common(3)]
    spans, unproven = _roles(cast, episodes, scene_time)
    for c in graph.characters:
        c.roles = spans.get(c.id, [])
    graph.characters.sort(key=lambda c: (-len(c.episodes_present), -c.lines, c.id))

    # loops and episodes
    graph.loops = sorted(d.loops, key=lambda l: (l.opened_ep, {"thread": 0, "scene": 1, "episode": 2}[l.scope], l.id))
    for ep_no in sorted(episodes):
        a = analyses.get(ep_no)
        sem = d.semantics.get(ep_no, {})
        graph.episodes.append(EpisodeNode(
            ep=ep_no, tension=_tension(a) if a else 0.0, emotions=sem.get("emotions", []),
            tropes=sem.get("tropes", []), characters=sorted(c for c, s in eps_of.items() if ep_no in s),
            opening_hook=a.opening_hook.score if a else 0, ending_cliff=a.ending_cliffhanger.score if a else 0,
            pickup=a.pickup.status if a else "",
            loops_opened=[l.id for l in graph.loops if l.opened_ep == ep_no],
            loops_closed=[l.id for l in graph.loops if l.answered_ep == ep_no],
            loops_carried=[l.id for l in graph.loops if l.scope != "scene" and l.opened_ep <= ep_no
                           and (l.answered_ep is None or l.answered_ep > ep_no)]))
    counts = Counter(t for e in graph.episodes for t in e.tropes)
    graph.show_tropes = [t for t, n in counts.most_common(TOP_TROPES)]
    graph.alerts = _alerts(d, graph, on, scores)
    graph.notes = d.notes + [
        f"{sum(not cast.merged(r, 'listed') for r in roots)} speaker-only characters: a speaker label that no "
        "episode's character list claimed."]
    if cast.role_claims:
        graph.notes.append(
            f"{len(cast.role_claims)} title/role aliases (e.g. 'Duchess', 'disgraced bride') kept out of identity "
            f"resolution: {sum(len(s) for s in spans.values())} became time-scoped roles with script evidence, "
            f"{unproven} had no evidence beat and were left out.")
    if d.model == "deterministic":
        graph.notes.append("Deterministic layer only: an ending question counts as answered when the next episode "
                           "picks the cliffhanger up (a proxy), and tropes/emotions come from keyword cues.")
    graph.warnings = d.warnings
    return graph


def _roles(cast: _Cast, episodes: dict, scene_time: dict) -> tuple[dict[str, list[RoleSpan]], int]:
    """Title / role aliases -> time-scoped roles. Evidence: in an episode whose analysis gave the role, a beat that
    names it (text or speaker label) in a scene the character is in. Roles without such a beat are left out."""
    claims: dict[tuple[str, str], set[int]] = defaultdict(set)
    for ep, cid, alias in cast.role_claims:
        claims[(cast.root(cid), canonical_name(alias)[0])].add(ep)
    out: dict[str, list[RoleSpan]] = defaultdict(list)
    unproven = 0
    for (who, role), eps in sorted(claims.items()):
        cue = re.compile(rf"\b{re.escape(_role_head(role))}(?:e?s)?\b", re.I)
        proof = {}
        for ep in sorted(eps):
            beat = next((b.id for b in getattr(episodes.get(ep), "beats", []) if who in scene_time.get((ep, b.scene), {})
                         and cue.search(f"{b.speaker or ''} {b.text}".replace("’", "'"))), None)
            if beat:
                proof[ep] = beat
        if not proof:
            unproven += 1
            continue
        out[who].append(RoleSpan(role=role, first_ep=min(proof), last_ep=max(proof), episodes=sorted(proof),
                                 evidence_beat=proof[min(proof)]))
    for spans in out.values():
        spans.sort(key=lambda s: (s.first_ep, s.role))
    return out, unproven


def _alerts(d: _Draft, graph: StoryGraph, on: dict[str, list[str]], scores: dict) -> list[StoryAlert]:
    """The forensic-alert analogue: each alert names its type, severity, confidence and evidence beats."""
    cast, out = d.cast, []
    core = {frozenset((e.a, e.b)) for e in graph.relationships if e.core}
    for e in graph.relationships:
        previous = e.label
        for ch in e.changes:
            flip = _polarity(previous) * _polarity(ch.to) < 0 or (not _polarity(previous) and _polarity(ch.to))
            conf = (0.8 if ch.beat_id else 0.65) if e.basis == "llm" else 0.5
            out.append(StoryAlert(id="", type="relationship_reversal", ep=ch.ep, confidence=conf,
                                  severity="high" if frozenset((e.a, e.b)) in core or flip else "medium",
                                  beat_ids=[ch.beat_id] if ch.beat_id else [], characters=[e.a, e.b],
                                  summary=f"{e.a} -> {e.b}: {previous} -> {ch.to}",
                                  evidence={"from": previous, "to": ch.to, "basis": e.basis,
                                            "conflict_intensity": e.conflict_intensity}))
            previous = ch.to
    reveals: dict[tuple[str, int], dict] = {}
    for r in cast.reveals + d.reveals:   # deterministic first; an LLM report for the same episode confirms it
        who = cast.root(r["character"])
        hit = next((k for k in reveals if k == (who, r["ep"])), None)
        if hit:
            reveals[hit]["confirmed"] = True
            reveals[hit]["what"] = reveals[hit].get("what") or r.get("what", "")
            reveals[hit]["beat_id"] = reveals[hit].get("beat_id") or r.get("beat_id")
        else:
            reveals[(who, r["ep"])] = dict(r, character=who)
    episodes = {e.number: e for e in d.result.episodes}
    for (who, ep), r in reveals.items():
        beats = [r["beat_id"]] if r.get("beat_id") else []
        named = next((b.id for b in getattr(episodes.get(ep), "beats", []) if b.speaker and on.get(b.id)[:1] == [who]
                      and not _descriptor(canonical_name(b.speaker)[0])), None)
        typed = max((s for s in scores.values() if s.beat_id.startswith(f"E{ep:02d}.") and s.hook_type == "reveal"),
                    key=lambda s: s.hook, default=None)
        beats += [x for x in (named, typed.beat_id if typed else None) if x and x not in beats]
        conf = 0.85 if r.get("confirmed") else 0.7 if r["basis"] == "llm" else 0.55
        was = cast.root(r["was"]) if r.get("was") else ""
        out.append(StoryAlert(id="", type="identity_reveal", ep=ep, confidence=conf, severity="high",
                              beat_ids=beats, characters=[who], summary=f"{who}: {r.get('what') or 'identity revealed'}",
                              evidence={"basis": r["basis"] + (" + llm" if r.get("confirmed") else ""),
                                        "known_before_as": [a for a in graph.characters if a.id == who][0].aliases[:3]
                                        if was else [], "what": r.get("what", "")}))
    last = max((e.ep for e in graph.episodes), default=0)
    conf = {"llm": 0.7, "analysis": 0.8, "pickup": 0.4, "": 0.5}
    for l in graph.loops:
        if l.scope == "scene":
            continue
        span = (l.answered_ep or last) - l.opened_ep
        if l.status == "open" and span >= 3 or l.scope == "thread":
            out.append(StoryAlert(id="", type="loop_opened", ep=l.opened_ep, confidence=conf.get(l.basis, 0.5),
                                  severity="high" if l.strength >= STRONG or l.scope == "thread" else "medium",
                                  beat_ids=[l.opened_beat] if l.opened_beat else [],
                                  summary=f"{l.id} ({l.kind}) opens; "
                                          + (f"answered E{l.answered_ep:02d}" if l.answered_ep else "still open"),
                                  evidence={"loop": l.id, "scope": l.scope, "open_for_episodes": span}))
        if l.answered_ep and l.answered_ep - l.opened_ep >= 2:
            out.append(StoryAlert(id="", type="loop_closed", ep=l.answered_ep, confidence=conf.get(l.basis, 0.5),
                                  severity="medium", beat_ids=[l.answered_beat] if l.answered_beat else [],
                                  summary=f"{l.id} ({l.kind}) pays off {l.answered_ep - l.opened_ep} episodes later",
                                  evidence={"loop": l.id, "opened_ep": l.opened_ep, "basis": l.basis}))
    out.sort(key=lambda a: (a.ep, a.type, a.summary))
    for i, a in enumerate(out, 1):
        a.id = f"A{i:03d}"
    return out


def build_story_graph(result: RunResult, client: Optional[LLMClient] = None, *, log: Optional[RunLog] = None,
                      max_prompt_tokens: int = MAX_PROMPT_TOKENS, batch_episodes: int = BATCH_EPISODES) -> StoryGraph:
    """Deterministic story graph from the saved analyses, enriched by one show-level LLM call when `client` is given."""
    d = _draft(result)
    if client is not None and _usable(result.analyses):
        _llm_pass(d, client, log, max_prompt_tokens, batch_episodes)
    return _assemble(d)


# ------------------------------------------------------------------ queries (get_top_syndicates / get_bidder_profile)
def top_conflict_pairs(graph: StoryGraph, n: int = 10) -> list[dict]:
    names = {c.id: c.name for c in graph.characters}
    edges = sorted(graph.relationships, key=lambda e: (-e.conflict_intensity, -e.shared_scenes))[:n]
    return [{"a_name": names.get(e.a, e.a), "b_name": names.get(e.b, e.b), **e.model_dump()} for e in edges]


def feud_pairs(graph: StoryGraph, n: int = 3) -> list[RelationEdge]:
    """The strongest core conflict pairs that are on opposite sides: the bond is a rivalry, any label it carried is
    hostile (enemy, betrayer, ...), or the graph saw it reverse. Allies and family without a hostile turn are left
    out, because conflict_intensity counts hostile beats EITHER of the two delivers in shared scenes: two villains
    who torment the heroine together score as high as the heroine and her enemy."""
    reversed_ = {frozenset(a.characters) for a in graph.alerts if a.type == "relationship_reversal"}
    feuds = [e for e in graph.relationships if e.core and (
        e.kind == "rivalry" or any(_polarity(x) < 0 for x in [e.label] + [c.to for c in e.changes])
        or frozenset((e.a, e.b)) in reversed_)]
    return sorted(feuds, key=lambda e: (-e.conflict_intensity, e.a, e.b))[:n]


def refresh_promo_counts(graph: StoryGraph, result: RunResult) -> StoryGraph:
    """CharacterNode.promos for the final list (the pipeline builds the graph before ranking, when it is empty)."""
    episodes = {e.number: e for e in result.episodes}
    on = beat_cast(result, graph)
    counts = Counter(cid for p in result.promos
                     for cid in {c for b in beats_inside(episodes, p.segments) for c in on.get(b.id, [])})
    for c in graph.characters:
        c.promos = counts[c.id]
    return graph


def find_character(graph: StoryGraph, who: str) -> Optional[CharacterNode]:
    """By id, name or any alias (canonicalised: 'LADY MARA' finds Mara)."""
    key = canonical_name(who)[0]
    by_id = {c.id: c for c in graph.characters}
    hit = by_id.get(who) or by_id.get(_slug(who))
    hit = hit or next((c for c in graph.characters if canonical_name(c.name)[0] == key), None)
    rows = Counter(r.canonical for r in graph.aliases if canonical_name(r.alias)[0] == key)
    return hit or (by_id.get(rows.most_common(1)[0][0]) if rows else None)


def character_profile(graph: StoryGraph, who: str) -> Optional[dict]:
    c = find_character(graph, who)
    if c is None:
        return None
    names = {x.id: x.name for x in graph.characters}
    costars = []
    for e in sorted((e for e in graph.relationships if c.id in (e.a, e.b)), key=lambda e: -e.shared_scenes)[:10]:
        mine = e.a == c.id
        costars.append({"partner": e.b if mine else e.a, "partner_name": names.get(e.b if mine else e.a, ""),
                        "label": e.label, "kind": e.kind, "shared_scenes": e.shared_scenes,
                        "my_strong": e.a_strong if mine else e.b_strong,
                        "partner_strong": e.b_strong if mine else e.a_strong,
                        "conflict_intensity": e.conflict_intensity, "core": e.core})
    return {**c.model_dump(), "co_stars": costars,
            "alerts": [a.model_dump() for a in graph.alerts if c.id in a.characters],
            "aliases_table": [r.model_dump() for r in graph.aliases if r.canonical == c.id]}


# ------------------------------------------------------------------ per-episode features for the engagement layer
FEATURES = {
    "analysis_failed": "1 if the episode's analysis failed (its other features are then 0)",
    "duration_s": "episode length in seconds",
    "episode_index_frac": "position in the show: 0 = first episode, 1 = last",
    "opening_hook": "the episode's own opening-hook score, 0-5",
    "ending_cliff": "the episode's own ending-cliffhanger score, 0-5",
    "strong_beats": "beats scored 4+ for hook or cliff",
    "tension": "0-1: 0.5 x mean top-3 cliff + 0.2 x mean top-3 hook + 0.3 x ending cliff, each / 5",
    "spoiler_max": "highest spoiler score (0-3) among the episode's beats",
    "pickup_delay_s": "seconds before the episode moves past the previous cliffhanger (0 if not given)",
    **{f"pickup_{p}": f"1 if the episode picks up the previous cliffhanger as '{p}'" for p in PICKUPS},
    **{f"hook_type_{t}": f"1 if the episode's strongest hook beat is typed '{t}'" for t in HOOK_TYPES},
    "loops_opened": "questions raised in the episode (scored-beat questions + the ending question)",
    "loops_closed": "questions answered in the episode (in-episode answers + earlier questions paid off here)",
    "loops_carried": "cross-episode questions (ending questions, long threads) still open at the episode's end",
    "oldest_open_loop_age": "episodes since the oldest carried question was raised (0 if none or raised here)",
    "payoffs_old": "questions answered here that were raised 2+ episodes earlier",
    "scene_loops_unresolved": "scored-beat questions the episode itself never answers",
    "characters_present": "characters on screen or listed (after cross-episode merging)",
    "new_characters": "characters first seen in this episode",
    "relationship_changes": "relationship turns recorded in this episode",
    "new_relationships": "named relationships first shown in this episode",
    "core_conflict_present": "1 if both sides of a core conflict pair appear in the episode",
    "max_conflict_intensity": "highest conflict_intensity (0-100) among pairs that both appear in the episode",
    "identity_reveals": "identity-reveal alerts in this episode",
    "relationship_reversals": "relationship-reversal alerts in this episode",
    **{f"emotion_{e}": f"1 if '{e}' is among the episode's top 3 emotions" for e in EMOTIONS},
    "trope_<slug>": "1 if the episode carries the trope; one column per show-level top trope (graph.show_tropes)",
}


def feature_doc(name: str) -> Optional[str]:
    return FEATURES.get(name) or (FEATURES["trope_<slug>"] if name.startswith("trope_") else None)


def episode_features(result: RunResult, graph: Optional[StoryGraph] = None) -> dict[int, dict[str, float]]:
    """Numbers per episode (names and meanings in FEATURES). Without a graph, the deterministic graph is built from
    the analyses alone; an LLM-enriched graph gives better loop links, bonds, reveals and tropes."""
    graph = graph or build_story_graph(result)
    analyses = {a.episode: a for a in _usable(result.analyses)}
    nodes = {e.ep: e for e in graph.episodes}
    loops = {l.id: l for l in graph.loops}
    numbers = sorted(e.number for e in result.episodes)
    trope_cols = {t: "trope_" + re.sub(r"[^a-z0-9]+", "_", t) for t in graph.show_tropes}
    core = [(e.a, e.b) for e in graph.relationships if e.core]
    out = {}
    for i, ep in enumerate(sorted(result.episodes, key=lambda e: e.number)):
        a, node = analyses.get(ep.number), nodes.get(ep.number) or EpisodeNode(ep=ep.number)
        f = {name: 0.0 for name in FEATURES if "<" not in name}
        f.update({col: 0.0 for col in trope_cols.values()})
        f["duration_s"] = round(ep.duration, 2)
        f["episode_index_frac"] = round(i / (len(numbers) - 1), 4) if len(numbers) > 1 else 0.0
        if a is None:
            f["analysis_failed"] = 1.0
            out[ep.number] = f
            continue
        f["opening_hook"], f["ending_cliff"] = float(a.opening_hook.score), float(a.ending_cliffhanger.score)
        f["strong_beats"] = float(sum(max(s.hook, s.cliff) >= STRONG for s in a.beats))
        f["tension"] = node.tension
        f["spoiler_max"] = float(max((s.spoiler for s in a.beats), default=0))
        f["pickup_delay_s"] = float(a.pickup.delay_s or 0.0)
        f[f"pickup_{a.pickup.status}"] = 1.0
        best = max(a.beats, key=lambda s: s.hook, default=None)
        if best and best.hook_type:
            f[f"hook_type_{best.hook_type}"] = 1.0
        f["loops_opened"], f["loops_closed"] = float(len(node.loops_opened)), float(len(node.loops_closed))
        f["loops_carried"] = float(len(node.loops_carried))
        f["oldest_open_loop_age"] = float(max((ep.number - loops[x].opened_ep for x in node.loops_carried), default=0))
        f["payoffs_old"] = float(sum(ep.number - loops[x].opened_ep >= 2 for x in node.loops_closed))
        f["scene_loops_unresolved"] = float(sum(loops[x].scope == "scene" and loops[x].status == "open"
                                                for x in node.loops_opened))
        present = set(node.characters)
        f["characters_present"] = float(len(present))
        f["new_characters"] = float(sum(c.first_ep == ep.number for c in graph.characters))
        f["relationship_changes"] = float(sum(ch.ep == ep.number for e in graph.relationships for ch in e.changes))
        f["new_relationships"] = float(sum(e.first_ep == ep.number and bool(e.kind) for e in graph.relationships))
        f["core_conflict_present"] = float(any(x in present and y in present for x, y in core))
        f["max_conflict_intensity"] = max((e.conflict_intensity for e in graph.relationships
                                           if e.a in present and e.b in present), default=0.0)
        f["identity_reveals"] = float(sum(x.type == "identity_reveal" and x.ep == ep.number for x in graph.alerts))
        f["relationship_reversals"] = float(sum(x.type == "relationship_reversal" and x.ep == ep.number
                                                for x in graph.alerts))
        for e in node.emotions:
            f[f"emotion_{e}"] = 1.0
        for t in node.tropes:
            if t in trope_cols:
                f[trope_cols[t]] = 1.0
        out[ep.number] = f
    return out


# ------------------------------------------------------------------ files and CLI
def write_story_graph(graph: StoryGraph, out_dir) -> Path:
    path = Path(out_dir) / "story_graph.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return path


def load_story_graph(path) -> StoryGraph:
    p = Path(path)
    return StoryGraph.model_validate_json((p / "story_graph.json" if p.is_dir() else p).read_text(encoding="utf-8"))


def _client(spec: Optional[str], use_cache: bool) -> LLMClient:
    from .config import ROOT, build_client, load_config
    from .llm.cache import DiskCache
    from .llm.registry import load_secrets, make_client
    cfg = load_config()
    if spec is None and "genome" in (cfg.get("models") or {}):
        return build_client(cfg, "genome", use_cache=use_cache)
    cache = DiskCache(ROOT / cfg.get("cache_dir", ".cache/llm"), enabled=use_cache)
    return make_client(spec or DEFAULT_MODEL, secrets=load_secrets(), cache=cache, max_tokens=16000)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a show's story graph (content genome) from a saved run.")
    parser.add_argument("--run", required=True, help="run folder (with result.json) or a result.json path")
    parser.add_argument("--out", help="folder for story_graph.json (default: the run folder)")
    parser.add_argument("--model", help=f"provider:model for the show-level pass (default {DEFAULT_MODEL})")
    parser.add_argument("--no-llm", action="store_true", help="deterministic layer only, no model call")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    run = Path(args.run)
    result = RunResult.model_validate_json(((run / "result.json") if run.is_dir() else run).read_text())
    out = Path(args.out) if args.out else (run if run.is_dir() else run.parent)
    log = RunLog()
    client = None if args.no_llm else _client(args.model, not args.no_cache)
    graph = build_story_graph(result, client, log=log)
    path = write_story_graph(graph, out)
    if log.rows:
        log.write_jsonl(out / "story_graph_log.jsonl")
    loops = [l for l in graph.loops if l.scope != "scene"]
    print(f"{path}  model={graph.model}")
    print(f"characters {len(graph.characters)} | aliases {len(graph.aliases)} | relationships "
          f"{len(graph.relationships)} ({sum(bool(e.kind) for e in graph.relationships)} named, "
          f"{sum(e.core for e in graph.relationships)} core) | alerts {len(graph.alerts)}")
    print(f"cross-episode loops {len(loops)}: {sum(l.status == 'answered' for l in loops)} answered, "
          f"{sum(l.status == 'open' for l in loops)} open | scene loops {len(graph.loops) - len(loops)}")
    print(f"top tropes: {', '.join(graph.show_tropes[:5]) or '-'} | warnings {len(graph.warnings)}")
    for row in log.rows:
        print(f"call {row.get('item')}: {row.get('provider', '')}:{row.get('model', '')} in {row.get('input_tokens', 0)} "
              f"out {row.get('output_tokens', 0)} {row.get('latency_s', 0)} s cached={row.get('cached')} "
              + (f"error={row['error'][:120]}" if "error" in row else ""))


if __name__ == "__main__":
    main()
