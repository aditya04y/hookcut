"""A show-local story graph with a sparse, transferable feature vector for each cut.

Edges cite parsed beat IDs and state whether a tag came from script wording, timing,
or an existing annotation. No audience outcomes or model calls enter this function.
Given the show's story graph (hookcut/story_graph.py), each cut also gets show-level
context: who is on screen, their archetypes and bonds, core conflict pairs, promo pull,
open loops raised or answered inside the cut, and identity reveals.
"""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from .schema import Beat, Candidate, Episode, EpisodeAnalysis, RunResult


# These English cue lists favor explainable matches over broad coverage.
_FAMILY = r"father|mother|mom|mum|dad|sister|brother|daughter|son|parent|family|sibling"
_PARTNER = r"wife|husband|spouse|bride|groom|fianc[ée]e?|ex-fianc[ée]e?"
_KIN = {"sibling": "brother|sister|sibling", "parent": "father|mother|mom|mum|dad|parent",
        "child": "son|daughter|child", "partner": _PARTNER}
_TERMS = {
    "stakes:family": _FAMILY,
    "stakes:money": r"money|cash|debts?|loans?|mortgage|foreclos\w*|inheritance|bankrupt\w*|wages",
    "stakes:safety": r"kill\w*|death|die[ds]?|dying|dead|murder\w*|bleed\w*|poison\w*|weapons?|burn\w*|funeral",
    "stakes:status": r"rank|reputation|disgrace\w*|throne|titles?|banish\w*",
    "stakes:justice": r"police|judge|trial|sentence[ds]?|arrest\w*|prison|investigat\w*|evidence",
    "emotion:fear": r"fear\w*|afraid|terrifi\w*|panic\w*|nervous\w*|anxious\w*|frantic\w*",
    "emotion:surprise": r"stunned|stuns|shock(?:ed)?|startled|disbelie\w*|caught off guard|gasps?",
    "emotion:anger": r"angry|anger|furi\w*|outrage\w*|enraged|seething",
    "emotion:grief": r"grief|griev\w*|sobs?|sobbing|mourn\w*|heartbroken",
    "emotion:affection": r"tender(?:ly|ness)?|loving(?:ly)?|affection\w*|kiss(?:es|ed|ing)?|embrace[ds]?",
    **{f"relationship:{key}": value for key, value in _KIN.items()},
}
_CUES = {key: re.compile(rf"\b(?:{value})\b", re.I) for key, value in _TERMS.items()}
_QUESTIONS = [
    ("identity", r"\b(who|identity|background)\b"),
    ("motive", r"\b(why|motive|reason)\b"),
    ("danger", r"\b(in time|survive|survival|rescue|escape|kill\w*|threat\w*|danger)\b"),
    ("culpability", r"\b(guilty|responsible|culprit|set the fire|to blame)\b"),
    ("discovery", r"\b(truth|secret|evidence|hid\w*|discover\w*|find out|reveal\w*)\b"),
    ("decision", r"\b(choose|accept|refuse|sign|forgive|agree|decide)\b"),
]
_CLIFF_CUES = [(key, re.compile(pattern, re.I)) for key, pattern in _QUESTIONS]
_ROLE_CUES = {
    "authority": re.compile(r"\b(leader|ruler|officer|guard|judge|enforcer)\b", re.I),
    "adversary": re.compile(r"\b(antagonist|villain|rival|conspirator|extortionist)\b", re.I),
    "ally": re.compile(r"\b(ally|friend|protector|mentor|confidant)\b", re.I),
    "heir": re.compile(r"\b(heir|heiress|successor)\b", re.I),
    "caregiver": re.compile(r"\b(healer|caregiver|nurse|physician|doctor)\b", re.I),
    "captive": re.compile(r"\b(captive|prisoner|imprisoned|hostage)\b", re.I),
}
_LIMITATIONS = [
    "Word and delivery cues cover English and can miss paraphrase, negation, irony and visual subtext.",
    "Hook, cliff, context and character-role annotations inherit the judgment of the saved run.",
    "Named relationships are attributed to a speaker or to the episode's character description. "
    "For character descriptions the cited beat locates the speaker, whose line may not mention that relationship.",
    "The graph covers the supplied cuts. An unmatched cue means that this extractor found no evidence.",
]


def _key(name: str) -> str:
    return " ".join(name.casefold().split())


def _feature_label(feature: str) -> str:
    dimension, value = feature.split(":", 1)
    word = value.replace("_", " ")
    labels = {"hook": f"{word} hook", "cliff": f"question about {word}",
              "stakes": f"{word} stakes", "emotion": f"{word} cue",
              "relationship": f"{word} mentioned", "role": f"{word} role",
              "opening": f"{word} opening beat", "plot": word, "payoff": word,
              "context": f"prior context {word}", "structure": f"{word}-segment cut",
              "tension": f"opening-to-ending suspense {word}",
              "archetype": f"{word} on screen", "pull": f"{word} promo-pull cast on screen",
              "bond": f"{word} bond on screen", "conflict": "core conflict pair on screen",
              "reveal": "identity reveal inside the cut",
              "loop": _LOOP_LABELS.get(value, f"raises a question of {word.removeprefix('opens ')}")}
    return labels[dimension].capitalize()


_LOOP_LABELS = {"spent": "question raised and answered inside the cut", "pays_off": "answers an earlier question",
                "carries_over": "raises a question carried to later episodes"}


class _Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.evidence: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        self.features: dict[str, set[str]] = {}

    def edge(self, source: str, relation: str, target: str, basis: str, *beat_ids: str):
        self.evidence[source, relation, target, basis].update(beat_ids)

    def beat(self, beat: Beat) -> str:
        key = f"beat:{beat.id}"
        self.nodes[key] = {"kind": "beat", "label": beat.id, "episode": beat.episode,
                           "start": beat.start, "end": beat.end, "beat_kind": beat.kind}
        return key

    def feature(self, candidate: Candidate, feature: str, basis: str, *beat_ids: str):
        self.features[candidate.id].add(feature)
        self.nodes[feature] = {"kind": "feature", "dimension": feature.split(":")[0],
                               "label": _feature_label(feature)}
        self.edge(f"clip:{candidate.id}", "has_feature", feature, basis, *beat_ids)


class _Cast:
    """Resolve only unambiguous episode aliases; speaker labels work without analysis."""
    def __init__(self, episode: Episode, analysis: EpisodeAnalysis | None):
        owners: dict[str, set[str]] = defaultdict(set)
        self.names: dict[str, str] = {}
        self.roles: dict[str, list[str]] = {}
        self.role_links: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for char in analysis.characters if analysis else []:
            canonical = _key(char.name)
            if not canonical:
                continue
            self.names[canonical] = char.name.strip()
            for label in [char.name, *char.aliases]:
                if label.strip():
                    owners[_key(label)].add(canonical)
            self.roles[canonical] = [key for key, cue in _ROLE_CUES.items() if cue.search(char.role)]
        for beat in episode.beats:
            if beat.speaker:
                label = _key(beat.speaker)
                if label not in owners:
                    owners[label].add(label)
                    self.names[label] = beat.speaker.strip().title()
        self.ids = {label: next(iter(names)) for label, names in owners.items() if len(names) == 1}
        # Join a speaker to another named person only for explicit, local declarations.
        self.claims = []
        if self.ids:
            names = "|".join(re.escape(n) for n in sorted(self.ids, key=lambda n: (-len(n), n)))
            who = rf"(?P<name>{names})"
            kin = rf"(?P<kin>{'|'.join(_KIN.values())})"
            patterns = (rf"{who}\s+is\s+my\s+{kin}", rf"my\s+{kin}\s+{who}",
                        rf"that(?:'s| is)\s+{who}\.\s+that(?:'s| is)\s+my\s+{kin}")
            self.claims = [re.compile(rf"\b{p}\b", re.I) for p in patterns]
            modifier = r"(?:(?:new|true|only|biological|younger|older|elder|cruel|cold|beloved)\s+)*"
            possessive = re.compile(rf"\b{who}'s\s+{modifier}{kin}\b", re.I)
            for char in analysis.characters if analysis else []:
                actor = _key(char.name)
                for match in possessive.finditer(char.role.replace("’", "'")):
                    target = self.ids[_key(match["name"])]
                    if actor != target:
                        self.role_links[actor].append((self.kin_group(match["kin"]), target))

    @staticmethod
    def kin_group(term: str) -> str:
        return next(key for key, words in _KIN.items() if re.fullmatch(words, term, re.I))

    def add_speaker(self, graph: _Graph, candidate: Candidate, beat: Beat):
        speaker = self.ids.get(_key(beat.speaker or ""))
        if not speaker:
            return
        source = f"character:{speaker}"
        graph.nodes[source] = {"kind": "character", "label": self.names[speaker]}
        graph.edge(f"beat:{beat.id}", "spoken_by", source, "speaker label", beat.id)
        for role in self.roles.get(speaker, []):
            feature = f"role:{role}"
            graph.feature(candidate, feature, "episode character description", beat.id)
            graph.edge(source, "described_as", feature, "episode character description", beat.id)
        for kin, target in self.role_links.get(speaker, []):
            node = f"character:{target}"
            graph.nodes[node] = {"kind": "character", "label": self.names[target]}
            graph.edge(source, f"described_as_{kin}_of", node, "episode character description", beat.id)
        for pattern in self.claims:
            for claim in pattern.finditer(beat.text.replace("’", "'")):
                target = self.ids[_key(claim["name"])]
                if target == speaker:
                    continue
                kin = self.kin_group(claim["kin"])
                node = f"character:{target}"
                graph.nodes[node] = {"kind": "character", "label": self.names[target]}
                graph.edge(source, f"calls_{kin}", node, "spoken relationship claim", beat.id)


class _Story:
    """Show-level context from a story graph, applied one cut at a time (only beats inside the cut count)."""

    def __init__(self, result: RunResult, story):
        from .story_graph import StoryGraph, beat_cast, beats_inside
        self.g = story if isinstance(story, StoryGraph) else StoryGraph.model_validate(story)
        self.on = beat_cast(result, self.g)
        self.chars = {c.id: c for c in self.g.characters}
        self.bonds = {frozenset((e.a, e.b)): e for e in self.g.relationships}
        # pull:high/medium/low = the cast's summed promo pull, split at this show's cut terciles
        episodes = {ep.number: ep for ep in result.episodes}
        sums = sorted(self.pull({c for b in beats_inside(episodes, cut.segments) for c in self.on.get(b.id, [])})
                      for cut in result.candidates)
        self.terciles = (sums[len(sums) // 3], sums[2 * len(sums) // 3]) if sums else (0.0, 0.0)
        self.opens, self.answers = defaultdict(list), defaultdict(list)
        for loop in self.g.loops:
            if loop.opened_beat:
                self.opens[loop.opened_beat].append(loop)
            if loop.answered_beat:
                self.answers[loop.answered_beat].append(loop)
        self.reveals = {b for a in self.g.alerts if a.type == "identity_reveal" for b in a.beat_ids}

    def pull(self, ids) -> float:
        return sum(self.chars[c].promo_pull for c in ids if c in self.chars)

    def add(self, graph: _Graph, candidate: Candidate, inside: dict[str, Beat]):
        clip = f"clip:{candidate.id}"
        present: dict[str, list[str]] = {}
        for bid in inside:
            for cid in self.on.get(bid, []):
                if cid in self.chars:
                    present.setdefault(cid, []).append(bid)
        for cid, ids in present.items():
            c = self.chars[cid]
            graph.nodes[f"character:{cid}"] = {"kind": "character", "label": c.name, "archetype": c.archetype,
                                               "promo_pull": c.promo_pull}
            graph.edge(clip, "features_character", f"character:{cid}", "story graph cast", *ids)
            if c.archetype and c.archetype != "other":
                graph.feature(candidate, "archetype:" + re.sub(r"[^a-z0-9]+", "_", c.archetype),
                              "story graph archetype", ids[0])
        if present and self.pull(present) > 0:
            total, (low, high) = self.pull(present), self.terciles
            level = "high" if total >= high else "medium" if total >= low else "low"
            star = max(present, key=lambda x: self.chars[x].promo_pull)
            graph.feature(candidate, f"pull:{level}", f"story graph promo pull {total:.2f}", present[star][0])
        last_ep = max(s.episode for s in candidate.segments)
        acting = {self.on[bid][0] for bid in inside if self.on.get(bid)}   # speaks or leads an action line
        for a, b in combinations(sorted(present), 2):
            e = self.bonds.get(frozenset((a, b)))
            if e is None:
                continue
            evidence = present[a][:1] + present[b][:1]
            label = next((ch.to for ch in reversed(e.changes) if ch.ep <= last_ep), e.label) or "co_present"
            graph.edge(f"character:{e.a}", "story_" + re.sub(r"[^a-z0-9]+", "_", label), f"character:{e.b}",
                       "story graph relationship", *evidence)
            if e.kind:
                graph.feature(candidate, f"bond:{e.kind}", "story graph relationship", *evidence)
            if e.core and a in acting and b in acting:
                graph.feature(candidate, "conflict:core_pair",
                              f"story graph conflict intensity {e.conflict_intensity:.0f}/100", *evidence)
        for bid in inside:
            for loop in self.opens.get(bid, []):
                graph.feature(candidate, f"loop:opens_{loop.kind}", "story graph open loop", bid)
                if loop.answered_beat in inside:
                    graph.feature(candidate, "loop:spent", "story graph open loop", bid, loop.answered_beat)
                elif loop.scope != "scene" and (loop.answered_ep or 10**6) > loop.opened_ep:
                    graph.feature(candidate, "loop:carries_over", "story graph open loop", bid)
            if any(loop.opened_beat not in inside for loop in self.answers.get(bid, [])):
                graph.feature(candidate, "loop:pays_off", "story graph open loop", bid)
            if bid in self.reveals:
                graph.feature(candidate, "reveal:inside", "story graph identity-reveal alert", bid)


def build_genome(result: RunResult, graph=None) -> dict:
    """Return JSON-ready nodes, edges, and binary feature IDs keyed by candidate ID.
    graph: optional story_graph.StoryGraph (or its JSON dict) for show-level context per cut."""
    story = _Story(result, graph) if graph is not None else None
    graph = _Graph()
    episodes = {ep.number: ep for ep in result.episodes}
    analyses = {a.episode: a for a in result.analyses}
    beats = {b.id: b for ep in result.episodes for b in ep.beats}
    scores = {s.beat_id: s for a in result.analyses for s in a.beats}
    casts = {n: _Cast(ep, analyses.get(n)) for n, ep in episodes.items()}
    for candidate in result.candidates:
        clip = f"clip:{candidate.id}"
        graph.nodes[clip] = {"kind": "clip", "label": candidate.id}
        graph.features[candidate.id] = set()
        inside = {}
        for seg in candidate.segments:
            ep = episodes.get(seg.episode)
            if ep is None:
                continue
            # Use only complete beats so a cue just past the cut cannot leak into its vector.
            included = [b for b in ep.beats if seg.start <= b.start < seg.end and b.end <= seg.end]
            inside.update((b.id, b) for b in included)
            episode_key = f"episode:{ep.number}"
            graph.nodes[episode_key] = {"kind": "episode", "label": f"Episode {ep.number}"}
            graph.edge(clip, "drawn_from", episode_key, "cut timestamps",
                       *(b.id for b in included[:1] + included[-1:]))
        for beat in inside.values():
            graph.edge(clip, "contains", graph.beat(beat), "cut timestamps", beat.id)
            casts[beat.episode].add_speaker(graph, candidate, beat)
            if beat.kind == "sfx":
                continue
            wording = f"{beat.delivery or ''} {beat.text}"
            for feature, cue in _CUES.items():
                if cue.search(wording):
                    graph.feature(candidate, feature, "script wording or delivery", beat.id)

        hook = inside.get(candidate.hook_beat_id)
        cliff = inside.get(candidate.cliff_beat_id)
        anchors = [b.id for b in [hook, cliff] if b]
        if inside:
            evidence = anchors or [next(iter(inside))]
            graph.feature(candidate, f"structure:{candidate.kind}", "cut timestamps", *evidence)
            graph.feature(candidate, "context:required" if candidate.needs_context else "context:unflagged",
                          "clip annotation", *evidence)
        if hook:
            graph.edge(clip, "opens_on", graph.beat(hook), "hook annotation", hook.id)
            graph.feature(candidate, f"opening:{hook.kind}", "script beat type", hook.id)
            if candidate.hook_type:
                graph.feature(candidate, f"hook:{candidate.hook_type}", "hook annotation", hook.id)
            a = analyses.get(hook.episode)
            if a and a.opening_hook.beat_id == hook.id:
                graph.feature(candidate, "plot:episode_opening", "episode annotation", hook.id)
        if cliff:
            graph.edge(clip, "cuts_after", graph.beat(cliff), "cliffhanger annotation", cliff.id)
            if candidate.open_question.strip():
                mechanism = next((k for k, cue in _CLIFF_CUES if cue.search(candidate.open_question)), "outcome")
                graph.feature(candidate, f"cliff:{mechanism}", "annotated open question", cliff.id)
            a = analyses.get(cliff.episode)
            if a and a.ending_cliffhanger.beat_id == cliff.id:
                graph.feature(candidate, "plot:episode_ending", "episode annotation", cliff.id)
            score = scores.get(cliff.id)
            answer = beats.get(score.resolved_by) if score and score.resolved_by else None
            if answer and answer.episode == cliff.episode and answer.start > cliff.start:
                graph.edge(graph.beat(cliff), "answered_by", graph.beat(answer),
                           "episode annotation", cliff.id, answer.id)
                # A partially visible answer may already give away the payoff.
                visible = any(s.episode == answer.episode and s.start < answer.end and answer.start < s.end
                              for s in candidate.segments)
                payoff = "inside_cut" if visible else "outside_cut"
                graph.feature(candidate, f"payoff:{payoff}", "annotated answer and cut timestamps", cliff.id, answer.id)
        if hook and cliff and hook.id != cliff.id:
            first, last = scores.get(hook.id), scores.get(cliff.id)
            # Compare like with like. Both endpoints must have an explicit positive
            # cliff score; a beat omitted from the analysis supplies no observation.
            if first and last and first.cliff and last.cliff:
                delta = last.cliff - first.cliff
                trend = "rises" if delta >= 2 else "falls" if delta <= -2 else "holds_steady"
                graph.feature(candidate, f"tension:{trend}",
                              f"annotated cliff scores {first.cliff}/5 to {last.cliff}/5", hook.id, cliff.id)
        if story is not None and inside:
            story.add(graph, candidate, inside)

    return {"features": {key: sorted(values) for key, values in graph.features.items()},
            "nodes": graph.nodes,
            "edges": [{"source": src, "relation": rel, "target": dst, "basis": basis, "beat_ids": sorted(ids)}
                      for (src, rel, dst, basis), ids in sorted(graph.evidence.items())],
            "limitations": list(_LIMITATIONS)}
