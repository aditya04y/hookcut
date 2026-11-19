"""Rebuild the hand-authored sample run with .venv/bin/python app/fixtures/make_demo_run.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hookcut.parse import parse_folder
from hookcut.report import episode_report
from hookcut.schema import (
    BeatScore, Candidate, ColdViewerVerdict, EpisodeAnalysis, EpisodeEdge, Pickup, Promo, RunResult, Segment,
)
from hookcut.textfmt import ep_tc


def build_demo() -> RunResult:
    episodes = parse_folder(ROOT / "samples")
    by_number = {episode.number: episode for episode in episodes}

    def beat(episode, index):
        return by_number[episode].beats[index - 1]

    def segment(episode, first, last):
        start, end = beat(episode, first), beat(episode, last)
        return Segment(episode=episode, start=start.start, end=end.end, first_beat_id=start.id, last_beat_id=end.id)

    def duration(segments):
        return round(sum(part.end - part.start for part in segments), 2)

    score_notes = {
        1: [
            (1, 5, 2, "reveal", "The ex has bought a stake in her future.", None),
            (5, 4, 3, "mystery", "Nora says the true cause of her mother's death is still hidden.", "What started the fire?"),
            (7, 4, 2, "humiliation", "The abandoned wedding makes the cost of trusting him personal.", None),
            (10, 3, 4, "threat", "The developer offers a way out with a price.", "Will Nora sign to get free of Caleb?"),
            (12, 3, 3, "shocking_line", "Nora calls out the takeover.", "Why didn't he come to her directly?"),
            (15, 3, 3, "threat", "Nora commits to signing.", "Can Caleb change her mind?"),
            (19, 4, 5, "mystery", "The demolition order suddenly looks like a cover-up.", "Why is Sloan targeting Booth 2?"),
        ],
        2: [
            (1, 4, 1, "threat", "Nora walks away from a sale with a deadline.", None),
            (7, 3, 1, "mystery", "The lipstick-marked cup gives the empty room a presence.", None),
            (11, 4, 3, "mystery", "The key exposes a secret between Caleb and Nora's mom.", "Why did Rosa trust Caleb?"),
            (13, 3, 3, "mystery", "The evidence is tangible.", "What did Rosa hide on the recorder?"),
            (15, 3, 3, "mystery", "Caleb begs her to hear the entire recording.", "What is Caleb afraid she'll hear?"),
            (18, 5, 5, "reveal", "The recording accuses the man standing beside Nora.", "Did Caleb set the fire?"),
            (19, 1, 4, None, "Nora recoils without giving him time to explain.", "Will Nora hear Caleb out?"),
        ],
        3: [
            (1, 5, 3, "reveal", "Caleb admits taking the money straight away.", "Who started the fire?"),
            (4, 4, 4, "reveal", "The voice on the tape is his father.", "Why was his father paying him?"),
            (7, 3, 3, "shocking_line", "Nora names the lie she grew up believing.", "Why did Caleb keep quiet?"),
            (8, 4, 4, "threat", "Caleb says his father threatened Nora.", "Will Caleb risk challenging his father now?"),
            (11, 3, 2, "mystery", "The security footage shows a late-night arrival.", None),
            (12, 4, 3, "reveal", "The gas can makes the accusation concrete.", "Who let Victor into the cinema?"),
            (13, 3, 1, "mystery", "Nora records the plate on her phone.", None),
            (17, 4, 5, "reveal", "The footage implicates Nora's brother.", "Why did Miles let Victor in?"),
            (19, 4, 5, "shocking_line", "Miles appears and challenges the footage.", "Who called for help, and what did they know?"),
        ],
        4: [
            (1, 3, 1, "shocking_line", "Caleb answers the accusation immediately.", None),
            (5, 4, 4, "reveal", "Victor lied to the emergency dispatcher.", "Was Rosa really in the building?"),
            (8, 3, 2, "mystery", "Caleb's father is calling as they uncover the fire.", None),
            (11, 4, 2, "shocking_line", "A funeral memory makes the call feel threatening.", None),
            (14, 5, 5, "reveal", "Victor offers proof Nora's mother is alive.", "Is Victor telling the truth?"),
            (18, 5, 5, "threat", "Rosa's voice confirms the impossible and gives a warning.", "What will happen if Caleb looks outside?"),
            (19, 2, 5, None, "Caleb reaches for the blind anyway.", "Can Nora stop Caleb in time?"),
        ],
    }
    openings = [
        (1, 5, "Nora's ex bought the loan on the theater she is trying to save."),
        (1, 4, "Nora puts the sale at risk to follow the evidence."),
        (1, 5, "Caleb admits taking the money while Nora is still at the door."),
        (1, 3, "Caleb answers Miles before anyone can leave the booth."),
    ]
    endings = [
        (19, 5, "Caleb asks why the buyer wants Booth 2 demolished first."),
        (18, 5, "The recording implicates Caleb and Nora takes a step back."),
        (19, 5, "Miles appears in the doorway and challenges their conclusion."),
        (19, 5, "Nora lunges as Caleb reaches for the blind."),
    ]
    pickup_notes = [
        "First episode.",
        "Nora refuses to sign and asks to go straight to Booth 2.",
        "Caleb answers the accusation with Nora still reaching for the door.",
        "Caleb names Miles as the caller at once.",
    ]
    analyses = []
    for episode in episodes:
        number = episode.number
        scores = [
            BeatScore(beat_id=beat(number, index).id, hook=hook, cliff=cliff, hook_type=kind,
                      reason=reason, open_question=question,
                      spoiler=3 if number == 4 and index in {14, 18, 19} else 0)
            for index, hook, cliff, kind, reason, question in score_notes[number]
        ]
        opening, ending = openings[number - 1], endings[number - 1]
        ending_id = beat(number, ending[0]).id
        ending_score = next(score for score in scores if score.beat_id == ending_id)
        analyses.append(EpisodeAnalysis(
            episode=number, summary=episode.logline, beats=scores,
            opening_hook=EpisodeEdge(beat_id=beat(number, opening[0]).id, score=opening[1], note=opening[2]),
            ending_cliffhanger=EpisodeEdge(beat_id=ending_id, score=ending[1], note=ending[2]),
            ending_open_question=ending_score.open_question or "",
            pickup=Pickup(status="not_applicable" if number == 1 else "immediate",
                          delay_s=None if number == 1 else 0.0,
                          first_new_beat_id=None if number == 1 else beat(number, 1).id,
                          note=pickup_notes[number - 1]),
            model="hand-authored", prompt_version="sample-v1",
        ))

    beat_scores = {score.beat_id: score for analysis in analyses for score in analysis.beats}
    candidate_specs = [
        ([(1, 1, 10)], (1, 1), (1, 10), 4.7, []),
        ([(1, 7, 19)], (1, 7), (1, 19), 3.9, ["Which earlier deal put Nora in this position?"]),
        ([(2, 11, 18)], (2, 11), (2, 18), 4.6, []),
        ([(2, 15, 19), (3, 1, 8)], (2, 15), (3, 8), 4.0, []),
        ([(2, 11, 13), (3, 1, 4)], (2, 11), (3, 4), 3.4, ["What did the skipped recording say?"]),
        ([(3, 12, 19)], (3, 12), (3, 19), 4.4, []),
        ([(4, 11, 19)], (4, 11), (4, 19), 4.8, []),
        ([(3, 13, 19)], (3, 13), (3, 19), 3.1, ["Who is already on the footage?"]),
    ]
    candidates = []
    for index, (parts, hook_at, cliff_at, prelim, context) in enumerate(candidate_specs, 1):
        segments = [segment(*part) for part in parts]
        hook, cliff = beat_scores[beat(*hook_at).id], beat_scores[beat(*cliff_at).id]
        candidates.append(Candidate(
            id=f"C{index:04d}", kind="single" if len(segments) == 1 else "multi", segments=segments,
            duration=duration(segments), hook_beat_id=hook.beat_id, cliff_beat_id=cliff.beat_id,
            hook_score=hook.hook, cliff_score=cliff.cliff, hook_type=hook.hook_type,
            open_question=cliff.open_question or "", spoiler=cliff.spoiler, needs_context=context,
            prelim_score=prelim,
        ))

    verdict_specs = [
        (1, "pass", 5, 4, True, "Nora's ex-fiancé bought the loan on the theater she plans to sell.",
         "She has to choose between his help and the buyer's clean exit.", []),
        (2, "borderline", 3, 4, True, "A woman is arguing with an ex while a buyer waits nearby.",
         "The couple disagrees over signing away a cinema.", ["The mortgage surprise happens before the clip."]),
        (3, "pass", 4, 5, True, "Nora and Caleb are listening to evidence her mother hid about a fire.",
         "The evidence Caleb brought up the stairs implicates him.", []),
        (4, "pass", 4, 4, True, "A recording accuses Caleb of arson and he admits taking the money.",
         "Nora wants to know why Caleb hid his father's threat.", ["Their former engagement takes time to infer."]),
        (5, "fail", 1, 2, False, "Nora confronts Caleb about an unspecified payment.",
         "The cut skips the accusation he is answering.", ["What did the recording say?", "What was the money for?"]),
        (6, "pass", 4, 5, True, "Two people watch footage that links both their families to a fire.",
         "Nora can send the footage, but her brother appears and challenges her.", []),
        (7, "pass", 4, 5, True, "A woman who buried her mother gets a live call from her.",
         "She has seconds to stop Caleb from looking out the window.", []),
    ]
    verdicts = [
        ColdViewerVerdict(
            candidate_id=candidates[number - 1].id, verdict=verdict, premise=premise, conflict=conflict,
            open_question=candidates[number - 1].open_question, confusions=confusions, hook_lands=lands,
            standalone=standalone, keep_watching=keep_watching, model="hand-authored",
        )
        for number, verdict, standalone, keep_watching, lands, premise, conflict, confusions in verdict_specs
    ]
    by_candidate = {verdict.candidate_id: verdict for verdict in verdicts}
    promo_specs = [
        (1, 1, "Her ex buying the mortgage makes the breakup and the sale collide in the opening line.",
         [], ("Open on the fire", [(1, 5, 10)], "Start with Nora naming the mystery around her mother's death.")),
        (3, 1, "The question about the key turns into a direct accusation against the man who brought it.",
         [], ("Open on the recorder", [(2, 13, 18)], "Start on the discovery in the reel tin.")),
        (6, 2, "A gas can starts the search for a culprit. The man on the footage walks into the room.",
         [], ("Open on the security footage", [(3, 11, 19)], "Let the black pickup arrive before Caleb names his father.")),
        (4, 2, "Caleb's immediate admission carries the accusation across the episode break.",
         ["Cross-episode cut"],
         ("Open on the accusation", [(2, 18, 19), (3, 1, 8)], "Skip the recorder setup and start on Victor's voice.")),
        (7, 3, "A funeral memory sets up the shock of hearing Rosa alive, followed by a warning Caleb is already ignoring.",
         ["Major spoiler"], ("Open on Victor's offer", [(4, 14, 19)], "Start on the call and let the headlights confirm the car.")),
    ]
    by_id = {entry.id: entry for episode in episodes for entry in episode.beats}
    promos = []
    for rank, (number, wave, why, flags, variant) in enumerate(promo_specs, 1):
        candidate = candidates[number - 1]
        hook, cliff = by_id[candidate.hook_beat_id], by_id[candidate.cliff_beat_id]
        label, alternative, note = variant
        alt_segments = [segment(*part) for part in alternative]
        promos.append(Promo(
            rank=rank, candidate_id=candidate.id, kind=candidate.kind, segments=candidate.segments,
            duration=candidate.duration, hook_time=ep_tc(hook.episode, hook.start), hook_text=hook.text,
            cliffhanger_time=ep_tc(cliff.episode, cliff.start), cliffhanger_text=cliff.text,
            open_question=candidate.open_question, why_it_works=why, post_wave=wave,
            flags=flags, cold_viewer=by_candidate[candidate.id],
            scores={"hook": candidate.hook_score, "cliff": candidate.cliff_score},
            variants=[{"label": label, "segments": [part.model_dump() for part in alt_segments],
                       "duration": duration(alt_segments), "hook_beat_id": alt_segments[0].first_beat_id, "note": note}],
        ))

    # Illustrative usage lets the demo show stage totals without making any requests.
    log = [
        {"stage": "demo", "item": "note", "message": "Hand-authored run with illustrative usage figures."},
        {"stage": "analyze", "item": "E01", "provider": "gemini", "model": "gemini-3.8-flash",
         "input_tokens": 2050, "output_tokens": 930, "latency_s": 9.2, "cached": False},
        {"stage": "analyze", "item": "E02", "provider": "gemini", "model": "gemini-3.8-flash",
         "input_tokens": 1900, "output_tokens": 880, "latency_s": 0.01, "cached": True},
        {"stage": "judge", "item": "C0001", "provider": "groq", "model": "qwen/qwen3.8-27b",
         "input_tokens": 680, "output_tokens": 156, "latency_s": 0.9, "cached": False},
        {"stage": "writer", "item": "shortlist", "provider": "groq", "model": "openai/gpt-oss-120b",
         "input_tokens": 1950, "output_tokens": 320, "latency_s": 1.2, "cached": False},
    ]
    return RunResult(show="The Marlowe After Midnight", created_at="2026-11-19T00:00:00",
                     config={"fixture": "hand-authored"}, episodes=episodes, analyses=analyses,
                     candidates=candidates, verdicts=verdicts, promos=promos,
                     episode_report=episode_report(episodes, analyses), run_log=log)


if __name__ == "__main__":
    Path(__file__).with_name("demo_run.json").write_text(build_demo().model_dump_json(indent=2) + "\n", encoding="utf-8")
