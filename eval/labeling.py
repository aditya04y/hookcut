"""Write the PRIVATE human-labeling packet for the hold-out episodes.

  .venv/bin/python -m eval.labeling --input data/pack [--out outputs/labeling]

outputs/ is git-ignored: the packet contains Kuku's script text. It holds one beat listing per hold-out episode
(timestamps, speakers, lines; camera directions shortened) and labels.csv, a copy of eval/labels_template.csv
(which is committed and has no script text). The packet deliberately does NOT show HookCut's picks: label blind.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

from hookcut.schema import Episode
from hookcut.textfmt import render_beats, tc

from . import HOLDOUT_EPISODES
from .common import ROOT

TEMPLATE = ROOT / "eval" / "labels_template.csv"
CAMERA_CHARS = 60

INSTRUCTIONS = """\
PRIVATE: contains Kuku script text. Do not commit, paste or share.

How to label (do this BEFORE looking at HookCut's picks for these episodes):
1. Read the episode once, as a promo editor.
2. Pick the single best promo window: 20-60 s, a hook in its first 3 s (a line or action that stops a
   scroller), ends on an unresolved moment, makes sense to someone who never saw the show.
3. In labels.csv write best_start / best_end in episode time (mm:ss.xx, as below, or seconds). If a second
   window is just as good, add another row for the same episode.
4. hook_note / cliff_note: a few words in your own words (no quotes from the script). labeler: your initials.
"""


def episode_sheet(ep: Episode, camera_chars: int = CAMERA_CHARS) -> str:
    head = [f"# Episode {ep.number}  ({tc(ep.duration)} long, {len(ep.beats)} beats)", "", INSTRUCTIONS]
    return "\n".join(head + ["```", render_beats(ep.beats, camera_chars=camera_chars), "```", ""])


def write_packet(episodes: Iterable[Episode], out_dir: str | Path = ROOT / "outputs" / "labeling",
                 holdout: Sequence[int] = HOLDOUT_EPISODES, camera_chars: int = CAMERA_CHARS) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_n = {e.number: e for e in episodes}
    written = []
    for n in holdout:
        if n in by_n:
            p = out / f"E{n:02d}.md"
            p.write_text(episode_sheet(by_n[n], camera_chars))
            written.append(p)
    labels = out / "labels.csv"
    if not labels.exists() and TEMPLATE.exists():   # never overwrite labels already filled in
        shutil.copy(TEMPLATE, labels)
    missing = [n for n in holdout if n not in by_n]
    (out / "README.md").write_text(INSTRUCTIONS + f"\nEpisodes: {', '.join(p.stem for p in written)}\n"
                                   + (f"Missing from input: {missing}\n" if missing else "")
                                   + "Score afterwards: .venv/bin/python -m eval.score_labels --run "
                                     "outputs/<run>/result.json --labels outputs/labeling/labels.csv\n")
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    from hookcut.pipeline import load_episodes
    ap = argparse.ArgumentParser(description="Write the private labeling packet for the hold-out episodes.")
    ap.add_argument("--input", default=str(ROOT / "data" / "pack"), help="episode folder (or a run's result.json)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "labeling"))
    a = ap.parse_args(argv)
    if a.input.endswith(".json"):
        from .common import load_run
        eps = load_run(a.input).episodes
    else:
        eps = load_episodes(a.input)
    for p in write_packet(eps, a.out):
        print(p)
    print(f"fill in {Path(a.out) / 'labels.csv'}")


if __name__ == "__main__":
    main()
