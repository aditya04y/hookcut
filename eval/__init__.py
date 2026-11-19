"""HookCut evaluation harness (pre-registered in eval/README.md before any tuning).

Constants here are the protocol; change them only with a dated note in eval/README.md.
"""

HOLDOUT_EPISODES = (5, 12, 19, 27, 33, 38, 41, 46, 48, 50)   # human-labelled, never used for prompt tuning
TUNING_EPISODES = (1, 23, 44)                                 # the analyst tunes prompts only on these
MAX_JUDGE_CALLS = 70                                          # hard cap per run_eval invocation
HIT_IOU = 0.3                                                 # label hit: temporal IoU >= 0.3, or ...
HIT_TOL_S = 3.0                                               # ... hook and cliff each within +-3 s
