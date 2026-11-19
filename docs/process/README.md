# How the build was run

HookCut was built by one person directing AI coding agents. A lead agent (Claude Code) owned the
data models, the pipeline and the docs; it wrote a brief for each worker and kept a shared team board. Workers ran
in parallel, each owning a fixed set of files and a fixed function signature, and appended one status line to the
board when done. Mistakes the agents (or the models inside the tool) made were logged in
[../ai_notes.md](../ai_notes.md).

| File | What it is |
|---|---|
| [team_board.md](team_board.md) | The board every worker read first: goal, stage contracts, file ownership, ground rules, decisions, and the workers' status lines. |
| [app_worker.md](app_worker.md) | Brief for the first Streamlit app, the synthetic sample show and the offline demo run (OpenAI Codex). |
| [adda_worker.md](adda_worker.md) | Brief for the AI Adda layer: tool registry, router, quota ledger, workflows, MCP server (OpenAI Codex). |
| [codex_review.md](codex_review.md) | Brief for an independent "senior promo editor" review of the top 10 picks (OpenAI Codex). |

The other workers (analyst, editor, evaluator, genome, engagement) were Claude Code subagents briefed the same way;
their contracts and results are on the board. The client's screenplays are confidential: names and quotes from
them have been replaced with placeholders such as "[the rival sister's reveal]" in these files.
