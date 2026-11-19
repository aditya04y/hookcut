You are the "adda" worker on HookCut, a submission for Kuku's AI Generalist hiring assignment (Problem 1: find
ranked 20–60 s promo clips in microdrama screenplays). The lead is another AI agent; two other workers ("genome" and
"engagement") build other layers in parallel. Work only inside the hookcut repo.

READ FIRST: docs/process/team_board.md (whole file, especially the "v2" section at the bottom), hookcut/pipeline.py,
hookcut/config.py, config.yaml, hookcut/llm/registry.py, hookcut/llm/base.py, hookcut/llm/fallback.py,
hookcut/runlog.py, hookcut/schema.py, and skim each stage module (analyze, candidates, judge, rank, writer, report,
validate, export, studio, audience, genome).

WHY: the user wants HookCut to mirror Kuku's own stack. Kuku's AI Adda is an orchestration layer that (1) routes tasks
to the most suitable AI model, (2) gives access to many tools through one interface, (3) connects AI tools to existing
company workflows, and (4) lets teams reuse solutions built by other departments. Build HookCut's version of it.

YOUR FILES (create/edit only these): hookcut/adda/** (new package), app/toolkit_view.py (new), tests/test_adda*.py
(new). Do not edit existing modules — wrap them. Do not git commit. Do not install packages (the disk is nearly full;
.venv has numpy, pandas, pydantic, pyyaml, streamlit, altair, mcp 2.x). You have no API keys: test with fake clients
and never call real model APIs. Do not read data/ or the folder holding the client's original pack (confidential client scripts); use samples/ and app/fixtures/ for anything you run.

BUILD
1. Tool registry (hookcut/adda/tools.py): a Tool dataclass {name, description, inputs (pydantic model), outputs
   (pydantic model or schema type), runs (callable), uses_model (task name or None), team (promo, script, audience,
   platform), cost_hint}. Register the existing capabilities as tools by wrapping their functions: parse_episodes,
   analyze_episodes (task "analyze"), build_candidates, cold_viewer_test (task "judge"), rank_promos,
   write_why_lines (task "writer"), episode_report, season_summary, validate_promos, export_outputs,
   audience studio build (studio.build_studio), and — only if the modules exist when you run — story_graph and
   engagement (import lazily; skip gracefully if missing). `list_tools()` returns their cards.
2. Router (hookcut/adda/router.py): per-task routing policy in hookcut/adda/routing.yaml — for each task a ranked list
   of candidate models with tags {family, tier: quality|balanced|budget, free_daily_requests, price_in, price_out,
   latency_hint}. `Router.choose(task, *, quality="balanced", exclude_families=(), max_cost=None) -> list[str]`
   returns provider:model specs in order, skipping models the QuotaLedger marks exhausted. Rules: the judge never
   shares a family with the analyzer of the same run; budget/quality tiers pick differently; explain each choice in a
   one-line reason. Prices: copy from config.yaml's prices_usd_per_mtok.
3. QuotaLedger (hookcut/adda/quota.py): a small JSON file at .cache/quota.json. `record_error(spec, exc_or_message)`
   recognises daily-quota 429s (e.g. Gemini RESOURCE_EXHAUSTED with quotaId GenerateRequestsPerDayPerProjectPerModel
   and a retryDelay like "80432s"; Groq "tokens per day" messages) and marks the spec exhausted until a reset time;
   `is_exhausted(spec)`, `status()`. Per-minute limits are NOT marked (they clear quickly).
4. Workflows (hookcut/adda/workflows/*.yaml + hookcut/adda/runner.py): YAML DAGs (sequential is fine) composed from
   tools — promo_finder (the full pipeline), script_review (parse → analyze → episode_report → season_summary
   [→ story_graph if present]), audience_update (load a saved run → studio/engagement re-rank), and quick_check
   (parse + validate an input folder without model calls). The runner passes a context dict between steps, asks the
   Router for each model-using step, records per-step timing/model/reason, and writes run_manifest.json. CLI:
   `.venv/bin/python -m hookcut.adda list|route <task>|run <workflow> --input <dir> --out <dir> [--quality ...]`.
   Reuse = any team's workflow can call any registered tool; show that in the YAML (script_review reuses the promo
   team's analyze tool).
5. MCP server (hookcut/adda/mcp_server.py) so any assistant (Claude, Codex, ChatGPT) can use HookCut's tools: mcp
   2.x renamed FastMCP to MCPServer (`from mcp.server.mcpserver import MCPServer`) — read the installed package under
   .venv/lib/python3.11/site-packages/mcp to get the API right. Expose list_tools, route(task, quality), run_workflow
   (name, input_dir, out_dir), episode_report(run_dir), promos(run_dir, top) — read-only on existing runs except
   run_workflow writing to out_dir. Add a short README section in hookcut/adda/README.md with the MCP client config
   snippet (stdio) and the CLI commands.
6. app/toolkit_view.py: show_toolkit(result=None) — tool cards grouped by team, the routing table for each task with
   reasons, quota-ledger status, workflow definitions, and (when a run_manifest.json or run_log is available) which
   model did which step and what it cost. Plain and readable.
7. tests/test_adda*.py (no network): registry lists the core tools; router respects tiers, family exclusion and the
   ledger; ledger parses a Gemini daily-quota 429 payload and a Groq tokens-per-day message, ignores per-minute 429s;
   runner executes quick_check on samples/ and a promo_finder run with fake LLM clients (monkeypatch
   hookcut.config.build_client) end to end; the MCP server module imports and registers its tools. Run
   `.venv/bin/python -m pytest -q` — the whole suite must stay green.

When done, end with a short report: what you built, commands to try, test results, and exactly how the lead should
wire Router/QuotaLedger into hookcut/config.py build_client (a ~10-line patch suggestion). Append one dated line
under "Status" at the bottom of docs/process/team_board.md (the only edit allowed outside your files).
