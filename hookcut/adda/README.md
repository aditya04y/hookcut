# HookCut AI Adda

Adda exposes HookCut’s screenplay and audience tools to multiple teams. Each workflow connects tool outputs to later
inputs through its YAML definition. For example, `script_review` uses the promo team’s `analyze_episodes` tool.

## CLI

Run from the repository root.

```sh
.venv/bin/python -m hookcut.adda list
.venv/bin/python -m hookcut.adda route analyze --quality quality
.venv/bin/python -m hookcut.adda route judge --exclude-family gemini
.venv/bin/python -m hookcut.adda run quick_check --input samples --out outputs/quick-check
.venv/bin/python -m hookcut.adda run audience_update --input app/fixtures/demo_run.json --out outputs/audience --simulate
.venv/bin/python -m hookcut.adda run script_review --input samples --out outputs/script-review
.venv/bin/python -m hookcut.adda run promo_finder --input samples --out outputs/adda-promos --quality budget
```

`quick_check` and `audience_update` run locally. Supply `--events path/to/impressions.csv` to update the audience
ranking from imported activity. The script and promo workflows use credentials from HookCut’s existing configuration.

Routes put the requested tier first, then follow the remaining tier order and each tier’s YAML ranking. The judge
excludes the families that produced episode analyses in the same run. `--max-cost` is a USD price cap for 1 million
input tokens plus 1 million output tokens. Free-tier counts are planning figures. Daily limits reported by providers
are kept in `.cache/quota.json` through the reported retry time, with a 24-hour default when no retry time is supplied.

Every workflow writes `run_manifest.json` with step timings, model choices and reasons, recorded model usage,
estimated paid cost, and output paths. Cached calls are excluded from estimated cost.

## MCP over stdio

Use this entry in a local MCP client configuration. Replace both paths with your checkout path.

```json
{
  "mcpServers": {
    "hookcut": {
      "command": "/absolute/path/hookcut/.venv/bin/python",
      "args": ["-m", "hookcut.adda.mcp_server"],
      "env": {
        "PYTHONPATH": "/absolute/path/hookcut"
      }
    }
  }
}
```

Set provider keys in the server's environment for workflows that call models. The MCP server exposes `list_tools`,
`route`, `run_workflow`, `episode_report`, and `promos`. Use absolute paths when calling it from another working
directory. `run_workflow` writes artifacts to its supplied `out_dir`.

To show the catalog and provenance in Streamlit, call `show_toolkit(result, run_dir=folder)` in the toolkit tab.
It can also show the catalog by itself or read the model log embedded in a saved `RunResult`.
