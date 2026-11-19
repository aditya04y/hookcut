"""Read-only Streamlit view of the Adda catalog and run provenance."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from hookcut.adda.router import TIERS, Router
from hookcut.adda.runner import list_workflows, model_usage
from hookcut.adda.tools import list_tools
from hookcut.config import load_config
from hookcut.schema import RunResult

TEAMS = {"promo": "Promo", "script": "Script", "audience": "Audience", "platform": "Platform"}


def _show_usage(result: RunResult | None, run_dir: str | Path | None) -> None:
    manifest_path = Path(run_dir) / "run_manifest.json" if run_dir is not None else None
    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st.caption("The run manifest could not be read.")
        else:
            st.subheader("Run steps")
            rows = []
            for step in manifest["steps"]:
                reasons = {option["spec"]: option["reason"] for option in step.get("routes", [])}
                for usage in step["usage"] or [{}]:
                    model = usage.get("model")
                    rows.append({
                        "Tool": step["tool"], "Status": step["status"], "Model": model or "—",
                        "Seconds": step["seconds"], "Estimated USD": usage.get("est_cost_usd", 0.0),
                        "Reason": reasons.get(model, step["reason"]),
                    })
            st.dataframe(rows, hide_index=True)
            return
    if result is not None and result.run_log:
        st.subheader("Model usage")
        prices = load_config().get("prices_usd_per_mtok") or {}
        usage = model_usage(result.run_log, prices)
        st.dataframe([{"Task": row["stage"], "Model": row["model"] or "—", "Calls": row["calls"],
                       "Cached": row["cached"], "Errors": row["errors"], "Seconds": row["latency_s"],
                       "Estimated USD": row["est_cost_usd"]} for row in usage], hide_index=True)


def show_toolkit(result: RunResult | None = None, *, run_dir: str | Path | None = None) -> None:
    """Render tool cards, model routing, daily quotas, and saved run provenance."""
    st.header("AI Adda")
    _show_usage(result, run_dir)

    st.subheader("Tools by team")
    tools = list_tools()
    for key, team in TEAMS.items():
        with st.expander(team):
            for tool in (card for card in tools if card["team"] == key):
                st.markdown(f"**{tool['name']}**")
                st.write(tool["description"])
                st.caption(" · ".join(filter(None, [tool["uses_model"], tool["cost_hint"]])))

    st.subheader("Model routing")
    router = Router()
    quality = st.selectbox("Routing tier", TIERS.keys(), key="adda_routing_tier")
    st.caption("Prices are estimated USD per million tokens. Daily requests are planning figures.")
    for task in router.tasks:
        with st.expander(task.title()):
            choices = router.decisions(task, quality=quality)
            st.dataframe([
                {"Model": row["spec"], "Family": row["family"], "Tier": row["tier"], "Eligible": row["eligible"],
                 "Free requests/day": row["free_daily_requests"], "Input USD": row["price_in"],
                 "Output USD": row["price_out"], "Latency": row["latency_hint"], "Reason": row["reason"]}
                for row in choices
            ], hide_index=True)
    st.caption("In a workflow, judging excludes every family that contributed to episode analysis.")

    st.subheader("Daily quota")
    status = router.ledger.status()
    if status:
        st.dataframe([{"Model": spec, "Resets (UTC)": row["reset_at"], "Reason": row["reason"]}
                      for spec, row in status.items()], hide_index=True)
    else:
        st.caption("No daily quota exhaustion recorded.")

    st.subheader("Workflows")
    by_name = {tool["name"]: tool["team"] for tool in tools}
    for workflow in list_workflows().values():
        with st.expander(workflow.name):
            st.write(workflow.description)
            st.table([{"Tool": step.tool, "Team": TEAMS.get(by_name.get(step.tool), "—"),
                       "Optional": step.optional, "File": step.artifact or "—"} for step in workflow.steps])
