"""Execute ordered YAML workflow steps against a shared context."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter

from .. import config
from ..llm.registry import load_secrets
from ..runlog import RunLog
from .quota import with_quota
from .router import TIERS, Router
from .tools import ToolRuntime, get_tool

WORKFLOWS = Path(__file__).with_name("workflows")


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    inputs: dict
    save: str
    artifact: str | None = None
    optional: bool = False


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    team: str
    description: str
    steps: list[Step]


def list_workflows() -> dict[str, Workflow]:
    workflows = [Workflow.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
                 for path in sorted(WORKFLOWS.glob("*.yaml"))]
    return {workflow.name: workflow for workflow in workflows}


def model_usage(rows: list[dict], prices: dict) -> list[dict]:
    log = RunLog(prices)
    log.rows = rows
    usage = []
    for stage_model, summary in log.summary()["by_stage_model"].items():
        stage, model = stage_model.split("|", 1)
        if model.startswith("?:"):
            model = model[2:] if ":" in model[2:] else None
        usage.append({"stage": stage, "model": model, **summary})
    return usage


def _client_for(task: str, context: dict, runtime: ToolRuntime, router: Router, quality: str,
                max_cost: float | None, secrets: dict, use_cache: bool, row: dict):
    families = set()
    if task == "judge":
        families = {router.family(analysis.model) for analysis in context.get("analyses", [])
                    if analysis.model and not analysis.model.startswith("failed:")}
        if any(log.get("stage") == "analyze" and log.get("recovered") for log in runtime.log.rows):
            # A recovered first reply has no LLMResult to identify its exact fallback.
            chain = runtime.models.get("analyze", "")
            families.update(router.family(spec) for spec in chain.split(" -> ") if spec)
    decisions = router.decisions(task, quality=quality, exclude_families=families, max_cost=max_cost)
    row["routes"] = [{key: choice[key] for key in ("spec", "eligible", "reason")} for choice in decisions]
    row["excluded_families"] = sorted(families)
    specs = [choice["spec"] for choice in decisions if choice["eligible"]]
    if not specs:
        raise ValueError(f"no eligible models for {task}; see routing decisions in run_manifest.json")
    routed_cfg = copy.deepcopy(runtime.cfg)
    routed_cfg.setdefault("models", {})[task] = specs[0]
    routed_cfg.setdefault("fallbacks", {})[task] = specs[1:]
    client = config.build_client(routed_cfg, task, secrets, use_cache)
    runtime.models[task] = row["configured_models"] = client.spec
    reasons = {choice["spec"]: choice["reason"] for choice in decisions}
    row["reason"] = reasons.get(client.spec.split(" -> ")[0], "Selected by the configured client.")
    return with_quota(client, router.ledger)


def run_workflow(name: str, input_dir: str | Path, out_dir: str | Path, *,
                 quality: str = "balanced", router: Router | None = None, cfg: dict | None = None,
                 max_cost: float | None = None, events: str | Path | None = None,
                 simulate: bool = False, use_cache: bool = True) -> dict:
    """Run steps in YAML order, resolve $context references, and return the saved manifest."""
    workflows = list_workflows()
    if name not in workflows:
        raise ValueError(f"unknown workflow {name!r}; choose from {', '.join(workflows)}")
    if quality not in TIERS:
        raise ValueError(f"unknown quality {quality!r}; choose from {', '.join(TIERS)}")
    if (events is not None or simulate) and name != "audience_update":
        raise ValueError("audience activity is supported by audience_update")
    if events is not None and simulate:
        raise ValueError("choose imported or simulated audience activity")

    workflow = workflows[name]
    router = router if router is not None else Router()
    cfg = cfg if cfg is not None else config.load_config()
    prices = cfg.get("prices_usd_per_mtok") or {}
    runtime = ToolRuntime(cfg, RunLog(prices))
    secrets = None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    context = {"input": Path(input_dir), "out": out, "events": Path(events) if events else None, "simulate": simulate}
    manifest = {"workflow": name, "team": workflow.team, "quality": quality, "status": "failed",
                "started_at": datetime.now(timezone.utc).isoformat(), "steps": [], "artifacts": {}}
    start = perf_counter()
    try:
        for step in workflow.steps:
            row = {"tool": step.tool, "status": "failed", "reason": "Local processing"}
            manifest["steps"].append(row)
            step_start, first_row = perf_counter(), len(runtime.log.rows)
            try:
                tool = get_tool(step.tool)
                if tool is None and step.optional:
                    row.update(status="skipped", reason="Optional tool unavailable.")
                    continue
                if tool is None:
                    raise ValueError(f"unknown tool {step.tool!r} in {name}")
                inputs = tool.inputs.model_validate({
                    key: context[value[1:]] if isinstance(value, str) and value.startswith("$") else value
                    for key, value in step.inputs.items()
                })
                runtime.client = None
                if tool.uses_model:
                    row["task"] = tool.uses_model
                    row["reason"] = "No model selected."
                    if secrets is None:
                        secrets = load_secrets()
                    runtime.client = _client_for(tool.uses_model, context, runtime, router, quality,
                                                 max_cost, secrets, use_cache, row)
                output = tool.runs(inputs, runtime)
                context[step.save] = output
                if step.artifact:
                    path = out / step.artifact
                    path.write_bytes(TypeAdapter(tool.outputs).dump_json(output, indent=2))
                    manifest["artifacts"][step.artifact] = str(path)
                if step.tool == "export_outputs":
                    manifest["artifacts"].update(output)
                row["status"] = "complete"
            except Exception as exc:
                row["error"] = type(exc).__name__
                raise
            finally:
                rows = runtime.log.rows[first_row:]
                row["seconds"] = round(perf_counter() - step_start, 4)
                row["models"] = sorted({f"{log['provider']}:{log['model']}" for log in rows
                                        if log.get("provider") and log.get("model")})
                row["usage"] = model_usage(rows, prices)
                row["est_cost_usd"] = sum(usage["est_cost_usd"] for usage in row["usage"])
        manifest["status"] = "complete"
    finally:
        manifest["seconds"] = round(perf_counter() - start, 4)
        manifest["cost"] = runtime.log.summary()["total"]
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
