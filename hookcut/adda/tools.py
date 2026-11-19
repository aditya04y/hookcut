"""Tool cards and small adapters for HookCut's existing stage functions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .. import analyze, candidates, export, judge, parse, rank, report, studio, validate, writer
from ..llm.base import LLMClient
from ..runlog import RunLog
from ..schema import Candidate, ColdViewerVerdict, Episode, EpisodeAnalysis, Promo, RunResult


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathInput(Inputs):
    path: Path


class EpisodeInputs(Inputs):
    episodes: list[Episode]


class AnalysisInputs(EpisodeInputs):
    analyses: list[EpisodeAnalysis]


class CandidateInputs(EpisodeInputs):
    candidates: list[Candidate]


class RankInputs(AnalysisInputs):
    candidates: list[Candidate]
    verdicts: list[ColdViewerVerdict]


class WriteInputs(EpisodeInputs):
    promos: list[Promo]


class PromoInputs(AnalysisInputs):
    promos: list[Promo]


class ReportInputs(Inputs):
    rows: list[dict]


class ResultInputs(Inputs):
    result: RunResult


class AudienceInputs(ResultInputs):
    events: Path | None = None
    simulate: bool = False


class ExportInputs(ResultInputs):
    validation_issues: list[dict]
    out_dir: Path


class AssembleInputs(AnalysisInputs):
    path: Path
    episode_report: list[dict]
    candidates: list[Candidate] = Field(default_factory=list)
    verdicts: list[ColdViewerVerdict] = Field(default_factory=list)
    promos: list[Promo] = Field(default_factory=list)


class InputValidation(BaseModel):
    episodes: int
    beats: int
    warnings: list[dict]


@dataclass
class ToolRuntime:
    cfg: dict
    log: RunLog
    models: dict[str, str] = field(default_factory=dict)
    client: LLMClient | None = None

    @property
    def promo_cfg(self) -> dict:
        return self.cfg.get("promo") or {}

    def concurrency(self, task: str, default: int) -> int:
        return (self.cfg.get("concurrency") or {}).get(task, default)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    inputs: type[BaseModel]
    outputs: type
    runs: Callable
    uses_model: str | None
    team: str
    cost_hint: str

    def card(self) -> dict:
        return {"name": self.name, "description": self.description, "team": self.team,
                "uses_model": self.uses_model, "cost_hint": self.cost_hint,
                "inputs": self.inputs.model_json_schema(), "outputs": TypeAdapter(self.outputs).json_schema()}


def _parse(inputs: PathInput, runtime: ToolRuntime) -> list[Episode]:
    episodes = parse.parse_folder(inputs.path)
    if not episodes:
        raise ValueError(f"no episode .txt files found in {inputs.path}")
    for episode in episodes:
        for warning in episode.warnings:
            runtime.log.note("parse", f"E{episode.number:02d}: {warning}")
    return episodes


def _validate_input(inputs: EpisodeInputs, _: ToolRuntime) -> InputValidation:
    return InputValidation(
        episodes=len(inputs.episodes),
        beats=sum(len(episode.beats) for episode in inputs.episodes),
        warnings=[{"episode": episode.number, "source_file": episode.source_file, "warning": warning}
                  for episode in inputs.episodes for warning in episode.warnings],
    )


def _analyze(inputs: EpisodeInputs, runtime: ToolRuntime) -> list[EpisodeAnalysis]:
    return analyze.analyze_show(inputs.episodes, runtime.client, log=runtime.log,
                                concurrency=runtime.concurrency("analyze", 4))


def _judge(inputs: CandidateInputs, runtime: ToolRuntime) -> list[ColdViewerVerdict]:
    return judge.judge_candidates(inputs.candidates, inputs.episodes, runtime.client, log=runtime.log,
                                  top_k=runtime.promo_cfg.get("judge_top_k", 40),
                                  concurrency=runtime.concurrency("judge", 2))


def _validate_promos(inputs: PromoInputs, runtime: ToolRuntime) -> list[dict]:
    issues = validate.validate_promos(inputs.promos, inputs.episodes, runtime.promo_cfg, inputs.analyses)
    for issue in issues:
        runtime.log.note("validate", json.dumps(issue))
    return issues


def _assemble(inputs: AssembleInputs, runtime: ToolRuntime) -> RunResult:
    show = re.sub(r"[_-]+", " ", inputs.path.resolve().name).strip()
    return RunResult(
        show=show, created_at=datetime.now().isoformat(timespec="seconds"),
        config={"models": runtime.models.copy(), "promo": runtime.promo_cfg},
        **{field: getattr(inputs, field) for field in AssembleInputs.model_fields if field != "path"},
        run_log=runtime.log.rows,
    )


def _export(inputs: ExportInputs, runtime: ToolRuntime) -> dict[str, str]:
    result, out = inputs.result, inputs.out_dir
    paths = export.write_outputs(result, out)
    summary = {
        "show": result.show, "episodes": len(result.episodes), "candidates": len(result.candidates),
        "finalists": len(result.verdicts), "promos": len(result.promos),
        "validation_issues": inputs.validation_issues, "cost": runtime.log.summary(),
    }
    summary_path = out / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_path = out / "run_log.jsonl"
    runtime.log.write_jsonl(log_path)
    return {**paths, summary_path.name: str(summary_path), log_path.name: str(log_path)}


def read_run(path: str | Path) -> RunResult:
    path = Path(path)
    if path.is_dir():
        path = path / "result.json"
    return RunResult.model_validate_json(path.read_bytes())


def _studio(inputs: AudienceInputs, _: ToolRuntime) -> dict:
    events = inputs.events.read_text(encoding="utf-8-sig") if inputs.events else None
    output = studio.build_studio(inputs.result, events_csv=events, simulate=inputs.simulate)
    output.pop("matrix", None)  # The studio CLI keeps viewer-level rows out of editor handoffs.
    return output


LOCAL = "Local processing"
TOOLS = {
    tool.name: tool for tool in (
        Tool("parse_episodes", "Read timestamped screenplay episodes from a folder.", PathInput, list[Episode],
             _parse, None, "script", LOCAL),
        Tool("validate_episodes", "Collect parser warnings and count timestamped beats.", EpisodeInputs,
             InputValidation, _validate_input, None, "script", LOCAL),
        Tool("analyze_episodes", "Score hooks, cliffhangers, and episode continuity.", EpisodeInputs,
             list[EpisodeAnalysis], _analyze, "analyze", "promo", "One request per episode"),
        Tool("build_candidates", "Find contiguous and joined clips from scored beats.", AnalysisInputs,
             list[Candidate], lambda i, rt: candidates.generate_candidates(i.episodes, i.analyses, rt.promo_cfg),
             None, "promo", LOCAL),
        Tool("cold_viewer_test", "Judge whether a new viewer can follow each finalist.", CandidateInputs,
             list[ColdViewerVerdict], _judge, "judge", "promo", "One request per finalist"),
        Tool("rank_promos", "Rank for editorial strength, variety, and posting wave.", RankInputs, list[Promo],
             lambda i, rt: rank.rank_promos(i.candidates, i.verdicts, i.episodes, i.analyses, rt.promo_cfg),
             None, "promo", LOCAL),
        Tool("write_why_lines", "Draft one line on why each ranked promo works.", WriteInputs, list[Promo],
             lambda i, rt: writer.write_why_lines(i.promos, i.episodes, rt.client, log=rt.log),
             "writer", "promo", "One request for the shortlist"),
        Tool("episode_report", "Report episode openings, endings, and cliffhanger pickup.", AnalysisInputs,
             list[dict], lambda i, _: report.episode_report(i.episodes, i.analyses), None, "script", LOCAL),
        Tool("season_summary", "Summarize script flags and candidate unlock episodes.", ReportInputs,
             dict, lambda i, _: report.season_summary(i.rows), None, "script", LOCAL),
        Tool("validate_promos", "Check clip duration, cut points, hook, and cliffhanger.", PromoInputs,
             list[dict], _validate_promos, None, "promo", LOCAL),
        Tool("assemble_run", "Package workflow outputs as a HookCut run.", AssembleInputs, RunResult,
             _assemble, None, "platform", LOCAL),
        Tool("export_outputs", "Write the ranked cuts, script report, and editor handoff.", ExportInputs,
             dict[str, str], _export, None, "platform", LOCAL),
        Tool("load_run", "Load result.json from a saved run directory or file.", PathInput, RunResult,
             lambda i, _: read_run(i.path), None, "platform", LOCAL),
        Tool("build_studio", "Re-rank saved cuts with audience activity when supplied.", AudienceInputs,
             dict, _studio, None, "audience", LOCAL),
    )
}


def get_tool(name: str) -> Tool | None:
    if name in TOOLS:
        return TOOLS[name]
    if name not in ("story_graph", "engagement"):
        return None
    module_name = f"hookcut.{name}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return None
    if name == "story_graph":
        return Tool(name, "Link characters and open questions across the season.", ResultInputs, module.StoryGraph,
                    lambda i, rt: module.build_story_graph(i.result, log=rt.log), None, "script", LOCAL)
    build = getattr(module, "build_engagement", None)
    if build is not None:
        return Tool(name, "Build activity matrices and hybrid recommendations.", ResultInputs, dict,
                    lambda i, _: build(i.result), None, "audience", LOCAL)
    return None


def list_tools() -> list[dict]:
    tools = [get_tool(name) for name in (*TOOLS, "story_graph", "engagement")]
    return [tool.card() for tool in tools if tool is not None]
