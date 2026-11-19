"""Expose the HookCut catalog, workflows, and saved runs over MCP stdio."""
from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from .router import Router
from .runner import run_workflow as execute_workflow
from .tools import list_tools as tool_cards
from .tools import read_run

server = MCPServer("hookcut", instructions="Inspect HookCut tools, run a workflow, or read an existing run.")
READ_ONLY = ToolAnnotations(read_only_hint=True)


@server.tool(annotations=READ_ONLY)
def list_tools() -> list[dict]:
    """Show HookCut tool cards, including input and output schemas."""
    return tool_cards()


@server.tool(annotations=READ_ONLY)
def route(task: str, quality: Literal["balanced", "quality", "budget"] = "balanced") -> list[dict]:
    """Explain the model choices for a task and show current quota exclusions."""
    return Router().decisions(task, quality=quality)


@server.tool()
def run_workflow(name: str, input_dir: str, out_dir: str) -> dict:
    """Execute an Adda workflow and write its outputs to the requested directory."""
    return execute_workflow(name, input_dir, out_dir)


@server.tool(annotations=READ_ONLY)
def episode_report(run_dir: str) -> list[dict]:
    """Read per-episode opening, ending, and pickup results from a saved run."""
    return read_run(run_dir).episode_report


@server.tool(annotations=READ_ONLY)
def promos(run_dir: str, top: Annotated[int, Field(ge=1)] = 10) -> list[dict]:
    """Read the highest-ranked promos from a saved run."""
    return [promo.model_dump(mode="json") for promo in sorted(read_run(run_dir).promos, key=lambda row: row.rank)[:top]]


if __name__ == "__main__":
    server.run(transport="stdio")
