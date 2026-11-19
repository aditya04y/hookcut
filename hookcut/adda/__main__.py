"""Command-line access to the Adda catalog, router, and workflows."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .router import TIERS, Router
from .runner import list_workflows, run_workflow
from .tools import list_tools


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List available tools and workflows")
    route = commands.add_parser("route", help="Explain the model choices for a task")
    route.add_argument("task")
    route.add_argument("--quality", choices=TIERS, default="balanced")
    route.add_argument("--exclude-family", action="append", default=[])
    route.add_argument("--max-cost", type=float, help="USD cap per 1M input plus 1M output tokens")
    run = commands.add_parser("run", help="Execute a workflow")
    run.add_argument("workflow", choices=list_workflows())
    run.add_argument("--input", required=True, help="Script folder or saved run for audience_update")
    run.add_argument("--out", required=True, help="Output directory")
    run.add_argument("--quality", choices=TIERS, default="balanced")
    run.add_argument("--max-cost", type=float, help="USD cap per 1M input plus 1M output tokens")
    activity = run.add_mutually_exclusive_group()
    activity.add_argument("--events", help="Audience impression CSV for audience_update")
    activity.add_argument("--simulate", action="store_true", help="Use labelled simulated activity")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            output = {"tools": list_tools(), "workflows": [w.model_dump() for w in list_workflows().values()]}
        elif args.command == "route":
            output = Router().decisions(args.task, quality=args.quality, exclude_families=args.exclude_family,
                                        max_cost=args.max_cost)
        else:
            output = run_workflow(args.workflow, args.input, args.out, quality=args.quality, max_cost=args.max_cost,
                                  events=args.events, simulate=args.simulate)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"HookCut Adda: {exc}\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
