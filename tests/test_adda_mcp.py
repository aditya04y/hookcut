"""The installed MCP SDK can start Adda and use its local tools over stdio."""
import asyncio
import sys
from pathlib import Path

from mcp.client import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from hookcut.adda.mcp_server import server

ROOT = Path(__file__).resolve().parents[1]


def test_mcp_tools_over_stdio(tmp_path):
    expected = {"list_tools", "route", "run_workflow", "episode_report", "promos"}
    assert {tool.name for tool in asyncio.run(server.list_tools())} == expected
    params = StdioServerParameters(command=sys.executable, args=["-m", "hookcut.adda.mcp_server"], cwd=ROOT,
                                   env={"PYTHONDONTWRITEBYTECODE": "1"})

    async def request_tools():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                assert {tool.name for tool in (await session.list_tools()).tools} == expected
                demo = str(ROOT / "app/fixtures/demo_run.json")
                response = await session.call_tool("promos", {"run_dir": demo, "top": 1})
                assert not response.is_error
                assert response.structured_content["result"][0]["rank"] == 1
                response = await session.call_tool("run_workflow", {
                    "name": "quick_check", "input_dir": str(ROOT / "samples"), "out_dir": str(tmp_path),
                })
                assert not response.is_error
                assert (tmp_path / "run_manifest.json").exists()

    asyncio.run(request_tools())
