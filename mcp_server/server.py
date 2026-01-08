#!/usr/bin/env python3
"""
SELCIE MCP Server

MCP server providing LLM access to SELCIE chameleon field calculations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

from tools import (
    CALCULATE_PHYSICAL_PARAMETERS_TOOL,
    handle_calculate_physical_parameters,
    CREATE_MESH_TOOL,
    handle_create_mesh,
    PLOT_MESH_TOOL,
    handle_plot_mesh,
    SOLVE_TOOL,
    handle_solve,
    EVALUATE_TOOL,
    handle_evaluate,
    PLOT_TOOL,
    handle_plot,
    GET_STATE_TOOL,
    handle_get_state,
    CLEAR_TOOL,
    handle_clear,
)


# Create the MCP server
server = Server("selcie")


# Tool registry: maps tool names to their handlers
TOOL_HANDLERS = {
    "calculate_physical_parameters": handle_calculate_physical_parameters,
    "create_mesh": handle_create_mesh,
    "plot_mesh": handle_plot_mesh,
    "solve": handle_solve,
    "evaluate": handle_evaluate,
    "plot": handle_plot,
    "get_state": handle_get_state,
    "clear": handle_clear,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        CALCULATE_PHYSICAL_PARAMETERS_TOOL,
        CREATE_MESH_TOOL,
        PLOT_MESH_TOOL,
        SOLVE_TOOL,
        EVALUATE_TOOL,
        PLOT_TOOL,
        GET_STATE_TOOL,
        CLEAR_TOOL,
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle tool calls."""
    handler = TOOL_HANDLERS.get(name)

    if handler is None:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "UNKNOWN_TOOL",
                "message": f"Unknown tool: {name}"
            }
        }, indent=2))]

    return await handler(arguments)


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
