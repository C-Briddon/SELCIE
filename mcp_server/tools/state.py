#!/usr/bin/env python3
"""Session state management tools for SELCIE MCP server."""

import json
from typing import Any

from mcp.types import TextContent, Tool

from utils.session import get_session


# get_state tool
GET_STATE_TOOL = Tool(
    name="get_state",
    description="""Query current session state: list all meshes and solutions.

Returns information about all meshes and solutions in the current session,
including their IDs, parameters, and status.
""",
    inputSchema={
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }
)


async def handle_get_state(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle get_state tool calls."""
    session = get_session()
    return [TextContent(type="text", text=json.dumps(session.to_dict(), indent=2))]


# clear tool
CLEAR_TOOL = Tool(
    name="clear",
    description="""Clear session objects to free memory.

Can clear all objects, just meshes, just solutions, or specific IDs.
""",
    inputSchema={
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "enum": ["all", "solutions", "meshes"],
                "description": "What to clear. Default: all",
                "default": "all"
            },
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific IDs to clear. If provided, clears those specific IDs regardless of 'what'"
            }
        }
    }
)


async def handle_clear(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle clear tool calls."""
    session = get_session()

    what = arguments.get("what", "all")
    ids = arguments.get("ids")

    cleared = session.clear(what=what, ids=ids)

    # Get remaining state
    remaining_meshes = list(session.meshes.keys())
    remaining_solutions = list(session.solutions.keys())

    return [TextContent(type="text", text=json.dumps({
        "cleared": cleared,
        "remaining": {
            "meshes": remaining_meshes,
            "solutions": remaining_solutions
        }
    }, indent=2))]
