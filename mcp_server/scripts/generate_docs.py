#!/usr/bin/env python3
"""Generate markdown documentation from MCP tool definitions."""

import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_tool_modules():
    """Get all tool modules that have tool definitions."""
    from mcp.types import Tool

    tools_dir = Path(__file__).parent.parent / "tools"
    tools = []

    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"tools.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)

            # Find all Tool objects in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, Tool):
                    tools.append((py_file.stem, attr))

        except Exception as e:
            print(f"Warning: Could not load {module_name}: {e}", file=sys.stderr)

    # Sort by tool name for consistent ordering
    tools.sort(key=lambda x: x[1].name)
    return tools


def format_type(schema: dict) -> str:
    """Format a JSON schema type as a readable string."""
    if "oneOf" in schema:
        types = []
        for option in schema["oneOf"]:
            types.append(format_type(option))
        return " | ".join(types)

    if "enum" in schema:
        return f'`{"` | `".join(str(e) for e in schema["enum"])}`'

    type_name = schema.get("type", "any")

    if type_name == "array":
        items = schema.get("items", {})
        item_type = format_type(items) if items else "any"
        return f"array[{item_type}]"

    if type_name == "object":
        if schema.get("additionalProperties"):
            return "object"
        return "object"

    return type_name


def generate_parameter_table(schema: dict) -> str:
    """Generate a markdown table of parameters from JSON schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        return "_No parameters_\n"

    lines = [
        "| Parameter | Type | Required | Description |",
        "|-----------|------|----------|-------------|"
    ]

    for name, prop in properties.items():
        type_str = format_type(prop)
        req_str = "Yes" if name in required else "No"
        desc = prop.get("description", "")

        # Add default value if present
        if "default" in prop:
            default_val = prop["default"]
            if isinstance(default_val, str):
                desc += f" Default: `\"{default_val}\"`"
            else:
                desc += f" Default: `{default_val}`"

        # Escape pipes in description
        desc = desc.replace("|", "\\|")

        lines.append(f"| `{name}` | {type_str} | {req_str} | {desc} |")

    return "\n".join(lines) + "\n"


def generate_nested_properties(schema: dict, indent: int = 0) -> str:
    """Generate documentation for nested object properties."""
    properties = schema.get("properties", {})
    if not properties:
        return ""

    lines = []
    prefix = "  " * indent

    for name, prop in properties.items():
        type_str = format_type(prop)
        desc = prop.get("description", "")

        lines.append(f"{prefix}- **`{name}`** ({type_str}): {desc}")

        # Recurse into nested objects
        if prop.get("type") == "object" and prop.get("properties"):
            nested = generate_nested_properties(prop, indent + 1)
            if nested:
                lines.append(nested)

    return "\n".join(lines)


def generate_tool_doc(name: str, tool) -> str:
    """Generate markdown documentation for a single tool."""
    lines = [
        f"## {tool.name}",
        "",
        tool.description.strip(),
        "",
        "### Parameters",
        "",
    ]

    schema = tool.inputSchema
    lines.append(generate_parameter_table(schema))

    # Document nested object properties
    properties = schema.get("properties", {})
    for prop_name, prop in properties.items():
        if prop.get("type") == "object" and prop.get("properties"):
            lines.append(f"#### `{prop_name}` options")
            lines.append("")
            lines.append(generate_nested_properties(prop))
            lines.append("")

    return "\n".join(lines)


def generate_full_docs(output_path: Path = None) -> str:
    """Generate full documentation for all tools."""
    modules = get_tool_modules()

    lines = [
        "# SELCIE MCP Server - Tool Reference",
        "",
        "Auto-generated documentation for all available tools.",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]

    # Generate TOC
    for name, tool in modules:
        lines.append(f"- [{tool.name}](#{tool.name})")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Generate tool docs
    for name, tool in modules:
        lines.append(generate_tool_doc(name, tool))
        lines.append("---")
        lines.append("")

    doc = "\n".join(lines)

    if output_path:
        output_path.write_text(doc)
        print(f"Documentation written to {output_path}")

    return doc


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate MCP tool documentation")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "TOOLS.md",
        help="Output file path (default: docs/TOOLS.md)"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of file"
    )

    args = parser.parse_args()

    if args.stdout:
        print(generate_full_docs())
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generate_full_docs(args.output)


if __name__ == "__main__":
    main()
