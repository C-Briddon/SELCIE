# SELCIE MCP Server

MCP server providing LLM access to SELCIE chameleon field calculations.

## Available Tools

- `calculate_physical_parameters` - Calculate physical parameters for chameleon fields
- `create_mesh` - Create computational meshes for simulations
- `plot_mesh` - Visualize generated meshes
- `solve` - Solve chameleon field equations
- `evaluate` - Evaluate field values at specific points
- `plot` - Plot simulation results
- `get_state` - Get current server state
- `clear` - Clear server state

## Setup

Ensure you have the SELCIE conda environment activated:

```bash
conda activate SELCIE
```

Install the MCP package:

```bash
pip install mcp
```

## Adding to Claude Code

```bash
claude mcp add selcie -- /path/to/conda/envs/SELCIE/bin/python /path/to/SELCIE/mcp_server/server.py
```

Replace `/path/to/conda/envs/SELCIE` with your conda environment path (e.g., `/opt/anaconda3/envs/SELCIE`) and `/path/to/SELCIE` with the actual path to your SELCIE installation.

To add for the current project only:

```bash
claude mcp add selcie --scope project -- /path/to/conda/envs/SELCIE/bin/python /path/to/SELCIE/mcp_server/server.py
```

To find your conda environment path:

```bash
conda env list
```

## Adding to Codex

```bash
codex --mcp-config '{"selcie": {"command": "/path/to/conda/envs/SELCIE/bin/python", "args": ["/path/to/SELCIE/mcp_server/server.py"]}}'
```

Or add to your Codex configuration file:

```json
{
  "mcpServers": {
    "selcie": {
      "command": "/path/to/conda/envs/SELCIE/bin/python",
      "args": ["/path/to/SELCIE/mcp_server/server.py"]
    }
  }
}
```

## Running the Server Manually

To test the server directly:

```bash
conda activate SELCIE
python server.py
```

The server communicates via stdio and expects MCP protocol messages.
