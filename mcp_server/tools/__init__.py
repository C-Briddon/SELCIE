"""SELCIE MCP tools."""

from .calculate_physical_parameters import (
    TOOL_DEFINITION as CALCULATE_PHYSICAL_PARAMETERS_TOOL,
    handle as handle_calculate_physical_parameters,
)
from .create_mesh import (
    TOOL_DEFINITION as CREATE_MESH_TOOL,
    handle as handle_create_mesh,
)
from .plot_mesh import (
    TOOL_DEFINITION as PLOT_MESH_TOOL,
    handle as handle_plot_mesh,
)
from .solve import (
    TOOL_DEFINITION as SOLVE_TOOL,
    handle as handle_solve,
)

__all__ = [
    "CALCULATE_PHYSICAL_PARAMETERS_TOOL",
    "handle_calculate_physical_parameters",
    "CREATE_MESH_TOOL",
    "handle_create_mesh",
    "PLOT_MESH_TOOL",
    "handle_plot_mesh",
    "SOLVE_TOOL",
    "handle_solve",
]
