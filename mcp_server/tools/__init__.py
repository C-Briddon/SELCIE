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
from .evaluate import (
    TOOL_DEFINITION as EVALUATE_TOOL,
    handle as handle_evaluate,
)
from .plot import (
    TOOL_DEFINITION as PLOT_TOOL,
    handle as handle_plot,
)
from .state import (
    GET_STATE_TOOL,
    handle_get_state,
    CLEAR_TOOL,
    handle_clear,
)
from .plot_step import (
    TOOL_DEFINITION as PLOT_STEP_TOOL,
    handle as handle_plot_step,
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
    "EVALUATE_TOOL",
    "handle_evaluate",
    "PLOT_TOOL",
    "handle_plot",
    "GET_STATE_TOOL",
    "handle_get_state",
    "CLEAR_TOOL",
    "handle_clear",
    "PLOT_STEP_TOOL",
    "handle_plot_step",
]
