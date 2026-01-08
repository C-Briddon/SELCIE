#!/usr/bin/env python3
"""Plot tool for SELCIE MCP server - visualize solutions."""

import base64
import io
import json
import os
from typing import Any

import numpy as np
from mcp.types import TextContent, ImageContent, Tool

from utils.session import get_session


TOOL_DEFINITION = Tool(
    name="plot",
    description="""Generate visualizations of chameleon field solutions.

Plot types:
- field_1d: Radial field profile φ(r)
- field_2d: 2D colormap of field
- force_1d: Radial gradient magnitude |∇φ|(r)
- force_2d: 2D colormap of gradient magnitude
- comparison: Compare multiple solutions on same plot
- density: Show density profile (if available)

Returns PNG image (base64 or saved to file).
""",
    inputSchema={
        "type": "object",
        "properties": {
            "solution_id": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}}
                ],
                "description": "Solution ID(s) to plot"
            },
            "plot_type": {
                "type": "string",
                "enum": ["field_1d", "field_2d", "force_1d", "force_2d", "comparison", "density"],
                "description": "Type of plot to generate"
            },
            "options": {
                "type": "object",
                "description": "Plot customization options",
                "properties": {
                    "log_r": {"type": "boolean", "description": "Use log scale for r-axis (1D plots)"},
                    "log_scale": {"type": "boolean", "description": "Use log scale for values"},
                    "colormap": {"type": "string", "description": "Matplotlib colormap name"},
                    "show_mesh": {"type": "boolean", "description": "Overlay mesh on 2D plots"},
                    "n_points": {"type": "integer", "description": "Number of sample points for 1D"},
                    "r_min": {"type": "number", "description": "Minimum radius for 1D"},
                    "r_max": {"type": "number", "description": "Maximum radius for 1D"},
                    "quantity": {"type": "string", "description": "Quantity to plot (comparison mode)"},
                    "legend_by": {"type": "string", "description": "Label legend by this field"},
                    "figsize": {"type": "array", "items": {"type": "number"}},
                    "title": {"type": "string", "description": "Custom title"}
                }
            },
            "output_path": {
                "type": "string",
                "description": "Save to file path. If not provided, returns base64 image"
            },
            "format": {
                "type": "string",
                "enum": ["png", "pdf", "svg"],
                "description": "Output format. Default: png",
                "default": "png"
            }
        },
        "required": ["solution_id", "plot_type"]
    }
)


def _load_solution_data(solution_id: str, session, need_grad: bool = False):
    """Load solution field data from HDF5 files."""
    import dolfin as d

    solution_info = session.get_solution(solution_id)
    if solution_info is None:
        return None, f"Solution '{solution_id}' not found"

    mesh_info = session.get_mesh(solution_info.mesh_id)
    if mesh_info is None:
        return None, f"Mesh '{solution_info.mesh_id}' not found"

    # Load mesh
    mesh_path = mesh_info.mesh_path
    mesh_file = os.path.join(mesh_path, "mesh.xdmf")

    mesh = d.Mesh()
    with d.XDMFFile(mesh_file) as f:
        f.read(mesh)

    # Create function space and load field
    V = d.FunctionSpace(mesh, "CG", 1)
    field = d.Function(V)

    # Get solution path
    mesh_dir = os.path.dirname(mesh_path)
    root_dir = os.path.dirname(mesh_dir)
    solution_path = os.path.join(root_dir, "Saved Solutions", solution_id)
    field_file = os.path.join(solution_path, "field.h5")

    if not os.path.exists(field_file):
        return None, f"Field file not found at {field_file}"

    with d.HDF5File(mesh.mpi_comm(), field_file, "r") as f:
        f.read(field, "field")

    # Load gradient if needed
    grad_mag = None
    if need_grad:
        grad_file = os.path.join(solution_path, "field_grad_mag.h5")
        if os.path.exists(grad_file):
            # Try CG1 first (same as field), then DG0 if that fails
            for space_type in ["CG1", "DG0"]:
                try:
                    if space_type == "CG1":
                        V_grad = V
                    else:
                        V_grad = d.FunctionSpace(mesh, "DG", 0)
                    grad_mag = d.Function(V_grad)
                    with d.HDF5File(mesh.mpi_comm(), grad_file, "r") as f:
                        f.read(grad_mag, "field_grad_mag")
                    break  # Success
                except Exception:
                    grad_mag = None
                    continue

    return {
        "mesh": mesh,
        "field": field,
        "grad_mag": grad_mag,
        "solution_info": solution_info,
        "mesh_info": mesh_info,
    }, None


def _evaluate_radial(field, mesh, n_points=100, r_min=None, r_max=None, log_spacing=False):
    """Evaluate field along radial direction."""
    coords = mesh.coordinates()
    if r_min is None:
        r_min = 0.01
    if r_max is None:
        r_max = float(coords[:, 0].max()) * 0.95

    if log_spacing and r_min > 0:
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
    else:
        r_values = np.linspace(r_min, r_max, n_points)

    field_values = []
    valid_r = []
    for r in r_values:
        try:
            val = field(r, 0.0)
            field_values.append(val)
            valid_r.append(r)
        except RuntimeError:
            pass  # Point outside mesh

    return np.array(valid_r), np.array(field_values)


def _plot_field_1d(data, options, ax):
    """Create 1D field profile plot."""
    n_points = options.get("n_points", 200)
    r_min = options.get("r_min")
    r_max = options.get("r_max")
    log_r = options.get("log_r", False)
    log_scale = options.get("log_scale", False)

    r, phi = _evaluate_radial(
        data["field"], data["mesh"],
        n_points=n_points, r_min=r_min, r_max=r_max, log_spacing=log_r
    )

    ax.plot(r, phi, 'b-', linewidth=2, label='φ(r)')

    if log_r:
        ax.set_xscale('log')
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel('φ', fontsize=11)
    ax.set_title(f"Field profile (α={data['solution_info'].alpha})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_force_1d(data, options, ax):
    """Create 1D force (gradient magnitude) plot."""
    if data["grad_mag"] is None:
        ax.text(0.5, 0.5, "Gradient not computed for this solution",
                ha='center', va='center', transform=ax.transAxes)
        return

    n_points = options.get("n_points", 200)
    r_min = options.get("r_min")
    r_max = options.get("r_max")
    log_r = options.get("log_r", False)
    log_scale = options.get("log_scale", True)

    r, grad = _evaluate_radial(
        data["grad_mag"], data["mesh"],
        n_points=n_points, r_min=r_min, r_max=r_max, log_spacing=log_r
    )

    ax.plot(r, grad, 'r-', linewidth=2, label='|∇φ|(r)')

    if log_r:
        ax.set_xscale('log')
    if log_scale and np.all(grad > 0):
        ax.set_yscale('log')

    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel('|∇φ|', fontsize=11)
    ax.set_title(f"Gradient magnitude (α={data['solution_info'].alpha})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_field_2d(data, options, ax):
    """Create 2D field colormap."""
    import matplotlib.tri as mtri

    mesh = data["mesh"]
    field = data["field"]
    colormap = options.get("colormap", "viridis")
    log_scale = options.get("log_scale", False)

    coords = mesh.coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

    # Get field values at vertices
    field_vals = field.compute_vertex_values(mesh)

    if log_scale and np.all(field_vals > 0):
        field_vals = np.log10(field_vals)
        cbar_label = 'log₁₀(φ)'
    else:
        cbar_label = 'φ'

    # Create triangulation
    cells = mesh.cells()
    triang = mtri.Triangulation(x, y, cells)

    # Plot
    tpc = ax.tripcolor(triang, field_vals, cmap=colormap, shading='gouraud')
    cbar = ax.figure.colorbar(tpc, ax=ax, label=cbar_label)

    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel('z', fontsize=11)
    ax.set_title(f"Field (α={data['solution_info'].alpha})", fontsize=12)
    ax.set_aspect('equal')


def _plot_force_2d(data, options, ax):
    """Create 2D gradient magnitude colormap."""
    import matplotlib.tri as mtri

    if data["grad_mag"] is None:
        ax.text(0.5, 0.5, "Gradient not computed for this solution",
                ha='center', va='center', transform=ax.transAxes)
        return

    mesh = data["mesh"]
    grad_mag = data["grad_mag"]
    colormap = options.get("colormap", "hot")
    log_scale = options.get("log_scale", True)

    coords = mesh.coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

    # Get gradient values at vertices
    grad_vals = grad_mag.compute_vertex_values(mesh)

    if log_scale and np.all(grad_vals > 0):
        grad_vals = np.log10(grad_vals)
        cbar_label = 'log₁₀(|∇φ|)'
    else:
        cbar_label = '|∇φ|'

    # Create triangulation
    cells = mesh.cells()
    triang = mtri.Triangulation(x, y, cells)

    # Plot
    tpc = ax.tripcolor(triang, grad_vals, cmap=colormap, shading='gouraud')
    cbar = ax.figure.colorbar(tpc, ax=ax, label=cbar_label)

    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel('z', fontsize=11)
    ax.set_title(f"Gradient magnitude (α={data['solution_info'].alpha})", fontsize=12)
    ax.set_aspect('equal')


def _plot_comparison(solutions_data, options, ax):
    """Compare multiple solutions on same plot."""
    quantity = options.get("quantity", "field")
    legend_by = options.get("legend_by", "alpha")
    n_points = options.get("n_points", 200)
    log_r = options.get("log_r", False)
    log_scale = options.get("log_scale", False)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, data in enumerate(solutions_data):
        color = colors[i % len(colors)]

        if quantity == "field":
            func = data["field"]
            ylabel = 'φ'
        else:
            func = data["grad_mag"]
            ylabel = '|∇φ|'
            if func is None:
                continue

        r, vals = _evaluate_radial(func, data["mesh"], n_points=n_points, log_spacing=log_r)

        if legend_by == "alpha":
            label = f"α={data['solution_info'].alpha}"
        elif legend_by == "solution_id":
            label = data['solution_info'].solution_id
        else:
            label = f"α={data['solution_info'].alpha}"

        ax.plot(r, vals, color=color, linewidth=2, label=label)

    if log_r:
        ax.set_xscale('log')
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"Comparison: {quantity}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()


async def handle(arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle plot tool calls."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        solution_id = arguments["solution_id"]
        plot_type = arguments["plot_type"]
        options = arguments.get("options", {})
        output_path = arguments.get("output_path")
        fmt = arguments.get("format", "png")

        session = get_session()

        # Handle single or multiple solutions
        if isinstance(solution_id, str):
            solution_ids = [solution_id]
        else:
            solution_ids = solution_id

        # Determine if we need gradient data
        need_grad = plot_type in ("force_1d", "force_2d") or \
                    (plot_type == "comparison" and options.get("quantity") == "gradient_magnitude")

        # Load solution data
        solutions_data = []
        for sid in solution_ids:
            data, error = _load_solution_data(sid, session, need_grad=need_grad)
            if error:
                return [TextContent(type="text", text=json.dumps({
                    "error": {
                        "code": "SOLUTION_LOAD_ERROR",
                        "message": error,
                        "available_solutions": list(session.solutions.keys())
                    }
                }, indent=2))]
            solutions_data.append(data)

        # Create figure
        figsize = options.get("figsize", [8, 6])
        fig, ax = plt.subplots(figsize=figsize)

        # Generate plot based on type
        if plot_type == "field_1d":
            _plot_field_1d(solutions_data[0], options, ax)
        elif plot_type == "force_1d":
            _plot_force_1d(solutions_data[0], options, ax)
        elif plot_type == "field_2d":
            _plot_field_2d(solutions_data[0], options, ax)
        elif plot_type == "force_2d":
            _plot_force_2d(solutions_data[0], options, ax)
        elif plot_type == "comparison":
            if len(solutions_data) < 2:
                return [TextContent(type="text", text=json.dumps({
                    "error": {
                        "code": "INVALID_PARAMS",
                        "message": "Comparison plot requires at least 2 solutions"
                    }
                }, indent=2))]
            _plot_comparison(solutions_data, options, ax)
        elif plot_type == "density":
            ax.text(0.5, 0.5, "Density plot not yet implemented\n(density not stored with solution)",
                    ha='center', va='center', transform=ax.transAxes)
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "INVALID_PLOT_TYPE",
                    "message": f"Unknown plot type: {plot_type}"
                }
            }, indent=2))]

        # Apply custom title if provided
        if options.get("title"):
            ax.set_title(options["title"], fontsize=12)

        plt.tight_layout()

        # Save or encode
        if output_path:
            plt.savefig(output_path, format=fmt, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return [TextContent(type="text", text=json.dumps({
                "plot_type": plot_type,
                "solution_ids": solution_ids,
                "output_path": output_path,
                "format": fmt
            }, indent=2))]
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            buf.seek(0)
            image_base64 = base64.standard_b64encode(buf.read()).decode('utf-8')

            return [
                ImageContent(type="image", data=image_base64, mimeType="image/png"),
                TextContent(type="text", text=json.dumps({
                    "plot_type": plot_type,
                    "solution_ids": solution_ids,
                }, indent=2))
            ]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "PLOT_ERROR",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        }, indent=2))]
