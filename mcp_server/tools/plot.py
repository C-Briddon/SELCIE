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
- field_2d: 2D colormap of field (2D meshes only)
- force_1d: Radial gradient magnitude |∇φ|(r)
- force_2d: 2D colormap of gradient magnitude (2D meshes only)
- comparison: Compare multiple solutions on same plot
- density or density_1d: Radial density profile ρ̂(r) with optional adiabatic field overlay
- density_2d: 2D colormap of density (2D meshes only)
- slice_xy: 2D slice through 3D field in xy-plane at given z
- slice_xz: 2D slice through 3D field in xz-plane at given y
- slice_yz: 2D slice through 3D field in yz-plane at given x

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
                "enum": ["field_1d", "field_2d", "force_1d", "force_2d", "comparison", "density", "density_1d", "density_2d", "slice_xy", "slice_xz", "slice_yz"],
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
                    "title": {"type": "string", "description": "Custom title"},
                    "slice_position": {"type": "number", "description": "Position of slice plane (default: 0)"},
                    "n_grid": {"type": "integer", "description": "Grid resolution for slice sampling (default: 100)"},
                    "quantity": {"type": "string", "enum": ["field", "gradient_magnitude", "density"], "description": "Quantity to plot in slice (default: field)"}
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


def _load_solution_data(solution_id: str, session, need_grad: bool = False, need_density: bool = False):
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

    # Get function space degree from solution info (default 2 for backwards compatibility)
    deg_V = getattr(solution_info, 'deg_V', 2)

    # Create function space and load field
    V = d.FunctionSpace(mesh, "CG", deg_V)
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

    # Load gradient vector if needed
    field_grad = None
    if need_grad:
        grad_file = os.path.join(solution_path, "field_grad.h5")
        if os.path.exists(grad_file):
            # field_grad is a vector function - use same degree as scalar field
            V_vec = d.VectorFunctionSpace(mesh, "CG", deg_V)
            field_grad = d.Function(V_vec)
            try:
                with d.HDF5File(mesh.mpi_comm(), grad_file, "r") as f:
                    f.read(field_grad, "field_grad")
            except Exception:
                field_grad = None

    # Load density if needed
    density = None
    if need_density:
        density_file = os.path.join(solution_path, "density.h5")
        if os.path.exists(density_file):
            try:
                V_dg = d.FunctionSpace(mesh, "DG", 0)
                density = d.Function(V_dg)
                with d.HDF5File(mesh.mpi_comm(), density_file, "r") as f:
                    f.read(density, "density")
            except Exception:
                density = None

    return {
        "mesh": mesh,
        "field": field,
        "field_grad": field_grad,
        "density": density,
        "solution_info": solution_info,
        "mesh_info": mesh_info,
    }, None


def _evaluate_radial(field, mesh, n_points=200, r_min=None, r_max=None, log_spacing=False):
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


def _evaluate_radial_gradient(field_grad, mesh, n_points=200, r_min=None, r_max=None, log_spacing=False):
    """Evaluate gradient magnitude along radial direction using field_grad vector and np.linalg.norm."""
    coords = mesh.coordinates()
    if r_min is None:
        r_min = 0.01
    if r_max is None:
        r_max = float(coords[:, 0].max()) * 0.95

    if log_spacing and r_min > 0:
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
    else:
        r_values = np.linspace(r_min, r_max, n_points)

    grad_mag_values = []
    valid_r = []
    for r in r_values:
        try:
            grad_vec = field_grad(r, 0.0)
            grad_mag = np.linalg.norm(grad_vec)
            grad_mag_values.append(grad_mag)
            valid_r.append(r)
        except RuntimeError:
            pass  # Point outside mesh

    return np.array(valid_r), np.array(grad_mag_values)


def _plot_force_1d(data, options, ax):
    """Create 1D force (gradient magnitude) plot."""
    if data["field_grad"] is None:
        ax.text(0.5, 0.5, "Gradient not computed for this solution",
                ha='center', va='center', transform=ax.transAxes)
        return

    n_points = options.get("n_points", 200)
    r_min = options.get("r_min")
    r_max = options.get("r_max")
    log_r = options.get("log_r", False)
    log_scale = options.get("log_scale", True)

    # Evaluate gradient vector and compute magnitude with np.linalg.norm (like Sphere_Standalone.py)
    r, grad = _evaluate_radial_gradient(
        data["field_grad"], data["mesh"],
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

    if data["field_grad"] is None:
        ax.text(0.5, 0.5, "Gradient not computed for this solution",
                ha='center', va='center', transform=ax.transAxes)
        return

    mesh = data["mesh"]
    field_grad = data["field_grad"]
    colormap = options.get("colormap", "hot")
    log_scale = options.get("log_scale", True)

    coords = mesh.coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

    # Compute gradient magnitude at each vertex using np.linalg.norm (like Sphere_Standalone.py)
    grad_vals = np.zeros(len(coords))
    for i, (xi, yi) in enumerate(coords):
        try:
            grad_vec = field_grad(xi, yi)
            grad_vals[i] = np.linalg.norm(grad_vec)
        except RuntimeError:
            grad_vals[i] = np.nan

    if log_scale and np.all(grad_vals[~np.isnan(grad_vals)] > 0):
        grad_vals = np.log10(np.where(grad_vals > 0, grad_vals, np.nan))
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
            r, vals = _evaluate_radial(data["field"], data["mesh"], n_points=n_points, log_spacing=log_r)
            ylabel = 'φ'
        else:
            # Gradient magnitude using field_grad and np.linalg.norm
            if data["field_grad"] is None:
                continue
            r, vals = _evaluate_radial_gradient(data["field_grad"], data["mesh"], n_points=n_points, log_spacing=log_r)
            ylabel = '|∇φ|'

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


def _evaluate_radial_density(density, mesh, n_points=200, r_min=None, r_max=None, log_spacing=False):
    """Evaluate density along radial direction."""
    coords = mesh.coordinates()
    if r_min is None:
        r_min = 0.01
    if r_max is None:
        r_max = float(coords[:, 0].max()) * 0.95

    if log_spacing and r_min > 0:
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
    else:
        r_values = np.linspace(r_min, r_max, n_points)

    density_values = []
    valid_r = []
    for r in r_values:
        try:
            val = density(r, 0.0)
            density_values.append(val)
            valid_r.append(r)
        except RuntimeError:
            pass  # Point outside mesh

    return np.array(valid_r), np.array(density_values)


def _plot_density_1d(data, options, ax):
    """Create 1D density profile plot."""
    if data["density"] is None:
        ax.text(0.5, 0.5, "Density not saved with this solution\n(re-solve to generate)",
                ha='center', va='center', transform=ax.transAxes)
        return

    n_points = options.get("n_points", 200)
    r_min = options.get("r_min")
    r_max = options.get("r_max")
    log_r = options.get("log_r", False)
    log_scale = options.get("log_scale", True)  # Default to log for density
    show_adiabatic = options.get("show_adiabatic", True)

    r, rho = _evaluate_radial_density(
        data["density"], data["mesh"],
        n_points=n_points, r_min=r_min, r_max=r_max, log_spacing=log_r
    )

    ax.plot(r, rho, 'b-', linewidth=2, label='ρ̂(r)')

    # Optionally overlay adiabatic field for comparison
    if show_adiabatic:
        n_power = data['solution_info'].n
        phi_adiabatic = np.where(rho > 0, np.power(rho, -1.0 / (n_power + 1)), np.nan)
        ax2 = ax.twinx()
        ax2.plot(r, phi_adiabatic, 'g--', linewidth=1.5, alpha=0.7, label='φ_adiabatic')
        ax2.set_ylabel('φ_adiabatic = ρ̂^{-1/(n+1)}', fontsize=10, color='green')
        ax2.tick_params(axis='y', labelcolor='green')

    if log_r:
        ax.set_xscale('log')
    if log_scale and np.all(rho > 0):
        ax.set_yscale('log')

    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel('ρ̂', fontsize=11)
    ax.set_title(f"Density profile (α={data['solution_info'].alpha})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')


def _plot_density_2d(data, options, ax):
    """Create 2D density colormap."""
    import matplotlib.tri as mtri

    if data["density"] is None:
        ax.text(0.5, 0.5, "Density not saved with this solution\n(re-solve to generate)",
                ha='center', va='center', transform=ax.transAxes)
        return

    mesh = data["mesh"]
    density = data["density"]
    colormap = options.get("colormap", "YlOrRd")
    log_scale = options.get("log_scale", True)

    coords = mesh.coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

    # Get density values at vertices (DG0 is cell-centered, need to evaluate at vertices)
    density_vals = np.zeros(len(coords))
    for i, (xi, yi) in enumerate(coords):
        try:
            density_vals[i] = density(xi, yi)
        except RuntimeError:
            density_vals[i] = np.nan

    if log_scale and np.all(density_vals[~np.isnan(density_vals)] > 0):
        density_vals = np.log10(np.where(density_vals > 0, density_vals, np.nan))
        cbar_label = 'log₁₀(ρ̂)'
    else:
        cbar_label = 'ρ̂'

    # Create triangulation
    cells = mesh.cells()
    triang = mtri.Triangulation(x, y, cells)

    # Plot
    tpc = ax.tripcolor(triang, density_vals, cmap=colormap, shading='gouraud')
    cbar = ax.figure.colorbar(tpc, ax=ax, label=cbar_label)

    ax.set_xlabel('r', fontsize=11)
    ax.set_ylabel('z', fontsize=11)
    ax.set_title(f"Density (α={data['solution_info'].alpha})", fontsize=12)
    ax.set_aspect('equal')


def _plot_slice(data, options, ax, plane: str):
    """Create 2D slice plot through 3D field.

    Args:
        data: Solution data dict with mesh, field, field_grad, density
        options: Plot options dict
        ax: Matplotlib axis
        plane: One of 'xy', 'xz', 'yz'
    """
    mesh = data["mesh"]
    field = data["field"]
    field_grad = data.get("field_grad")
    density = data.get("density")

    # Check mesh is 3D
    coords = mesh.coordinates()
    if coords.shape[1] != 3:
        ax.text(0.5, 0.5, f"Slice plots require 3D mesh\n(this mesh is {coords.shape[1]}D)",
                ha='center', va='center', transform=ax.transAxes)
        return

    # Get options
    slice_pos = options.get("slice_position", 0.0)
    n_grid = options.get("n_grid", 100)
    colormap = options.get("colormap", "viridis")
    log_scale = options.get("log_scale", False)
    quantity = options.get("quantity", "field")

    # Determine axis indices and bounds based on plane
    if plane == "xy":
        ax1_idx, ax2_idx, fixed_idx = 0, 1, 2
        ax1_label, ax2_label = 'x', 'y'
        fixed_label = 'z'
    elif plane == "xz":
        ax1_idx, ax2_idx, fixed_idx = 0, 2, 1
        ax1_label, ax2_label = 'x', 'z'
        fixed_label = 'y'
    elif plane == "yz":
        ax1_idx, ax2_idx, fixed_idx = 1, 2, 0
        ax1_label, ax2_label = 'y', 'z'
        fixed_label = 'x'
    else:
        ax.text(0.5, 0.5, f"Unknown plane: {plane}", ha='center', va='center', transform=ax.transAxes)
        return

    # Get mesh bounds
    ax1_min, ax1_max = coords[:, ax1_idx].min(), coords[:, ax1_idx].max()
    ax2_min, ax2_max = coords[:, ax2_idx].min(), coords[:, ax2_idx].max()

    # Create sampling grid
    ax1_vals = np.linspace(ax1_min * 0.95, ax1_max * 0.95, n_grid)
    ax2_vals = np.linspace(ax2_min * 0.95, ax2_max * 0.95, n_grid)
    ax1_grid, ax2_grid = np.meshgrid(ax1_vals, ax2_vals)

    # Sample field on grid
    values = np.full_like(ax1_grid, np.nan)

    for i in range(n_grid):
        for j in range(n_grid):
            # Construct 3D point
            point = [0.0, 0.0, 0.0]
            point[ax1_idx] = ax1_vals[j]
            point[ax2_idx] = ax2_vals[i]
            point[fixed_idx] = slice_pos

            try:
                if quantity == "field":
                    values[i, j] = field(*point)
                elif quantity == "gradient_magnitude" and field_grad is not None:
                    grad_vec = field_grad(*point)
                    values[i, j] = np.linalg.norm(grad_vec)
                elif quantity == "density" and density is not None:
                    values[i, j] = density(*point)
            except RuntimeError:
                pass  # Point outside mesh

    # Check if we got any valid values
    if np.all(np.isnan(values)):
        ax.text(0.5, 0.5, f"No valid points in slice at {fixed_label}={slice_pos}\n"
                f"Mesh bounds: {fixed_label} ∈ [{coords[:, fixed_idx].min():.3f}, {coords[:, fixed_idx].max():.3f}]",
                ha='center', va='center', transform=ax.transAxes)
        return

    # Apply log scale if requested
    if quantity == "field":
        cbar_label = 'φ'
    elif quantity == "gradient_magnitude":
        cbar_label = '|∇φ|'
    elif quantity == "density":
        cbar_label = 'ρ̂'
    else:
        cbar_label = quantity

    if log_scale and np.nanmin(values) > 0:
        values = np.log10(values)
        cbar_label = f'log₁₀({cbar_label})'

    # Plot using contourf for smoother appearance (like 2D plots)
    # Mask NaN values for contourf
    valid_mask = ~np.isnan(values)
    if np.any(valid_mask):
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        levels = np.linspace(vmin, vmax, 100)
        cf = ax.contourf(ax1_grid, ax2_grid, values, levels=levels, cmap=colormap, extend='both')
        cbar = ax.figure.colorbar(cf, ax=ax, label=cbar_label)
    else:
        # Fallback to imshow if contourf fails
        im = ax.imshow(values, extent=[ax1_min, ax1_max, ax2_min, ax2_max],
                       origin='lower', cmap=colormap, aspect='equal')
        cbar = ax.figure.colorbar(im, ax=ax, label=cbar_label)

    ax.set_aspect('equal')

    ax.set_xlabel(ax1_label, fontsize=11)
    ax.set_ylabel(ax2_label, fontsize=11)
    alpha_val = data['solution_info'].alpha
    alpha_str = f"{alpha_val:.2e}" if alpha_val >= 1e4 else f"{alpha_val:.2f}"
    ax.set_title(f"{quantity.replace('_', ' ').title()} slice at {fixed_label}={slice_pos:.3f} (α={alpha_str})", fontsize=12)


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

        # Determine if we need gradient or density data
        is_slice = plot_type.startswith("slice_")
        slice_quantity = options.get("quantity", "field") if is_slice else None
        need_grad = plot_type in ("force_1d", "force_2d") or \
                    (plot_type == "comparison" and options.get("quantity") == "gradient_magnitude") or \
                    (is_slice and slice_quantity == "gradient_magnitude")
        need_density = plot_type in ("density", "density_1d", "density_2d") or \
                       (is_slice and slice_quantity == "density")

        # Load solution data
        solutions_data = []
        for sid in solution_ids:
            data, error = _load_solution_data(sid, session, need_grad=need_grad, need_density=need_density)
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
        elif plot_type == "density" or plot_type == "density_1d":
            _plot_density_1d(solutions_data[0], options, ax)
        elif plot_type == "density_2d":
            _plot_density_2d(solutions_data[0], options, ax)
        elif plot_type == "slice_xy":
            _plot_slice(solutions_data[0], options, ax, plane="xy")
        elif plot_type == "slice_xz":
            _plot_slice(solutions_data[0], options, ax, plane="xz")
        elif plot_type == "slice_yz":
            _plot_slice(solutions_data[0], options, ax, plane="yz")
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
