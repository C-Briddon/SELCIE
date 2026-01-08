#!/usr/bin/env python3
"""Solve tool for SELCIE MCP server."""

import os
import time
from typing import Any

import numpy as np
from mcp.types import TextContent, Tool

from utils.session import get_session, SolutionInfo


TOOL_DEFINITION = Tool(
    name="solve",
    description="""Solve the chameleon field equation on a mesh with specified density profile.

Uses SELCIE's Picard or Newton solver to compute the chameleon scalar field
throughout the domain. The dimensionless field equation is:
    α ∇²φ + φ^{-(n+1)} = ρ̂

where ρ̂ = ρ/ρ₀ is the dimensionless density (ρ₀ is the reference density used to compute α).

Parameters:
- mesh_id: Reference to a previously created mesh
- alpha: Dimensionless coupling constant (from calculate_physical_parameters)
- density: Dimensionless density ρ̂ = ρ/ρ₀ per region (e.g., if ρ₀ = vacuum density, then vacuum → 1.0)
- n: Potential power (default: 1)
- method: Solver method - "picard", "newton", or "auto"
- tol: Convergence tolerance (default: 1e-14)
- max_iter: Maximum iterations (default: 100)
- relaxation: Relaxation factor for Picard (default: 1.0)
- initial_guess: "constant" (default), "adiabatic", or "previous"
""",
    inputSchema={
        "type": "object",
        "properties": {
            "mesh_id": {
                "type": "string",
                "description": "ID of the mesh to solve on"
            },
            "alpha": {
                "type": "number",
                "description": "Dimensionless α parameter (coupling constant)"
            },
            "density": {
                "type": "object",
                "description": "Dimensionless density ρ̂ = ρ/ρ₀ per region (ρ₀ is the reference density used to compute α). Each key is a region name, value is: number, {expression: str}, or {file: str, skip_header?: int}",
                "additionalProperties": True
            },
            "n": {
                "type": "integer",
                "description": "Potential power index. Default: 1",
                "default": 1
            },
            "method": {
                "type": "string",
                "enum": ["picard", "auto"],
                "description": "Solver method. Default: auto (uses picard with relaxation based on alpha)",
                "default": "auto"
            },
            "tol": {
                "type": "number",
                "description": "Convergence tolerance. Default: 1e-14",
                "default": 1e-14
            },
            "max_iter": {
                "type": "integer",
                "description": "Maximum iterations. Default: 100",
                "default": 100
            },
            "relaxation": {
                "type": "number",
                "description": "Relaxation factor (0-1]. Default: 1.0",
                "default": 1.0
            },
            "initial_guess": {
                "type": "string",
                "enum": ["constant", "adiabatic", "previous"],
                "description": "Initial guess strategy. Default: constant",
                "default": "constant"
            },
            "custom_id": {
                "type": "string",
                "description": "Custom solution ID. Default: auto-generated"
            }
        },
        "required": ["mesh_id", "alpha", "density"]
    }
)


def _symmetry_to_selcie(symmetry: str) -> str:
    """Convert MCP symmetry to SELCIE DensityProfile symmetry string."""
    mapping = {
        "axial": "vertical axis-symmetry",
        "none": "translation symmetry",
    }
    return mapping.get(symmetry, symmetry)


def _create_density_function(spec: Any, mesh_symmetry: str, mesh_dimension: int) -> callable:
    """
    Create a density function from a specification.

    Parameters
    ----------
    spec : Any
        Density specification - number, dict with 'expression', or dict with 'file'
    mesh_symmetry : str
        Mesh symmetry ('axial' or 'none')
    mesh_dimension : int
        Mesh dimension (2 or 3)

    Returns
    -------
    callable
        Function that takes (x) and returns density
    """
    # Constant density
    if isinstance(spec, (int, float)):
        rho = float(spec)
        return lambda x: rho

    # Expression-based density
    if isinstance(spec, dict) and "expression" in spec:
        expr_str = spec["expression"]
        # Build safe namespace for eval
        safe_namespace = {
            "np": np,
            "sqrt": np.sqrt,
            "exp": np.exp,
            "log": np.log,
            "log10": np.log10,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "abs": np.abs,
            "pow": pow,
            "pi": np.pi,
        }

        def expr_func(x):
            # x is array-like: x[0], x[1], (x[2] for 3D)
            local_vars = safe_namespace.copy()
            if mesh_symmetry == "axial":
                local_vars["r"] = x[0]
                local_vars["z"] = x[1]
            else:
                local_vars["x"] = x[0]
                local_vars["y"] = x[1]
                if len(x) > 2:
                    local_vars["z"] = x[2]
            # Also provide r for Cartesian as sqrt(x^2 + y^2)
            if mesh_symmetry != "axial":
                local_vars["r"] = np.sqrt(x[0]**2 + x[1]**2)
            return eval(expr_str, {"__builtins__": {}}, local_vars)

        return expr_func

    # File-based (tabulated) density
    if isinstance(spec, dict) and "file" in spec:
        file_path = spec["file"]
        skip_header = spec.get("skip_header", 0)

        # Load data
        data = np.loadtxt(file_path, skiprows=skip_header)
        n_cols = data.shape[1]

        # Auto-detect column format based on symmetry and columns
        if mesh_symmetry == "axial":
            if n_cols == 2:
                # (r, rho) - 1D radial profile
                from scipy.interpolate import interp1d
                r_data = data[:, 0]
                rho_data = data[:, 1]
                interp = interp1d(r_data, rho_data, bounds_error=False, fill_value=(rho_data[0], rho_data[-1]))

                def tabulated_func(x):
                    r = x[0]
                    return float(interp(r))

                return tabulated_func

            elif n_cols == 3:
                # (r, z, rho) - 2D axisymmetric profile
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                r_data = data[:, 0]
                z_data = data[:, 1]
                rho_data = data[:, 2]
                points = np.column_stack([r_data, z_data])
                interp = LinearNDInterpolator(points, rho_data)
                nearest = NearestNDInterpolator(points, rho_data)

                def tabulated_func(x):
                    r, z = x[0], x[1]
                    val = interp(r, z)
                    if np.isnan(val):
                        val = nearest(r, z)
                    return float(val)

                return tabulated_func

        else:  # none symmetry
            if n_cols == 3 and mesh_dimension == 2:
                # (x, y, rho) - 2D Cartesian
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                points = data[:, :2]
                rho_data = data[:, 2]
                interp = LinearNDInterpolator(points, rho_data)
                nearest = NearestNDInterpolator(points, rho_data)

                def tabulated_func(x):
                    val = interp(x[0], x[1])
                    if np.isnan(val):
                        val = nearest(x[0], x[1])
                    return float(val)

                return tabulated_func

            elif n_cols == 4 and mesh_dimension == 3:
                # (x, y, z, rho) - 3D Cartesian
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                points = data[:, :3]
                rho_data = data[:, 3]
                interp = LinearNDInterpolator(points, rho_data)
                nearest = NearestNDInterpolator(points, rho_data)

                def tabulated_func(x):
                    val = interp(x[0], x[1], x[2])
                    if np.isnan(val):
                        val = nearest(x[0], x[1], x[2])
                    return float(val)

                return tabulated_func

        raise ValueError(f"Unsupported column count {n_cols} for symmetry '{mesh_symmetry}' and dimension {mesh_dimension}")

    raise ValueError(f"Invalid density specification: {spec}")


def _choose_method(alpha: float, method: str) -> tuple[str, float]:
    """
    Choose solver method and relaxation based on alpha.

    Returns (method, relaxation_factor)
    """
    if method == "picard":
        return "picard", 1.0

    # Auto mode: use picard with relaxation based on alpha
    if alpha < 1:
        return "picard", 1.0
    elif alpha < 1000:
        # Use relaxation for moderate nonlinearity
        relax = max(0.5, 1.0 - alpha / 2000)
        return "picard", relax
    else:
        # High alpha: use strong relaxation
        return "picard", 0.5


async def handle(arguments: dict) -> list[TextContent]:
    """Handle solve tool call."""
    import dolfin as d

    # Suppress FEniCS output
    d.set_log_level(d.LogLevel.WARNING)

    # Get session and parameters
    session = get_session()
    mesh_id = arguments["mesh_id"]
    alpha = arguments["alpha"]
    density_spec = arguments["density"]
    n = arguments.get("n", 1)
    method = arguments.get("method", "auto")
    tol = arguments.get("tol", 1e-14)
    max_iter = arguments.get("max_iter", 100)
    relaxation = arguments.get("relaxation", 1.0)
    initial_guess = arguments.get("initial_guess", "constant")
    custom_id = arguments.get("custom_id")

    # Validate mesh exists
    mesh_info = session.get_mesh(mesh_id)
    if mesh_info is None:
        return [TextContent(
            type="text",
            text=f'{{"error": "Mesh \'{mesh_id}\' not found. Available meshes: {list(session.meshes.keys())}"}}'
        )]

    # Get mesh properties
    mesh_path = mesh_info.mesh_path
    mesh_dimension = mesh_info.dimension
    mesh_symmetry = mesh_info.symmetry
    regions = mesh_info.regions
    n_cells = mesh_info.n_cells

    # Warn about large meshes
    large_mesh_warning = None
    if n_cells > 100000:
        large_mesh_warning = f"Very large mesh ({n_cells:,} cells). This may take a long time."
    elif n_cells > 50000:
        large_mesh_warning = f"Large mesh ({n_cells:,} cells). This may take a while."

    # Validate density specification has required regions
    # We need a density function for each subdomain marker
    # Get max marker
    if not regions:
        return [TextContent(
            type="text",
            text='{"error": "Mesh has no region information. Cannot assign density profiles."}'
        )]

    max_marker = max(regions.values())
    n_subdomains = max_marker + 1

    # Create density functions ordered by marker
    # First, map region names to markers
    marker_to_func = {}
    density_stats = {"rho_min": float("inf"), "rho_max": float("-inf")}

    for region_name, density_value in density_spec.items():
        if region_name not in regions:
            return [TextContent(
                type="text",
                text=f'{{"error": "Region \'{region_name}\' not found in mesh. Available regions: {list(regions.keys())}"}}'
            )]

        marker = regions[region_name]
        func = _create_density_function(density_value, mesh_symmetry, mesh_dimension)
        marker_to_func[marker] = func

        # Track density stats (estimate from constant or sample expression)
        if isinstance(density_value, (int, float)):
            rho = float(density_value)
            density_stats["rho_min"] = min(density_stats["rho_min"], rho)
            density_stats["rho_max"] = max(density_stats["rho_max"], rho)

    # Check all markers have density
    for marker in range(n_subdomains):
        if marker not in marker_to_func:
            # Find which region this marker corresponds to
            missing_regions = [name for name, m in regions.items() if m == marker]
            return [TextContent(
                type="text",
                text=f'{{"error": "No density specified for marker {marker} (regions: {missing_regions}). Provide density for all regions."}}'
            )]

    # Build profiles list ordered by marker
    profiles = [marker_to_func[i] for i in range(n_subdomains)]

    # Choose method
    actual_method, auto_relaxation = _choose_method(alpha, method)
    if method == "auto":
        relaxation = auto_relaxation

    start_time = time.time()

    try:
        # Import SELCIE
        from SELCIE.DensityProfiles import DensityProfile
        from SELCIE.SolverChameleon import FieldSolver

        # Convert symmetry for SELCIE
        selcie_symmetry = _symmetry_to_selcie(mesh_symmetry)

        # Extract just the directory name from mesh_path
        # SELCIE expects: 'Saved Meshes/<filename>'
        # mesh_path is full path like '/path/to/Saved Meshes/mesh_001'
        mesh_dir = os.path.dirname(mesh_path)  # e.g., '/path/to/Saved Meshes'
        mesh_name = os.path.basename(mesh_path)  # e.g., 'mesh_001'
        parent_dir = os.path.dirname(mesh_dir)  # e.g., '/path/to'

        # Create DensityProfile
        density_profile = DensityProfile(
            filename=mesh_name,
            dimension=mesh_dimension,
            symmetry=selcie_symmetry,
            profiles=profiles,
            path=parent_dir
        )

        # Handle initial guess
        initial_field_profiles = None
        if initial_guess == "adiabatic":
            # Adiabatic: phi = rho^(-1/(n+1))
            adiabatic_profiles = []
            for func in profiles:
                def make_adiabatic(f, n_val):
                    return lambda x: pow(max(f(x), 1e-30), -1.0 / (n_val + 1))
                adiabatic_profiles.append(make_adiabatic(func, n))
            # TODO: Use InitialiseField class if needed
            # For now, use None which gives constant initial guess at minimum field value
            initial_field_profiles = None

        # Create solver
        solver = FieldSolver(
            alpha=alpha,
            n=n,
            density_profile=density_profile,
            initial_field_profiles=initial_field_profiles
        )

        # Run solver (picard iteration with optimized linear solver)
        # Use optimized linear solver for larger meshes, default for smaller
        if n_cells > 10000:
            picard_result = solver.picard(
                display_progress=False,
                tol_du=tol,
                relaxation_parameter=relaxation,
                maxiter=max_iter,
                linear_solver="krylov",
                krylov_method="cg",
                krylov_preconditioner="hypre_amg"
            )
        else:
            picard_result = solver.picard(
                display_progress=False,
                tol_du=tol,
                relaxation_parameter=relaxation,
                maxiter=max_iter,
            )

        # Extract convergence info from picard
        converged = picard_result["converged"]
        iterations = picard_result["iterations"]
        final_du_norm = picard_result["final_du_norm"]

        # Get field statistics
        field_vector = solver.field.vector()
        field_min = float(field_vector.min())
        field_max = float(field_vector.max())
        field_mean = float(np.mean(field_vector.get_local()))

        # Calculate PDE strong residual (how well the field satisfies the equation)
        # This is different from du_norm but useful for solution quality assessment
        pde_residual = None
        try:
            solver.calc_field_residual()
            if solver.residual is not None:
                res_vector = solver.residual.vector()
                pde_residual = float(d.norm(res_vector, 'linf'))
        except RuntimeError:
            pass  # Can fail for some edge cases

        # Get field at specific points
        field_at_origin = None
        try:
            if mesh_dimension == 2:
                field_at_origin = float(solver.field(0.0, 0.0))
            elif mesh_dimension == 3:
                field_at_origin = float(solver.field(0.0, 0.0, 0.0))
        except Exception:
            pass  # Point may be outside mesh

        # Compute field gradient (needed for fifth force)
        grad_computed = False
        grad_mag_min = None
        grad_mag_max = None
        try:
            solver.calc_field_grad_vector()
            solver.calc_field_grad_mag()
            grad_computed = True
            if solver.field_grad_mag is not None:
                grad_mag_min = float(solver.field_grad_mag.vector().min())
                grad_mag_max = float(solver.field_grad_mag.vector().max())
        except Exception:
            pass  # Gradient calculation can fail for some configurations

        # Save solution info to session
        solution_id = session.generate_solution_id(custom_id)

        # Save field and gradient data to HDF5
        solution_dir = os.path.join(parent_dir, "Saved Solutions")
        os.makedirs(solution_dir, exist_ok=True)
        solution_path = os.path.join(solution_dir, solution_id)
        os.makedirs(solution_path, exist_ok=True)

        # Save field to HDF5
        with d.HDF5File(solver.mesh.mpi_comm(), os.path.join(solution_path, "field.h5"), "w") as f:
            f.write(solver.field, "field")

        # Save field gradient (vector) if computed
        if grad_computed and solver.field_grad is not None:
            with d.HDF5File(solver.mesh.mpi_comm(), os.path.join(solution_path, "field_grad.h5"), "w") as f:
                f.write(solver.field_grad, "field_grad")

        # Save gradient magnitude if computed
        if grad_computed and solver.field_grad_mag is not None:
            with d.HDF5File(solver.mesh.mpi_comm(), os.path.join(solution_path, "field_grad_mag.h5"), "w") as f:
                f.write(solver.field_grad_mag, "field_grad_mag")

        # Store solution info
        solution_info = SolutionInfo(
            solution_id=solution_id,
            mesh_id=mesh_id,
            profile_id="",  # No separate profile
            alpha=alpha,
            n=n,
            converged=converged,
            iterations=iterations,
            final_residual=final_du_norm,
            field_min=field_min,
            field_max=field_max
        )
        session.add_solution(solution_info)

        runtime = time.time() - start_time

        # Build response
        result = {
            "solution_id": solution_id,
            "mesh_id": mesh_id,
            "alpha": alpha,
            "n": n,
            "density_stats": {
                "rho_min": density_stats["rho_min"] if density_stats["rho_min"] != float("inf") else None,
                "rho_max": density_stats["rho_max"] if density_stats["rho_max"] != float("-inf") else None
            },
            "status": "converged" if converged else "max_iterations",
            "iterations": iterations,
            "final_du_norm": final_du_norm,
            "pde_residual": pde_residual,
            "method_used": actual_method,
            "relaxation_used": relaxation,
            "initial_guess": initial_guess,
            "field_stats": {
                "min": field_min,
                "max": field_max,
                "mean": field_mean,
                "at_origin": field_at_origin
            },
            "gradient_stats": {
                "computed": grad_computed,
                "magnitude_min": grad_mag_min,
                "magnitude_max": grad_mag_max,
            },
            "saved_files": {
                "field": "field.h5",
                "field_grad": "field_grad.h5" if grad_computed else None,
                "field_grad_mag": "field_grad_mag.h5" if grad_computed else None,
            },
            "solution_path": solution_path,
            "runtime_seconds": round(runtime, 2)
        }

        # Add warning if mesh was large
        if large_mesh_warning:
            result["warning"] = large_mesh_warning

        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        import traceback
        error_result = {
            "solution_id": None,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "suggestion": "Check density specification and mesh compatibility. Try reducing alpha or using picard method with relaxation < 1."
        }
        import json
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
