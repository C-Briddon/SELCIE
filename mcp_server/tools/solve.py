#!/usr/bin/env python3
"""Solve tool for SELCIE MCP server."""

import json
import os
import time

import numpy as np
from mcp.types import TextContent, Tool

from utils.session import get_session, SolutionInfo
from utils.density import create_density_function, SPHERICAL_GEOMETRIES


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
- tol: Convergence tolerance (default: 1e-14)
- max_iter: Maximum iterations (default: 100)
- relaxation: Relaxation factor for Picard iteration (0-1]. In tests 1 performs well, and is faster, so is recommeneded. Default: 1.0
- initial_guess: "constant" (default, recommended for SELCIE), "adiabatic", or "previous"
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
                "description": "Dimensionless density ρ̂ = ρ/ρ₀ per region. Value is: number, {expression: str}, or {file: str, format?: 'tabulated'|'grid', columns?: int[], bounds?: number[], skip_header?: int, npz_key?: str}. For tabulated: columns selects columns (1-based). For grid: bounds maps grid to spatial coordinates. For expressions: spherical geometries use r=spherical radius.",
                "additionalProperties": True
            },
            "n": {
                "type": "integer",
                "description": "Potential power index. Default: 1",
                "default": 1
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
                "description": "Relaxation factor for Picard iteration (0-1]. In tests 1 performs well, and is faster, so is recommeneded. Default: 1.0",
                "default": 1.0
            },
            "initial_guess": {
                "type": "string",
                "enum": ["constant", "adiabatic", "previous"],
                "description": "Initial guess strategy. Default: constant (recommended for SELCIE)",
                "default": "constant"
            },
            "custom_id": {
                "type": "string",
                "description": "Custom solution ID. Default: auto-generated"
            },
            "deg_V": {
                "type": "integer",
                "description": "Function space degree (1=CG1, 2=CG2). Default: 2",
                "default": 2
            }
        },
        "required": ["mesh_id", "alpha", "density"]
    }
)


def _symmetry_to_selcie(symmetry: str) -> str:
    """Convert MCP symmetry to SELCIE DensityProfile symmetry string."""
    mapping = {
        "axial": "vertical axis-symmetry",
        "translation": "translation symmetry",
    }
    return mapping.get(symmetry, symmetry)


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
    tol = arguments.get("tol", 1e-14)
    max_iter = arguments.get("max_iter", 100)
    relaxation = arguments.get("relaxation", 1.0)
    initial_guess = arguments.get("initial_guess", "constant")
    custom_id = arguments.get("custom_id")
    deg_V = arguments.get("deg_V", 2)
    # MCP cannot have print values, this is for debugging only
    display_progress = arguments.get("display_progress", False)
    linear_solver = arguments.get("linear_solver", "krylov")
    krylov_method = arguments.get("krylov_method", "cg")
    krylov_preconditioner = arguments.get("krylov_preconditioner", "hypre_amg")

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
    mesh_geometry = mesh_info.geometry
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
        try:
            func = create_density_function(density_value, mesh_symmetry, mesh_dimension, mesh_geometry)
        except ValueError as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "INVALID_EXPRESSION", "message": str(e)}}, indent=2)
            )]
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
            initial_field_profiles=initial_field_profiles,
            deg_V=deg_V
        )

        # Run solver (picard iteration with optimized linear solver)
        # Use optimized linear solver for larger meshes, default for smaller
        if n_cells > 10000:
            picard_result = solver.picard(
                display_progress=display_progress,
                tol_du=tol,
                relaxation_parameter=relaxation,
                maxiter=max_iter,
                linear_solver=linear_solver,
                krylov_method=krylov_method,
                krylov_preconditioner=krylov_preconditioner
            )
        else:
            picard_result = solver.picard(
                display_progress=display_progress,
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

        # Project and save density field
        density_saved = False
        density_min = None
        density_max = None
        try:
            # Create DG0 space for piecewise constant density (matches region-based definition)
            V_dg = d.FunctionSpace(solver.mesh, "DG", 0)
            density_func = d.Function(V_dg)

            # Interpolate density_profile (UserExpression) onto DG0 space
            density_func.interpolate(density_profile)

            # Get density statistics
            density_min = float(density_func.vector().min())
            density_max = float(density_func.vector().max())

            # Save to HDF5
            with d.HDF5File(solver.mesh.mpi_comm(), os.path.join(solution_path, "density.h5"), "w") as f:
                f.write(density_func, "density")
            density_saved = True
        except Exception:
            pass  # Density interpolation can fail in some edge cases

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
            field_max=field_max,
            deg_V=deg_V,
            symmetry=mesh_symmetry
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
                "saved": density_saved,
                "rho_min": density_min if density_saved else (density_stats["rho_min"] if density_stats["rho_min"] != float("inf") else None),
                "rho_max": density_max if density_saved else (density_stats["rho_max"] if density_stats["rho_max"] != float("-inf") else None)
            },
            "status": "converged" if converged else "max_iterations_not_converged",
            "iterations": iterations,
            "final_du_norm": final_du_norm,
            "pde_residual": pde_residual,
            "method_used": "picard",
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
                "density": "density.h5" if density_saved else None,
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
            "suggestion": "Check density specification and mesh compatibility. Try relaxation < 1 for difficult convergence. If geometry has small holes or high curvature, try increasing mesh resolution (e.g., 'fine' or 'very_fine')."
        }
        import json
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
