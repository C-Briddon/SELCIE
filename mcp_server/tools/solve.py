#!/usr/bin/env python3
"""Solve tool for SELCIE MCP server."""

import json
import math
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

All values are dimensionless. If using physical parameters from calculate_physical_parameters:
- Density: ρ̂ = ρ_physical / rho_0
- Coordinates in mesh: x̂ = x_physical / L

Parameters:
- mesh_id: Reference to a previously created mesh
- alpha: Dimensionless coupling constant (from calculate_physical_parameters)
- density: Dimensionless density ρ̂ = ρ/ρ₀ per region (e.g., if ρ₀ = vacuum density, then vacuum → 1.0)
- n: Potential power (default: 1)
- tol: Convergence tolerance (default: 1e-14)
- max_iter: Maximum iterations (default: 100)
- relaxation: Relaxation factor for Picard iteration (0-1]. In tests 1 performs well, and is faster, so is recommeneded. Default: 1.0
- initial_guess: "constant" (default, recommended for SELCIE), "adiabatic" (start from φ = ρ̂^{-1/(n+1)}; good for smooth density profiles, can diverge for discontinuous region densities), "previous" (start from an earlier solution on the same mesh; see initial_guess_solution_id), or "boundary" (uniform at the dirichlet_bc value; good warm start for weakly-perturbed/unscreened solves)
- dirichlet_bc: optional φ̂ value pinned on the outer domain boundary (radial geometries only). Without it the solve uses natural (no-flux) BCs everywhere and the field level floats to the box-average equilibrium ⟨φ̂^{-(n+1)}⟩ = ⟨ρ̂⟩, which depends on domain size. The symmetry axis always keeps the natural condition.
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
                "enum": ["constant", "adiabatic", "previous", "boundary"],
                "description": "Initial guess strategy. 'constant' starts from the minimum field value (recommended for SELCIE). 'adiabatic' starts from φ = ρ̂^{-1/(n+1)} per region; converges faster for smooth density profiles (e.g. NFW) but can diverge when region densities are discontinuous. 'previous' starts from an earlier solution on the same mesh (see initial_guess_solution_id). 'boundary' starts uniformly at the dirichlet_bc value (requires dirichlet_bc; good for unscreened/weakly-perturbed solves). Default: constant, or boundary when dirichlet_bc is set (the constant start diverges against a pinned boundary).",
                "default": "constant"
            },
            "dirichlet_bc": {
                "type": "number",
                "description": "Optional Dirichlet boundary value φ̂ (> 0) pinned on the outer domain boundary. Supported for geometries with a circular outer boundary (domain_radius param). The symmetry axis keeps the natural (no-flux) condition. Default: none (natural BCs; field level floats with domain size). For strongly screened/low-α regimes, the robust recipe is: first solve WITHOUT dirichlet_bc (natural BCs; the floating level acts as a free continuation), then re-solve with dirichlet_bc + initial_guess='previous' — pinning the boundary directly from a cold start can diverge."
            },
            "initial_guess_solution_id": {
                "type": "string",
                "description": "Solution ID to start from when initial_guess='previous'. Must be a solution on the same mesh with the same deg_V. Default: most recent solution on this mesh."
            },
            "custom_id": {
                "type": "string",
                "description": "Custom solution ID. Default: auto-generated"
            },
            "deg_V": {
                "type": "integer",
                "description": "Function space degree (1=CG1, 2=CG2). Default: 2",
                "default": 2
            },
            "display_progress": {
                "type": "boolean",
                "description": "Print Picard iteration progress to stderr/stdout. Default: false",
                "default": False
            },
            "linear_solver": {
                "type": "string",
                "enum": ["default", "krylov"],
                "description": "Linear solver for large meshes. 'default' uses SELCIE's original solve; 'krylov' uses iterative Krylov solving. Default: krylov",
                "default": "krylov"
            },
            "krylov_method": {
                "type": "string",
                "description": "Krylov method used when linear_solver='krylov'. Default: cg",
                "default": "cg"
            },
            "krylov_preconditioner": {
                "type": "string",
                "description": "Krylov preconditioner used when linear_solver='krylov'. Default: hypre_amg",
                "default": "hypre_amg"
            },
            "debug": {
                "type": "boolean",
                "description": "Include Python tracebacks in error responses. Default: false",
                "default": False
            }
        },
        "required": ["mesh_id", "alpha", "density"]
    }
)


def _finite_or_none(value):
    """Map non-finite floats to None so responses stay valid JSON."""
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _symmetry_to_selcie(symmetry: str) -> str:
    """Convert MCP symmetry to SELCIE DensityProfile symmetry string."""
    mapping = {
        "axial": "vertical axis-symmetry",
        "translation": "translation symmetry",
    }
    return mapping.get(symmetry, symmetry)


def _prepare_density_spec(density_spec: dict, regions: dict[str, int]) -> dict:
    """Return a solver-ready density spec without mutating caller input."""
    prepared = dict(density_spec)

    # Expand "object" to all object_* regions if mesh has multi-region objects
    if "object" in prepared and "object" not in regions:
        object_regions = [r for r in regions.keys() if r.startswith("object_")]
        if object_regions:
            object_density = prepared.pop("object")
            for obj_region in object_regions:
                if obj_region not in prepared:
                    prepared[obj_region] = object_density

    # Auto-assign vacuum density to measuring_boundary if not explicitly provided.
    # This makes the common case seamless: measurement regions are physically vacuum.
    if "measuring_boundary" in regions and "measuring_boundary" not in prepared:
        if "vacuum" in prepared:
            prepared["measuring_boundary"] = prepared["vacuum"]

    return prepared


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
    initial_guess = arguments.get("initial_guess")
    dirichlet_bc = arguments.get("dirichlet_bc")
    if initial_guess is None:
        # the constant (minimum-value) start diverges against a pinned
        # boundary far above the minimum, so Dirichlet solves default to
        # the boundary-value start instead
        initial_guess = "boundary" if dirichlet_bc is not None else "constant"
    custom_id = arguments.get("custom_id")
    deg_V = arguments.get("deg_V", 2)
    # MCP cannot have print values, this is for debugging only
    display_progress = arguments.get("display_progress", False)
    linear_solver = arguments.get("linear_solver", "krylov")
    krylov_method = arguments.get("krylov_method", "cg")
    krylov_preconditioner = arguments.get("krylov_preconditioner", "hypre_amg")
    debug = arguments.get("debug", False)

    if linear_solver not in ("default", "krylov"):
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": {
                    "code": "INVALID_PARAMETER",
                    "message": "linear_solver must be 'default' or 'krylov'",
                }
            }, indent=2)
        )]

    # Validate mesh exists
    mesh_info = session.get_mesh(mesh_id)
    if mesh_info is None:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": {
                    "code": "MESH_NOT_FOUND",
                    "message": f"Mesh '{mesh_id}' not found. Available meshes: {list(session.meshes.keys())}",
                }
            }, indent=2)
        )]

    # Get mesh properties
    mesh_path = mesh_info.mesh_path
    mesh_dimension = mesh_info.dimension
    mesh_symmetry = mesh_info.symmetry
    mesh_geometry = mesh_info.geometry
    regions = mesh_info.regions
    n_cells = mesh_info.n_cells

    # Validate Dirichlet BC request (outer boundary identified by domain_radius)
    bc_outer_radius = None
    if dirichlet_bc is not None:
        if not (isinstance(dirichlet_bc, (int, float)) and dirichlet_bc > 0):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": {
                        "code": "INVALID_PARAMETER",
                        "message": "dirichlet_bc must be a positive number (chameleon field φ̂ > 0).",
                    }
                }, indent=2)
            )]
        bc_outer_radius = (mesh_info.params or {}).get("domain_radius")
        if bc_outer_radius is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": {
                        "code": "DIRICHLET_UNSUPPORTED",
                        "message": f"dirichlet_bc requires a geometry with a circular outer boundary (domain_radius param); geometry '{mesh_geometry}' has none.",
                    }
                }, indent=2)
            )]
    if initial_guess == "boundary" and dirichlet_bc is None:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": {
                    "code": "INVALID_PARAMETER",
                    "message": "initial_guess='boundary' requires dirichlet_bc to be set.",
                }
            }, indent=2)
        )]

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
            text=json.dumps({
                "error": {
                    "code": "NO_REGIONS",
                    "message": "Mesh has no region information. Cannot assign density profiles.",
                }
            }, indent=2)
        )]

    max_marker = max(regions.values())
    n_subdomains = max_marker + 1

    # Create density functions ordered by marker
    # First, map region names to markers
    marker_to_func = {}
    density_stats = {"rho_min": float("inf"), "rho_max": float("-inf")}

    density_spec = _prepare_density_spec(density_spec, regions)

    for region_name, density_value in density_spec.items():
        if region_name not in regions:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": {
                        "code": "UNKNOWN_REGION",
                        "message": f"Region '{region_name}' not found in mesh. Available regions: {list(regions.keys())}",
                    }
                }, indent=2)
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
                text=json.dumps({
                    "error": {
                        "code": "MISSING_DENSITY",
                        "message": f"No density specified for marker {marker} (regions: {missing_regions}). Provide density for all regions.",
                    }
                }, indent=2)
            )]

    # Build profiles list ordered by marker
    profiles = [marker_to_func[i] for i in range(n_subdomains)]

    # Resolve previous solution for initial_guess='previous' before starting
    # the (potentially expensive) solve
    prev_solution_info = None
    prev_field_path = None
    if initial_guess == "previous":
        prev_id = arguments.get("initial_guess_solution_id")
        if prev_id is not None:
            prev_solution_info = session.get_solution(prev_id)
            if prev_solution_info is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": {
                            "code": "SOLUTION_NOT_FOUND",
                            "message": f"initial_guess_solution_id '{prev_id}' not found. Available solutions: {list(session.solutions.keys())}",
                        }
                    }, indent=2)
                )]
        else:
            same_mesh = [s for s in session.solutions.values() if s.mesh_id == mesh_id]
            if same_mesh:
                prev_solution_info = same_mesh[-1]

        if prev_solution_info is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": {
                        "code": "INITIAL_GUESS_UNAVAILABLE",
                        "message": f"initial_guess='previous' requires an existing solution on mesh '{mesh_id}', but none was found. Solve once with initial_guess='constant' first.",
                    }
                }, indent=2)
            )]

        if prev_solution_info.mesh_id != mesh_id:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": {
                        "code": "INITIAL_GUESS_MESH_MISMATCH",
                        "message": f"Solution '{prev_solution_info.solution_id}' was computed on mesh '{prev_solution_info.mesh_id}', not '{mesh_id}'. The previous solution must be on the same mesh.",
                    }
                }, indent=2)
            )]

        if prev_solution_info.deg_V != deg_V:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": {
                        "code": "INITIAL_GUESS_DEGREE_MISMATCH",
                        "message": f"Solution '{prev_solution_info.solution_id}' used deg_V={prev_solution_info.deg_V} but this solve uses deg_V={deg_V}. Set deg_V={prev_solution_info.deg_V} or choose a different solution.",
                    }
                }, indent=2)
            )]

        prev_field_path = os.path.join(
            os.path.dirname(os.path.dirname(mesh_info.mesh_path)),
            "Saved Solutions", prev_solution_info.solution_id, "field.h5"
        )
        if not os.path.exists(prev_field_path):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": {
                        "code": "INITIAL_GUESS_UNAVAILABLE",
                        "message": f"Field file for solution '{prev_solution_info.solution_id}' not found at {prev_field_path}. It may have been cleared.",
                    }
                }, indent=2)
            )]

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

        # Dirichlet BC on the outer boundary: label it 1 (axis/rest stays 0,
        # keeping the natural condition there). Facets are marked from their
        # vertices alone (check_midpoint=False): mesh vertices sit on the
        # circle to float precision, while chord midpoints sag inward and
        # would silently fail the radius test on coarse meshes. Axis facets
        # always have an interior vertex that fails the test.
        BCs = None
        if dirichlet_bc is not None:
            bc_r_cut = (1.0 - 1e-4) * float(bc_outer_radius)

            def _outer_boundary(x, _r=bc_r_cut):
                return float(np.linalg.norm(x)) >= _r

            density_profile.assign_boundary_labels([_outer_boundary],
                                                   check_midpoint=False)
            n_bc_facets = int(np.sum(
                density_profile.boundary.array() == 1))
            if n_bc_facets == 0:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": {
                            "code": "DIRICHLET_NOT_APPLIED",
                            "message": "No outer-boundary facets were labeled for the Dirichlet BC; the solve would silently run with natural BCs. Check the mesh geometry (domain_radius) or report this as a bug.",
                        }
                    }, indent=2)
                )]
            BCs = [None, ("Dirichlet", repr(float(dirichlet_bc)))]

        # Handle initial guess
        # FieldSolver wraps initial_field_profiles (one function per
        # subdomain, ordered by marker) in SELCIE's InitialiseField
        initial_field_profiles = None
        if initial_guess == "adiabatic":
            # Adiabatic: phi = rho^(-1/(n+1))
            def make_adiabatic(f, n_val):
                return lambda x: pow(max(f(x), 1e-30), -1.0 / (n_val + 1))
            initial_field_profiles = [make_adiabatic(func, n) for func in profiles]
        elif initial_guess == "boundary":
            bc_val = float(dirichlet_bc)
            initial_field_profiles = [
                (lambda x, _v=bc_val: _v) for _ in profiles
            ]

        # Create solver
        solver = FieldSolver(
            alpha=alpha,
            n=n,
            density_profile=density_profile,
            initial_field_profiles=initial_field_profiles,
            deg_V=deg_V
        )

        # Start from a previous solution's field if requested
        if prev_field_path is not None:
            with d.HDF5File(solver.mesh.mpi_comm(), prev_field_path, "r") as f:
                f.read(solver.field, "field")

        # Project density to DG0 before solving (stats, saving, and the
        # maximum-principle bound check on the solved field)
        density_func = None
        density_min = None
        density_max = None
        try:
            V_dg = d.FunctionSpace(solver.mesh, "DG", 0)
            density_func = d.Function(V_dg)
            density_func.interpolate(density_profile)
            density_min = float(density_func.vector().min())
            density_max = float(density_func.vector().max())
        except Exception:
            pass  # Density interpolation can fail in some edge cases

        # Run solver (picard iteration with optimized linear solver)
        # Use optimized linear solver for larger meshes, default for smaller
        if n_cells > 10000:
            picard_result = solver.picard(
                display_progress=display_progress,
                BCs=BCs,
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
                BCs=BCs,
                tol_du=tol,
                relaxation_parameter=relaxation,
                maxiter=max_iter,
            )

        # Extract convergence info from picard
        converged = picard_result["converged"]
        iterations = picard_result["iterations"]
        final_du_norm = picard_result["final_du_norm"]
        diverged = not converged and not math.isfinite(final_du_norm)

        # Get field statistics
        field_vector = solver.field.vector()
        field_min = float(field_vector.min())
        field_max = float(field_vector.max())
        field_mean = float(np.mean(field_vector.get_local()))

        # Physicality guards (a "converged" status does not guarantee a
        # physical solution; the chameleon field must be positive and is
        # bounded above by the background equilibrium)
        solution_warnings = []
        if math.isfinite(field_min) and field_min <= 0:
            solution_warnings.append(
                "Field has non-positive values - unphysical for the chameleon (phi must be > 0). Treat this solution as invalid. Robust recipe for screened regimes: solve WITHOUT dirichlet_bc first (natural BCs), then re-solve with dirichlet_bc + initial_guess='previous'.")
        bound_candidates = []
        if dirichlet_bc is not None:
            bound_candidates.append(float(dirichlet_bc))
        if density_min is not None and density_min > 0:
            bound_candidates.append(density_min ** (-1.0 / (n + 1)))
        if bound_candidates and math.isfinite(field_max) and \
                field_max > 1.5 * max(bound_candidates):
            solution_warnings.append(
                f"Field maximum ({field_max:.3e}) exceeds the maximum-principle bound (~{max(bound_candidates):.3e}) - likely a runaway iteration; treat as unphysical.")

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

        # Save the density field projected before the solve
        density_saved = False
        if density_func is not None:
            try:
                with d.HDF5File(solver.mesh.mpi_comm(), os.path.join(solution_path, "density.h5"), "w") as f:
                    f.write(density_func, "density")
                density_saved = True
            except Exception:
                pass  # Saving can fail in some edge cases

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
            "status": "converged" if converged else ("diverged" if diverged else "max_iterations_not_converged"),
            "iterations": iterations,
            "final_du_norm": _finite_or_none(final_du_norm),
            "pde_residual": _finite_or_none(pde_residual),
            "method_used": "picard",
            "relaxation_used": relaxation,
            "initial_guess": initial_guess,
            "dirichlet_bc": dirichlet_bc,
            "warnings": solution_warnings or None,
            "initial_guess_solution_id": prev_solution_info.solution_id if prev_solution_info else None,
            "field_stats": {
                "min": _finite_or_none(field_min),
                "max": _finite_or_none(field_max),
                "mean": _finite_or_none(field_mean),
                "at_origin": _finite_or_none(field_at_origin)
            },
            "gradient_stats": {
                "computed": grad_computed,
                "magnitude_min": _finite_or_none(grad_mag_min),
                "magnitude_max": _finite_or_none(grad_mag_max),
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

        if diverged:
            result["suggestion"] = (
                "Solver diverged (du_norm is not finite). Try initial_guess='constant' "
                "(recommended for discontinuous densities), relaxation < 1, or check "
                "the density specification."
            )

        # Add warning if mesh was large
        if large_mesh_warning:
            result["warning"] = large_mesh_warning

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {
            "solution_id": None,
            "status": "failed",
            "error": str(e),
            "suggestion": "Check density specification and mesh compatibility. Try relaxation < 1 for difficult convergence. If geometry has small holes or high curvature, try increasing mesh resolution (e.g., 'fine' or 'very_fine')."
        }
        if debug:
            import traceback
            error_result["traceback"] = traceback.format_exc()
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
