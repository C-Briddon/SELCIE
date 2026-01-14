#!/usr/bin/env python3
"""Evaluate tool for SELCIE MCP server."""

import json
import os
from typing import Any

import numpy as np
from mcp.types import TextContent, Tool

from utils.session import get_session


TOOL_DEFINITION = Tool(
    name="evaluate",
    description="""Evaluate field values and derived quantities at specified locations.

Modes:
- radial: Sample along radial direction from origin
- line: Sample along arbitrary line between two points
- points: Evaluate at specific coordinates
- grid: Sample on regular 2D grid
- max_in_region: Find max/min values within a region, with optional minimum distance from other region(s). Use min_distance_from='all' to exclude points near any other domain boundary.
- integrate: Compute volume integrals over a region. Returns total force, mass, volume. Essential for torsion balance experiments, Casimir force measurements, and any extended object where thin-shell effects matter.

Quantities (for point-based modes):
- field: Chameleon field φ
- gradient_magnitude: |∇φ| (dimensionless). Multiply by grad_to_acceleration_g from calculate_physical_parameters to get acceleration in units of g.
- density: ρ̂ at evaluation points (if available)
- adiabatic_field: ρ̂^{-1/(n+1)} for comparison
- field_deviation: (φ - φ_adiabatic) / φ_adiabatic

Quantities (for integrate mode) - all in rescaled (dimensionless) units:
- force: Total force F = ∫ρ̂∇φ̂ dV̂ on region (includes symmetry Jacobian). Multiply by force_scale_N from calculate_physical_parameters to get Newtons.
- mass: Total mass M = ∫ρ̂ dV̂. Multiply by mass_scale_kg from calculate_physical_parameters to get kg.
""",
    inputSchema={
        "type": "object",
        "properties": {
            "solution_id": {
                "type": "string",
                "description": "ID of the solution to evaluate"
            },
            "mode": {
                "type": "string",
                "enum": ["radial", "line", "points", "grid", "max_in_region", "integrate"],
                "description": "Evaluation mode"
            },
            "params": {
                "type": "object",
                "description": "Mode-specific parameters",
                "properties": {
                    "n_points": {"type": "integer", "description": "Number of sample points (radial, line)"},
                    "r_min": {"type": "number", "description": "Minimum radius (radial)"},
                    "r_max": {"type": "number", "description": "Maximum radius (radial)"},
                    "log_spacing": {"type": "boolean", "description": "Use log spacing (radial)"},
                    "direction": {"type": "array", "description": "Direction vector [r, z] (radial)"},
                    "start": {"type": "array", "description": "Start point (line)"},
                    "end": {"type": "array", "description": "End point (line)"},
                    "coordinates": {"type": "array", "description": "List of [r, z] points (points)"},
                    "r_range": {"type": "array", "description": "[r_min, r_max] (grid)"},
                    "z_range": {"type": "array", "description": "[z_min, z_max] (grid)"},
                    "n_r": {"type": "integer", "description": "Number of r points (grid)"},
                    "n_z": {"type": "integer", "description": "Number of z points (grid)"},
                    "region": {"type": "string", "description": "Region name (max_in_region, integrate). Default: vacuum"},
                    "quantity": {"type": "string", "enum": ["force", "mass"], "description": "Quantity to integrate (integrate mode). Default: force"},
                    "min_distance_from": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}}
                        ],
                        "description": "Region(s) to keep distance from (max_in_region). Can be: a region name, 'all' for all other regions, or a list of region names"
                    },
                    "min_distance": {"type": "number", "description": "Minimum distance from boundary of exclusion region(s) (max_in_region)"},
                    "n_samples": {"type": "integer", "description": "Number of random samples (max_in_region). Default: 1000"}
                }
            },
            "quantities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Quantities to compute. Default: ['field', 'gradient_magnitude']"
            }
        },
        "required": ["solution_id", "mode"]
    }
)


def _generate_radial_points(params: dict, mesh_bounds: dict) -> np.ndarray:
    """Generate points along radial direction."""
    n_points = params.get("n_points", 200)
    r_min = params.get("r_min", 0.01)
    r_max = params.get("r_max", mesh_bounds.get("r_max", 1.0))
    log_spacing = params.get("log_spacing", True)
    direction = params.get("direction", [1, 0])

    # Normalize direction
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    # Generate radii
    if log_spacing and r_min > 0:
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
    else:
        radii = np.linspace(r_min, r_max, n_points)

    # Generate points along direction
    points = np.outer(radii, direction)
    return points, radii


def _generate_line_points(params: dict) -> np.ndarray:
    """Generate points along a line."""
    start = np.array(params["start"])
    end = np.array(params["end"])
    n_points = params.get("n_points", 200)

    t = np.linspace(0, 1, n_points)
    points = start + np.outer(t, end - start)
    distances = np.linalg.norm(points - start, axis=1)
    return points, distances


def _generate_grid_points(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate points on a regular grid."""
    r_range = params["r_range"]
    z_range = params["z_range"]
    n_r = params.get("n_r", 50)
    n_z = params.get("n_z", 50)

    r = np.linspace(r_range[0], r_range[1], n_r)
    z = np.linspace(z_range[0], z_range[1], n_z)
    R, Z = np.meshgrid(r, z)

    points = np.column_stack([R.ravel(), Z.ravel()])
    return points, r, z


async def handle(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle evaluate tool calls."""
    try:
        # Import FEniCS here to avoid import issues
        import dolfin as d

        solution_id = arguments["solution_id"]
        mode = arguments["mode"]
        params = arguments.get("params", {})
        quantities = arguments.get("quantities", ["field", "gradient_magnitude"])

        # Get session and solution info
        session = get_session()
        solution_info = session.get_solution(solution_id)

        if solution_info is None:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "SOLUTION_NOT_FOUND",
                    "message": f"Solution '{solution_id}' not found",
                    "available_solutions": list(session.solutions.keys())
                }
            }, indent=2))]

        # Get mesh info
        mesh_info = session.get_mesh(solution_info.mesh_id)
        if mesh_info is None:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "MESH_NOT_FOUND",
                    "message": f"Mesh '{solution_info.mesh_id}' for solution not found"
                }
            }, indent=2))]

        # Load the mesh
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

        # Determine solution path (same root as mesh structure)
        # mesh_path is like /tmp/selcie_meshes/Saved Meshes/mesh_001
        # We need /tmp/selcie_meshes/Saved Solutions/solution_001
        mesh_dir = os.path.dirname(mesh_path)  # /tmp/selcie_meshes/Saved Meshes
        root_dir = os.path.dirname(mesh_dir)   # /tmp/selcie_meshes
        solution_path = os.path.join(root_dir, "Saved Solutions", solution_id)
        field_file = os.path.join(solution_path, "field.h5")

        if not os.path.exists(field_file):
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "FIELD_NOT_FOUND",
                    "message": f"Field file not found at {field_file}"
                }
            }, indent=2))]

        with d.HDF5File(mesh.mpi_comm(), field_file, "r") as f:
            f.read(field, "field")

        # Load density if available and needed
        density = None
        density_file = os.path.join(solution_path, "density.h5")
        needs_density = any(q in quantities for q in ["density", "adiabatic_field", "field_deviation"])
        if needs_density and os.path.exists(density_file):
            try:
                # Density is stored in DG0 space
                V_dg = d.FunctionSpace(mesh, "DG", 0)
                density = d.Function(V_dg)
                with d.HDF5File(mesh.mpi_comm(), density_file, "r") as f:
                    f.read(density, "density")
            except Exception:
                density = None

        # Compute gradient directly from field (more accurate than loading pre-computed)
        field_grad = None
        if "gradient_magnitude" in quantities:
            # Project grad(field) onto vector space with same degree
            V_vec = d.VectorFunctionSpace(mesh, "CG", deg_V)
            field_grad = d.project(d.grad(field), V_vec)

        # Get mesh bounds for radial mode
        coords = mesh.coordinates()
        mesh_bounds = {
            "r_min": float(coords[:, 0].min()),
            "r_max": float(coords[:, 0].max()),
            "z_min": float(coords[:, 1].min()),
            "z_max": float(coords[:, 1].max())
        }

        # Generate evaluation points based on mode
        if mode == "radial":
            points, position_values = _generate_radial_points(params, mesh_bounds)
            position_key = "r"
        elif mode == "line":
            if "start" not in params or "end" not in params:
                return [TextContent(type="text", text=json.dumps({
                    "error": {
                        "code": "INVALID_PARAMS",
                        "message": "Line mode requires 'start' and 'end' parameters"
                    }
                }, indent=2))]
            points, position_values = _generate_line_points(params)
            position_key = "distance"
        elif mode == "points":
            if "coordinates" not in params:
                return [TextContent(type="text", text=json.dumps({
                    "error": {
                        "code": "INVALID_PARAMS",
                        "message": "Points mode requires 'coordinates' parameter"
                    }
                }, indent=2))]
            points = np.array(params["coordinates"])
            position_values = None
            position_key = None
        elif mode == "grid":
            if "r_range" not in params or "z_range" not in params:
                return [TextContent(type="text", text=json.dumps({
                    "error": {
                        "code": "INVALID_PARAMS",
                        "message": "Grid mode requires 'r_range' and 'z_range' parameters"
                    }
                }, indent=2))]
            try:
                points, r_values, z_values = _generate_grid_points(params)
            except KeyError as e:
                return [TextContent(type="text", text=json.dumps({
                    "error": {
                        "code": "INVALID_PARAMS",
                        "message": f"Grid mode missing required parameter: {e}"
                    }
                }, indent=2))]
            position_key = "grid"
        elif mode == "max_in_region":
            return await _handle_max_in_region(
                arguments, mesh, field, field_grad, mesh_info, solution_info, mesh_bounds
            )
        elif mode == "integrate":
            return await _handle_integrate(
                arguments, mesh, field, mesh_info, solution_info
            )
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "INVALID_MODE",
                    "message": f"Unknown mode: {mode}"
                }
            }, indent=2))]

        # Evaluate quantities at points
        n_points = len(points)
        data = {}
        valid_mask = np.ones(n_points, dtype=bool)

        # Evaluate field at all points
        field_values = np.zeros(n_points)
        for i, pt in enumerate(points):
            try:
                field_values[i] = field(pt[0], pt[1])
            except RuntimeError:
                # Point outside mesh
                field_values[i] = np.nan
                valid_mask[i] = False

        if "field" in quantities:
            data["field"] = field_values.tolist()

        # Evaluate gradient magnitude (compute on-the-fly from gradient vector)
        if "gradient_magnitude" in quantities and field_grad is not None:
            grad_values = np.zeros(n_points)
            for i, pt in enumerate(points):
                try:
                    grad_vec = field_grad(pt[0], pt[1])
                    grad_values[i] = np.linalg.norm(grad_vec)
                except RuntimeError:
                    grad_values[i] = np.nan

            data["gradient_magnitude"] = grad_values.tolist()

        # Evaluate density at points
        if "density" in quantities:
            if density is not None:
                density_values = np.zeros(n_points)
                for i, pt in enumerate(points):
                    try:
                        density_values[i] = density(pt[0], pt[1])
                    except RuntimeError:
                        density_values[i] = np.nan
                data["density"] = density_values.tolist()
            else:
                data["density_note"] = "Density not saved with this solution (re-solve to generate)"

        # Adiabatic field: phi_adiabatic = rho^{-1/(n+1)}
        if "adiabatic_field" in quantities:
            if density is not None:
                n_power = solution_info.n
                density_values = np.zeros(n_points)
                adiabatic_values = np.zeros(n_points)
                for i, pt in enumerate(points):
                    try:
                        rho = density(pt[0], pt[1])
                        density_values[i] = rho
                        # Avoid division by zero
                        if rho > 0:
                            adiabatic_values[i] = pow(rho, -1.0 / (n_power + 1))
                        else:
                            adiabatic_values[i] = np.nan
                    except RuntimeError:
                        adiabatic_values[i] = np.nan
                data["adiabatic_field"] = adiabatic_values.tolist()
            else:
                data["adiabatic_field_note"] = "Density not saved with this solution (re-solve to generate)"

        # Field deviation from adiabatic: (phi - phi_adiabatic) / phi_adiabatic
        if "field_deviation" in quantities:
            if density is not None:
                n_power = solution_info.n
                deviation_values = np.zeros(n_points)
                for i, pt in enumerate(points):
                    try:
                        phi = field(pt[0], pt[1])
                        rho = density(pt[0], pt[1])
                        if rho > 0:
                            phi_adiabatic = pow(rho, -1.0 / (n_power + 1))
                            if phi_adiabatic > 0:
                                deviation_values[i] = (phi - phi_adiabatic) / phi_adiabatic
                            else:
                                deviation_values[i] = np.nan
                        else:
                            deviation_values[i] = np.nan
                    except RuntimeError:
                        deviation_values[i] = np.nan
                data["field_deviation"] = deviation_values.tolist()
            else:
                data["field_deviation_note"] = "Density not saved with this solution (re-solve to generate)"

        # Build response
        result = {
            "solution_id": solution_id,
            "mode": mode,
            "n_points": n_points,
            "n_valid": int(valid_mask.sum()),
            "data": data
        }

        # Add position data
        if mode == "radial":
            result["data"]["r"] = position_values.tolist()
        elif mode == "line":
            result["data"]["distance"] = position_values.tolist()
            result["data"]["coordinates"] = points.tolist()
        elif mode == "points":
            result["data"]["coordinates"] = points.tolist()
        elif mode == "grid":
            result["r_values"] = r_values.tolist()
            result["z_values"] = z_values.tolist()
            result["grid_shape"] = [len(z_values), len(r_values)]

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "EVALUATE_ERROR",
                "message": str(e)
            }
        }, indent=2))]


async def _handle_max_in_region(
    arguments: dict[str, Any],
    mesh,
    field,
    field_grad,
    mesh_info,
    solution_info,
    mesh_bounds: dict
) -> list[TextContent]:
    """Handle max_in_region mode."""
    import dolfin as d

    params = arguments.get("params", {})
    quantities = arguments.get("quantities", ["field", "gradient_magnitude"])

    region = params.get("region", "vacuum")
    min_distance_from = params.get("min_distance_from")
    min_distance = params.get("min_distance", 0)
    n_samples = params.get("n_samples", 1000)

    # Load the subdomain markers
    mesh_path = mesh_info.mesh_path
    subdomains_file = os.path.join(mesh_path, "mesh.xdmf")

    # Read subdomain markers
    mvc = d.MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with d.XDMFFile(subdomains_file) as f:
        f.read(mvc, "Subdomain")
    subdomains = d.MeshFunction("size_t", mesh, mvc)

    # Get region marker
    regions = mesh_info.regions
    if region not in regions:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "INVALID_REGION",
                "message": f"Region '{region}' not found. Available: {list(regions.keys())}"
            }
        }, indent=2))]

    target_marker = regions[region]

    # Determine which regions to keep distance from
    # min_distance_from can be:
    #   - None: no distance filtering
    #   - "all": all regions except the target region
    #   - a specific region name
    #   - a list of region names
    source_markers = []
    exclude_regions_desc = None

    if min_distance_from:
        if min_distance_from == "all":
            # All regions except target
            source_markers = [m for name, m in regions.items() if name != region]
            exclude_regions_desc = f"all other regions ({', '.join(n for n in regions if n != region)})"
        elif isinstance(min_distance_from, list):
            # List of specific regions
            for r in min_distance_from:
                if r not in regions:
                    return [TextContent(type="text", text=json.dumps({
                        "error": {
                            "code": "INVALID_REGION",
                            "message": f"Region '{r}' not found. Available: {list(regions.keys())}"
                        }
                    }, indent=2))]
                source_markers.append(regions[r])
            exclude_regions_desc = ", ".join(min_distance_from)
        else:
            # Single region name
            if min_distance_from not in regions:
                return [TextContent(type="text", text=json.dumps({
                    "error": {
                        "code": "INVALID_REGION",
                        "message": f"Region '{min_distance_from}' not found. Available: {list(regions.keys())}"
                    }
                }, indent=2))]
            source_markers = [regions[min_distance_from]]
            exclude_regions_desc = min_distance_from

    # Collect cell centers in target region
    target_cells = []
    target_centers = []
    for cell in d.cells(mesh):
        if subdomains[cell] == target_marker:
            target_cells.append(cell.index())
            target_centers.append(cell.midpoint().array()[:2])

    target_centers = np.array(target_centers)

    if len(target_centers) == 0:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "EMPTY_REGION",
                "message": f"No cells found in region '{region}'"
            }
        }, indent=2))]

    # If distance filtering is requested, compute distances from exclusion regions
    if len(source_markers) > 0 and min_distance > 0:
        # Collect unique vertex indices from all source region cells
        source_vertex_indices = set()
        for cell in d.cells(mesh):
            if subdomains[cell] in source_markers:
                for vertex in d.vertices(cell):
                    source_vertex_indices.add(vertex.index())

        if len(source_vertex_indices) == 0:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "EMPTY_REGION",
                    "message": f"No cells found in exclusion regions: {exclude_regions_desc}"
                }
            }, indent=2))]

        # Get coordinates of unique vertices
        coords = mesh.coordinates()
        source_boundary_points = coords[list(source_vertex_indices), :2]

        # Filter target points by distance from source boundaries using KD-tree
        # This is O(N log M) instead of O(N × M) for the naive approach
        from scipy.spatial import cKDTree
        tree = cKDTree(source_boundary_points)
        distances, _ = tree.query(target_centers, k=1)
        valid_mask = distances >= min_distance

        target_centers = target_centers[valid_mask]

        if len(target_centers) == 0:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "NO_VALID_POINTS",
                    "message": f"No points in '{region}' are >= {min_distance} from {exclude_regions_desc}"
                }
            }, indent=2))]

    # Sample points (or use all if fewer than n_samples)
    if len(target_centers) > n_samples:
        indices = np.random.choice(len(target_centers), n_samples, replace=False)
        sample_points = target_centers[indices]
    else:
        sample_points = target_centers

    n_valid = len(sample_points)

    # Evaluate quantities at sample points
    data = {}

    # Field values
    field_values = np.array([field(pt[0], pt[1]) for pt in sample_points])

    if "field" in quantities:
        max_idx = np.argmax(field_values)
        min_idx = np.argmin(field_values)
        data["field"] = {
            "max": float(field_values[max_idx]),
            "max_location": sample_points[max_idx].tolist(),
            "min": float(field_values[min_idx]),
            "min_location": sample_points[min_idx].tolist(),
            "mean": float(np.mean(field_values))
        }

    # Gradient magnitude (compute on-the-fly from gradient vector)
    if "gradient_magnitude" in quantities and field_grad is not None:
        grad_values = np.array([np.linalg.norm(field_grad(pt[0], pt[1])) for pt in sample_points])

        max_idx = np.argmax(grad_values)
        min_idx = np.argmin(grad_values)
        data["gradient_magnitude"] = {
            "max": float(grad_values[max_idx]),
            "max_location": sample_points[max_idx].tolist(),
            "min": float(grad_values[min_idx]),
            "min_location": sample_points[min_idx].tolist(),
            "mean": float(np.mean(grad_values))
        }

    result = {
        "solution_id": arguments["solution_id"],
        "mode": "max_in_region",
        "region": region,
        "min_distance_from": min_distance_from,
        "min_distance": min_distance,
        "n_samples": n_samples,
        "n_valid_samples": n_valid,
        "data": data
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_integrate(
    arguments: dict[str, Any],
    mesh,
    field,
    mesh_info,
    solution_info,
) -> list[TextContent]:
    """Handle integrate mode - compute volume integrals over a region.

    Computes total force F = ∫ρ∇φ dV and mass M = ∫ρ dV over a named region.
    Handles symmetry correctly:
    - axial: includes 2πr Jacobian for revolution
    - translation: result is per unit length in z
    """
    import dolfin as d
    from math import pi, sqrt

    params = arguments.get("params", {})
    region = params.get("region", "vacuum")
    quantity = params.get("quantity", "force")

    # Validate quantity
    if quantity not in ["force", "mass"]:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "INVALID_QUANTITY",
                "message": f"Invalid quantity '{quantity}'. Must be 'force' or 'mass'."
            }
        }, indent=2))]

    # Get region marker
    regions = mesh_info.regions
    if region not in regions:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "INVALID_REGION",
                "message": f"Region '{region}' not found. Available: {list(regions.keys())}"
            }
        }, indent=2))]

    target_marker = regions[region]

    # Load subdomain markers
    mesh_path = mesh_info.mesh_path
    subdomains_file = os.path.join(mesh_path, "mesh.xdmf")

    mvc = d.MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with d.XDMFFile(subdomains_file) as f:
        f.read(mvc, "Subdomain")
    subdomains = d.MeshFunction("size_t", mesh, mvc)

    # Load density function
    mesh_dir = os.path.dirname(mesh_path)
    root_dir = os.path.dirname(mesh_dir)
    solution_path = os.path.join(root_dir, "Saved Solutions", arguments["solution_id"])
    density_file = os.path.join(solution_path, "density.h5")

    if not os.path.exists(density_file):
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "DENSITY_NOT_FOUND",
                "message": "Density not saved with this solution. Re-solve to generate density field."
            }
        }, indent=2))]

    # Load density (DG0 space)
    V_dg = d.FunctionSpace(mesh, "DG", 0)
    density = d.Function(V_dg)
    with d.HDF5File(mesh.mpi_comm(), density_file, "r") as f:
        f.read(density, "density")

    # Get symmetry from solution info
    symmetry = getattr(solution_info, 'symmetry', 'axial')

    # Create measure restricted to target region
    dx_region = d.Measure("dx", domain=mesh, subdomain_data=subdomains, subdomain_id=target_marker)

    # Spatial coordinates for Jacobian
    x = d.SpatialCoordinate(mesh)

    # Set up Jacobian based on symmetry
    if symmetry == "axial":
        # Axisymmetric: include 2πr factor
        r = x[0]
        jacobian = 2 * pi * r
        coord_names = ("r", "z")
    else:
        # Translation or 3D: no extra factor (result is per unit length for translation)
        jacobian = d.Constant(1.0)
        coord_names = ("x", "y")

    # Compute integrals
    result_data = {}

    # Volume integral (with Jacobian)
    volume = d.assemble(jacobian * d.Constant(1.0) * dx_region)
    result_data["volume"] = float(volume)

    # Mass integral: M = ∫ρ dV
    mass = d.assemble(jacobian * density * dx_region)
    result_data["mass"] = float(mass)

    if quantity == "force":
        # Force integral: F = ∫ρ∇φ dV
        grad_phi = d.grad(field)

        # Integrate each component
        F_0 = d.assemble(jacobian * density * grad_phi[0] * dx_region)
        F_1 = d.assemble(jacobian * density * grad_phi[1] * dx_region)

        F_magnitude = sqrt(float(F_0)**2 + float(F_1)**2)

        result_data[f"F_{coord_names[0]}"] = float(F_0)
        result_data[f"F_{coord_names[1]}"] = float(F_1)
        result_data["F_magnitude"] = F_magnitude

    # Count cells in region for sanity check
    n_cells = sum(1 for cell in d.cells(mesh) if subdomains[cell] == target_marker)
    result_data["n_cells"] = n_cells

    result = {
        "solution_id": arguments["solution_id"],
        "mode": "integrate",
        "quantity": quantity,
        "region": region,
        "symmetry": symmetry,
        "data": result_data,
        "scaling": {
            "description": "Values are in rescaled (dimensionless) units.",
            "mass_physical": "M_physical[kg] = mass_scale_kg × mass",
            "force_physical": "F_physical[N] = force_scale_N × force",
            "note": "Use calculate_physical_parameters tool to get mass_scale_kg and force_scale_N."
        }
    }

    # Add note about units for translation symmetry
    if symmetry == "translation":
        result["scaling"]["symmetry_note"] = "For translation symmetry, values are per unit length in z."

    return [TextContent(type="text", text=json.dumps(result, indent=2))]
