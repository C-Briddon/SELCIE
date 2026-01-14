"""
create_mesh tool.

Generate finite element meshes for chameleon field simulations using geometry templates.
"""

import json
import tempfile
import os
from typing import Any

from mcp.types import Tool, TextContent

from SELCIE import MeshingTools

from utils.session import get_session, MeshInfo
from utils.density import extract_density_value, SPHERICAL_GEOMETRIES


# Mesh quality settings: (CellSizeMin, CellSizeMax, DistMax) relative to object size
# Consistent 2.5x ramp between levels, 4x max/min ratio
MESH_QUALITY_SETTINGS = {
    "very_coarse": {"cell_min_factor": 0.125, "cell_max_factor": 0.5, "dist_max_factor": 2.0},
    "coarse": {"cell_min_factor": 0.05, "cell_max_factor": 0.2, "dist_max_factor": 1.25},
    "medium": {"cell_min_factor": 0.02, "cell_max_factor": 0.08, "dist_max_factor": 0.5},
    "fine": {"cell_min_factor": 0.008, "cell_max_factor": 0.032, "dist_max_factor": 0.2},
    "very_fine": {"cell_min_factor": 0.0032, "cell_max_factor": 0.0128, "dist_max_factor": 0.1},
}

# Default symmetry for each geometry
# "axial" = 2D mesh revolved around z-axis (axisymmetric)
# "translation" = 2D mesh extended in z direction (translation-invariant)
# "none" = 3D mesh, no symmetry transformation
DEFAULT_SYMMETRY = {
    "sphere_in_vacuum": "axial",
    "ellipse_in_vacuum": "axial",
    "ellipsoid_in_vacuum": "none",  # True 3D mesh
    "cylinder_in_vacuum": "axial",
    "shell_in_vacuum": "axial",
    "two_spheres": "axial",
    "sphere_near_wall": "axial",
    "box_2d": "translation",
    "box_3d": "none",  # True 3D mesh
    "disk": "axial",
    "sphere_domain": "axial",
    "sphere_in_profile": "axial",
    "parallel_plates": "translation",
    "custom_2d": "axial",
    "custom_3d": "none",  # True 3D mesh
}

# Geometries where symmetry can be overridden by the user
# All other geometries have fixed symmetry required for correct 3D shape
CONFIGURABLE_SYMMETRY = {
    "custom_2d",  # Can be axial (revolved) or translation (extruded)
}

# Refinement limits to prevent excessive cell counts
REFINEMENT_LIMITS = {
    "min_cell_size_factor": 0.002,  # Minimum cell size as fraction of object size
    "max_refinement_ratio": 50,      # Max ratio of largest to smallest cells
    "target_boundary_cells": 5,      # Target cells across thin shell
}

# Default maximum cell count to prevent accidentally creating huge meshes
DEFAULT_MAX_CELLS = 200000


def estimate_physics_refinement(
    object_size: float,
    physics_params: dict,
    base_quality: dict,
) -> dict:
    """Estimate mesh refinement settings based on physics parameters.

    Args:
        object_size: Characteristic size of the subdomain (radius, etc.)
        physics_params: Dict with 'lambda_subdomain'
        base_quality: Base quality settings to modify

    Returns:
        Modified quality settings dict
    """
    lambda_subdomain = physics_params.get("lambda_subdomain")

    if lambda_subdomain is None:
        return base_quality

    # Estimate thin shell thickness
    # In thin shell regime: δ ≈ λ_subdomain
    # Cap between reasonable bounds
    shell_thickness = lambda_subdomain

    # Apply limits
    min_shell = object_size * REFINEMENT_LIMITS["min_cell_size_factor"] * 3
    max_shell = object_size * 0.5  # Can't be larger than half the object

    shell_thickness = max(min_shell, min(max_shell, shell_thickness))

    # Target cell size to resolve shell with ~5 cells
    target_cell_min = shell_thickness / REFINEMENT_LIMITS["target_boundary_cells"]

    # Apply minimum cell size limit
    min_allowed = object_size * REFINEMENT_LIMITS["min_cell_size_factor"]
    target_cell_min = max(min_allowed, target_cell_min)

    # Ensure refinement ratio isn't too extreme
    max_cell = base_quality["cell_max_factor"] * object_size
    if max_cell / target_cell_min > REFINEMENT_LIMITS["max_refinement_ratio"]:
        target_cell_min = max_cell / REFINEMENT_LIMITS["max_refinement_ratio"]

    # Calculate refinement zone - extend 2-3 shell thicknesses from boundary
    dist_max = min(3 * shell_thickness, base_quality["dist_max_factor"] * object_size)

    return {
        "cell_min_factor": target_cell_min / object_size,
        "cell_max_factor": base_quality["cell_max_factor"],
        "dist_max_factor": dist_max / object_size,
        "shell_thickness": shell_thickness,
        "physics_refined": True,
    }


def _compute_lambda(alpha: float, rho: float, n: int = 1) -> float:
    """Compute Compton wavelength from physics parameters.

    λ = √(α / n(n+1)) × ρ^(-(n+2)/(2(n+1)))

    Args:
        alpha: Dimensionless coupling constant
        rho: Dimensionless density
        n: Potential power index

    Returns:
        Compton wavelength λ
    """
    import math
    exponent = -(n + 2) / (2 * (n + 1))
    return math.sqrt(alpha / (n * (n + 1))) * (rho ** exponent)


TOOL_DEFINITION = Tool(
    name="create_mesh",
    description=(
        "Generate a finite element mesh for chameleon field simulations. "
        "Uses geometry templates that automatically handle subdomain creation, "
        "symmetry, and mesh refinement. Templates include object-in-vacuum "
        "(sphere_in_vacuum, ellipse_in_vacuum, etc.), plain domains (box_2d, disk, etc.), "
        "and custom shapes from file.\n\n"
        "IMPORTANT: For thin-shell problems (high α, high density contrast), provide "
        "physics_params with alpha and density to enable automatic mesh refinement "
        "near object boundaries. This ensures the thin shell region is properly resolved."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "geometry": {
                "type": "string",
                "enum": [
                    "sphere_in_vacuum", "ellipse_in_vacuum",
                    "cylinder_in_vacuum", "shell_in_vacuum", "two_spheres", "sphere_near_wall",
                    "sphere_in_profile",
                    "box_2d", "box_3d", "disk", "sphere_domain", "parallel_plates",
                    "custom_2d", "custom_3d"
                ],
                "description": (
                    "Geometry template. Choose based on physical setup:\n\n"
                    "OBJECT-IN-VACUUM (screening/force calculations):\n"
                    "- sphere_in_vacuum: Spherical source in vacuum. Regions: object, vacuum. "
                    "Fixed symmetry: axial (2D). 'r' = spherical radius.\n"
                    "- ellipse_in_vacuum: Oblate/prolate ellipsoid in vacuum. Regions: object, vacuum. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- cylinder_in_vacuum: Cylindrical source in vacuum. Regions: cylinder, vacuum. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- shell_in_vacuum: Hollow spherical shell in vacuum. Regions: shell, vacuum. "
                    "Fixed symmetry: axial (2D). 'r' = spherical radius.\n"
                    "- two_spheres: Two spheres for force calculations. Regions: sphere_1, sphere_2, vacuum. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- sphere_near_wall: Sphere near planar wall. Regions: sphere, wall, vacuum. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- sphere_in_profile: Sphere in spatially-varying density profile (e.g., NFW, isothermal). "
                    "Regions: sphere, background. Fixed symmetry: axial (2D). 'r' = spherical radius. "
                    "Use center_z to offset sphere along z-axis.\n\n"
                    "PLAIN DOMAINS (no interior object):\n"
                    "- sphere_domain: For spherically-symmetric profiles (NFW, isothermal). Regions: domain. "
                    "Fixed symmetry: axial (2D). 'r' = spherical radius.\n"
                    "- disk: For cylindrically-symmetric profiles. Regions: domain. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- box_2d: 2D Cartesian rectangle. Regions: domain. Default symmetry: translation (2D).\n"
                    "- box_3d: 3D Cartesian box. Regions: domain. Fixed symmetry: none (true 3D).\n"
                    "- parallel_plates: Two parallel plates with vacuum gap. Regions: vacuum, plate. "
                    "Fixed symmetry: translation (2D extended in y). 'x' = perpendicular to plates.\n\n"
                    "CUSTOM SHAPES:\n"
                    "- custom_2d: Arbitrary 2D shape from points. Points are [r, z] for axial symmetry, "
                    "[x, y] for translation. Regions: object, vacuum. Default symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- custom_3d: Arbitrary 3D shape from contours. Regions: object. Fixed symmetry: none (true 3D)."
                ),
            },
            "params": {
                "type": "object",
                "description": "Geometry-specific parameters.",
                "properties": {
                    "object_radius": {"type": "number", "description": "Radius of spherical source"},
                    "vacuum_radius": {"type": "number", "description": "Outer radius of vacuum region"},
                    "domain_radius": {"type": "number", "description": "Outer radius of domain (sphere_in_profile)"},
                    "center_z": {"type": "number", "description": "Z-position of sphere center (sphere_in_profile, default 0)"},
                    "wall_thickness": {"type": "number", "description": "Wall thickness (sphere_in_vacuum, sphere_near_wall)"},
                    "rx": {"type": "number", "description": "Semi-axis in r/x direction"},
                    "ry": {"type": "number", "description": "Semi-axis in z/y direction"},
                    "rz": {"type": "number", "description": "Semi-axis in z direction (3D)"},
                    "radius": {"type": "number", "description": "Radius for cylinder, disk, sphere_domain"},
                    "height": {"type": "number", "description": "Height for cylinder, box"},
                    "width": {"type": "number", "description": "Width for box"},
                    "depth": {"type": "number", "description": "Depth for box_3d"},
                    "inner_radius": {"type": "number", "description": "Inner radius for shell"},
                    "outer_radius": {"type": "number", "description": "Outer radius for shell"},
                    "radius_1": {"type": "number", "description": "First sphere radius (two_spheres)"},
                    "radius_2": {"type": "number", "description": "Second sphere radius (two_spheres)"},
                    "separation": {"type": "number", "description": "Center-to-center separation (two_spheres)"},
                    "wall_distance": {"type": "number", "description": "Distance from sphere center to wall (sphere_near_wall)"},
                    "points": {"type": "array", "description": "Array of [r,z] points for custom_2d"},
                    "shape_file": {"type": "string", "description": "Path to file with shape points (custom_2d)"},
                    "contour_file": {"type": "string", "description": "Path to 3D contour file (custom_3d)"},
                    "plate_separation": {"type": "number", "description": "Gap between inner surfaces of plates (parallel_plates)"},
                    "plate_thickness": {"type": "number", "description": "Thickness of each plate (parallel_plates)"},
                    "domain_height": {"type": "number", "description": "Height of domain in y direction (parallel_plates, default=plate_separation)"},
                },
            },
            "mesh_quality": {
                "type": "string",
                "enum": ["very_coarse", "coarse", "medium", "fine", "very_fine"],
                "default": "medium",
                "description": "Mesh resolution.",
            },
            "symmetry": {
                "type": "string",
                "enum": ["axial", "translation", "none"],
                "description": (
                    "Override symmetry for geometries with 'Default symmetry'. "
                    "Geometries with 'Fixed symmetry' ignore this parameter. "
                    "SELCIE always solves 3D problems; 2D meshes are slices with implied symmetry:\n"
                    "- axial: 2D mesh in (r,z) revolved around z-axis (axisymmetric).\n"
                    "- translation: 2D mesh in (x,y) extended infinitely in z (translation-invariant).\n"
                    "- none: Use geometry's default symmetry."
                ),
            },
            "custom_id": {
                "type": "string",
                "description": "Custom mesh ID.",
            },
            "allow_large_mesh": {
                "type": "boolean",
                "description": f"Allow meshes exceeding {DEFAULT_MAX_CELLS:,} cells. Default: false.",
                "default": False,
            },
            "physics_params": {
                "type": "object",
                "description": (
                    "Physics parameters for automatic thin-shell mesh refinement. "
                    "Option 1: Provide 'lambda' dict mapping region names to Compton wavelengths. "
                    "Option 2: Provide 'alpha', 'density' dict, and 'n' - lambdas will be computed per region. "
                    "The mesh will be refined near boundaries of dense regions to resolve thin shells."
                ),
                "properties": {
                    "lambda": {
                        "type": "object",
                        "description": (
                            "Direct specification of Compton wavelength per region. "
                            "Example: {\"object\": 0.001, \"wall\": 0.002}. "
                            "If provided, alpha/density are ignored."
                        ),
                        "additionalProperties": {"type": "number"},
                    },
                    "alpha": {
                        "type": "number",
                        "description": (
                            "Dimensionless coupling constant α. Used with 'density' to compute "
                            "λ = √(α/n(n+1)) × ρ^(-(n+2)/(2(n+1))) for each region."
                        ),
                    },
                    "density": {
                        "type": "object",
                        "description": "Dimensionless density ρ̂ = ρ/ρ₀ per region. Value is: number, {expression: str}, or {file: str, format?: 'tabulated'|'grid', columns?: int[], bounds?: number[], skip_header?: int, npz_key?: str}. For tabulated: columns selects columns (1-based). For grid: bounds maps grid to spatial coordinates. For non-numeric values, max density is used to compute λ.",
                        "additionalProperties": True,
                    },
                    "n": {
                        "type": "integer",
                        "description": "Potential power index (default: 1)",
                        "default": 1,
                    },
                },
            },
        },
        "required": ["geometry", "params"],
    },
)


def _get_mesh_dir() -> str:
    """Get directory for storing mesh files."""
    mesh_dir = os.path.join(tempfile.gettempdir(), "selcie_meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    # Also create Saved Meshes subdirectory (required by SELCIE)
    saved_meshes_dir = os.path.join(mesh_dir, "Saved Meshes")
    os.makedirs(saved_meshes_dir, exist_ok=True)
    return mesh_dir


def _sphere_points(radius: float, n_points: int = 50) -> list:
    """Generate points for a sphere boundary (half-plane for axisymmetric)."""
    import numpy as np
    theta = np.linspace(0, np.pi, n_points, endpoint=True)
    r = radius * np.sin(theta)  # radial distance from axis
    z = radius * np.cos(theta)  # height along axis
    return [[float(ri), float(zi), 0.0] for ri, zi in zip(r, z)]


def _ellipse_points(rx: float, ry: float, n_points: int = 50) -> list:
    """Generate points for an ellipse boundary (half-plane for axisymmetric)."""
    import numpy as np
    theta = np.linspace(0, np.pi, n_points, endpoint=True)
    r = rx * np.sin(theta)  # radial distance from axis
    z = ry * np.cos(theta)  # height along axis
    return [[float(ri), float(zi), 0.0] for ri, zi in zip(r, z)]


def _create_sphere_in_vacuum(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create sphere in vacuum geometry."""
    object_radius = params["object_radius"]
    vacuum_radius = params["vacuum_radius"]
    wall_thickness = params.get("wall_thickness")

    # Create the sphere using explicit points for smooth boundary
    # (using create_ellipse results in jagged boundaries due to GMSH arc discretization)
    n_boundary_points = 50  # Matches Sphere_Standalone.py
    points = _sphere_points(object_radius, n_boundary_points)
    points = MT.constrain_distance(points)
    MT.points_to_surface(points)

    # Mark as subdomain with refinement
    # cell_min based on object size to resolve boundary
    # cell_max based on vacuum_radius so cells can grow large far from object
    cell_min = quality["cell_min_factor"] * object_radius
    cell_max = quality["cell_max_factor"] * vacuum_radius
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum)
    bg_cell_min = quality["cell_min_factor"] * vacuum_radius
    bg_cell_max = quality["cell_max_factor"] * vacuum_radius
    bg_dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=bg_dist_max,
        background_radius=vacuum_radius,
        wall_thickness=wall_thickness,  # None if not specified
        #refine_outer_wall_boundary=True,  # Use scaled cell sizes for wall
        symmetry="vertical",  # Axisymmetric - only mesh r >= 0
    )

    regions = ["object", "vacuum"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": vacuum_radius,
        "z_min": -vacuum_radius,
        "z_max": vacuum_radius,
    }

    return regions, bounds


def _create_ellipse_in_vacuum(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create ellipse in vacuum geometry."""
    rx = params["rx"]
    ry = params["ry"]
    vacuum_radius = params["vacuum_radius"]

    # Create the ellipse using explicit points for smooth boundary
    n_boundary_points = 50
    points = _ellipse_points(rx, ry, n_boundary_points)
    points = MT.constrain_distance(points)
    MT.points_to_surface(points)

    # Mark as subdomain with refinement
    # cell_min based on ellipse size to resolve boundary
    # cell_max based on vacuum_radius so cells can grow large far from object
    char_size = max(rx, ry)
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * vacuum_radius
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum) - no wall by default
    bg_cell_min = quality["cell_min_factor"] * vacuum_radius
    bg_cell_max = quality["cell_max_factor"] * vacuum_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=dist_max,
        background_radius=vacuum_radius,
        wall_thickness=None,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["object", "vacuum"]
    bounds = {
        "r_min": 0.0,
        "r_max": vacuum_radius,
        "z_min": -vacuum_radius,
        "z_max": vacuum_radius,
    }

    return regions, bounds


def _create_disk(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create simple disk domain (no interior object)."""
    radius = params["radius"]

    # For plain domains, use radius directly (no object boundary to resolve)
    cell_min = quality["cell_min_factor"] * radius
    cell_max = quality["cell_max_factor"] * radius
    dist_max = quality["dist_max_factor"] * radius
    MT.create_background_mesh(
        CellSizeMin=cell_min,
        CellSizeMax=cell_max,
        DistMax=dist_max,
        background_radius=radius,
        wall_thickness=None,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["domain"]
    bounds = {
        "r_min": 0.0,
        "r_max": radius,
        "z_min": -radius,
        "z_max": radius,
    }

    return regions, bounds


def _create_box_2d(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create 2D rectangular domain."""
    width = params["width"]
    height = params["height"]

    # Create rectangle
    MT.create_rectangle(dx=width, dy=height)

    # For plain domains, use min dimension directly (no object boundary to resolve)
    char_size = min(width, height)
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
    dist_max = quality["dist_max_factor"] * max(width, height)
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    regions = ["domain"]
    bounds = {
        "x_min": -width / 2,
        "x_max": width / 2,
        "y_min": -height / 2,
        "y_max": height / 2,
    }

    return regions, bounds


def _create_parallel_plates(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create parallel plates with vacuum gap.

    Creates two parallel plates separated by a vacuum region.
    Uses translation symmetry (symmetry="none") - the 2D mesh
    extends via translation in the y direction.

    Layout (in x-y plane):
        [plate_L] [vacuum] [plate_R]
        x: 0 to plate_thickness | plate_thickness to plate_thickness+separation | ...
    """
    plate_separation = params["plate_separation"]
    plate_thickness = params["plate_thickness"]
    domain_height = params.get("domain_height", plate_separation)

    total_width = 2 * plate_thickness + plate_separation

    # Create vacuum region first (inner)
    # Centered in x, full height in y
    vac_x_min = plate_thickness
    vac_x_max = plate_thickness + plate_separation
    vacuum = MT.points_to_surface([
        (vac_x_min, 0, 0),
        (vac_x_max, 0, 0),
        (vac_x_max, domain_height, 0),
        (vac_x_min, domain_height, 0),
    ])

    # Subdomain for vacuum - finer near plate boundaries
    cell_min_vac = quality["cell_min_factor"] * plate_separation
    cell_max_vac = quality["cell_max_factor"] * plate_separation
    dist_max = quality["dist_max_factor"] * plate_separation
    MT.create_subdomain(CellSizeMin=cell_min_vac, CellSizeMax=cell_max_vac, DistMax=dist_max)

    # Create full domain (outer) embedding vacuum
    MT.points_to_surface([
        (0, 0, 0),
        (total_width, 0, 0),
        (total_width, domain_height, 0),
        (0, domain_height, 0),
    ], embed=vacuum)

    # Subdomain for plates
    # cell_min based on plate_thickness to resolve boundary
    # cell_max based on plate_separation so cells can grow large
    cell_min_plate = quality["cell_min_factor"] * plate_thickness
    cell_max_plate = quality["cell_max_factor"] * plate_separation
    MT.create_subdomain(CellSizeMin=cell_min_plate, CellSizeMax=cell_max_plate, DistMax=dist_max)

    # Regions in creation order: vacuum first, then plate
    regions = ["vacuum", "plate"]
    bounds = {
        "x_min": 0,
        "x_max": total_width,
        "y_min": 0,
        "y_max": domain_height,
    }

    return regions, bounds


def _create_shell_in_vacuum(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create hollow shell in vacuum geometry."""
    import numpy as np

    inner_radius = params["inner_radius"]
    outer_radius = params["outer_radius"]
    vacuum_radius = params["vacuum_radius"]

    # Create shell as a closed polygon (outer boundary + inner boundary reversed)
    # This defines a shell annulus in axisymmetric coords
    n_points = 30

    # Outer semicircle: from top (0, outer_r) to bottom (0, -outer_r)
    theta_outer = np.linspace(0, np.pi, n_points, endpoint=True)
    outer_r = outer_radius * np.sin(theta_outer)
    outer_z = outer_radius * np.cos(theta_outer)

    # Inner semicircle: from bottom (0, -inner_r) to top (0, inner_r) - reversed
    theta_inner = np.linspace(np.pi, 0, n_points, endpoint=True)
    inner_r = inner_radius * np.sin(theta_inner)
    inner_z = inner_radius * np.cos(theta_inner)

    # Combine into closed loop (outer down, inner up)
    shell_points = []
    for ri, zi in zip(outer_r, outer_z):
        shell_points.append([float(ri), float(zi), 0.0])
    for ri, zi in zip(inner_r, inner_z):
        shell_points.append([float(ri), float(zi), 0.0])

    shell_points = MT.constrain_distance(shell_points)
    MT.points_to_surface(shell_points)

    # Mark as subdomain with refinement
    # cell_min based on shell thickness to resolve thin shell
    # cell_max based on vacuum_radius so cells can grow large far from shell
    char_size = outer_radius - inner_radius
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * vacuum_radius
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum) - no wall by default
    bg_cell_min = quality["cell_min_factor"] * vacuum_radius
    bg_cell_max = quality["cell_max_factor"] * vacuum_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=dist_max,
        background_radius=vacuum_radius,
        wall_thickness=None,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["shell", "vacuum"]
    bounds = {
        "r_min": 0.0,
        "r_max": vacuum_radius,
        "z_min": -vacuum_radius,
        "z_max": vacuum_radius,
    }

    return regions, bounds


def _create_cylinder_in_vacuum(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create cylinder in vacuum geometry (2D axisymmetric = rectangle)."""
    radius = params["radius"]
    height = params["height"]
    vacuum_radius = params["vacuum_radius"]

    # In 2D axisymmetric, a cylinder is a rectangle in (r, z)
    rect = MT.create_rectangle(dx=radius, dy=height)
    MT.translate_x(rect, dx=radius / 2)  # Move to positive r

    # Mark as subdomain with refinement
    # cell_min based on cylinder size to resolve boundary
    # cell_max based on vacuum_radius so cells can grow large far from object
    char_size = min(radius, height)
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * vacuum_radius
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum) - no wall by default
    bg_cell_min = quality["cell_min_factor"] * vacuum_radius
    bg_cell_max = quality["cell_max_factor"] * vacuum_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=dist_max,
        background_radius=vacuum_radius,
        wall_thickness=None,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["cylinder", "vacuum"]
    bounds = {
        "r_min": 0.0,
        "r_max": vacuum_radius,
        "z_min": -vacuum_radius,
        "z_max": vacuum_radius,
    }

    return regions, bounds


def _create_sphere_domain(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create simple spherical domain (no interior object)."""
    radius = params["radius"]

    # For plain domains, use radius directly (no object boundary to resolve)
    cell_min = quality["cell_min_factor"] * radius
    cell_max = quality["cell_max_factor"] * radius
    dist_max = quality["dist_max_factor"] * radius
    MT.create_background_mesh(
        CellSizeMin=cell_min,
        CellSizeMax=cell_max,
        DistMax=dist_max,
        background_radius=radius,
        wall_thickness=None,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["domain"]
    bounds = {
        "r_min": 0.0,
        "r_max": radius,
        "z_min": -radius,
        "z_max": radius,
    }

    return regions, bounds


def _create_sphere_in_profile(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create sphere in density profile geometry.

    The sphere can be displaced along z-axis using center_z parameter
    while maintaining axial symmetry. The background density profile
    is measured from origin (r=0), so a sphere at center_z=5 sits at
    radius 5 from the profile center.
    """
    object_radius = params["object_radius"]
    domain_radius = params["domain_radius"]
    center_z = params.get("center_z", 0.0)  # Default centered

    # Create sphere using explicit points for smooth boundary in r >= 0
    # (using create_ellipse results in sphere extending to r < 0)
    n_boundary_points = 50
    points = _sphere_points(object_radius, n_boundary_points)

    # Offset z-coordinates if center_z != 0
    if center_z != 0.0:
        points = [[p[0], p[1] + center_z, p[2]] for p in points]

    points = MT.constrain_distance(points)
    MT.points_to_surface(points)

    # Mark sphere as subdomain
    # cell_min based on object size to resolve boundary
    # cell_max based on domain_radius so cells can grow large far from object
    cell_min = quality["cell_min_factor"] * object_radius
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background
    bg_cell_min = quality["cell_min_factor"] * domain_radius
    bg_cell_max = quality["cell_max_factor"] * domain_radius
    bg_dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=bg_dist_max,
        background_radius=domain_radius,
        wall_thickness=None,
        symmetry="vertical",
    )

    regions = ["sphere", "background"]
    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
    }
    return regions, bounds


def _create_two_spheres(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create two spheres (source + test mass) in vacuum."""
    radius_1 = params["radius_1"]
    radius_2 = params["radius_2"]
    separation = params["separation"]
    vacuum_radius = params["vacuum_radius"]

    # Create first sphere (source) at origin using explicit points
    # cell_min based on sphere size to resolve boundary
    # cell_max based on vacuum_radius so cells can grow large far from object
    n_boundary_points = 50
    points_1 = _sphere_points(radius_1, n_boundary_points)
    points_1 = MT.constrain_distance(points_1)
    MT.points_to_surface(points_1)
    cell_min = quality["cell_min_factor"] * radius_1
    cell_max = quality["cell_max_factor"] * vacuum_radius
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create second sphere (test mass) offset along z using explicit points
    points_2 = _sphere_points(radius_2, n_boundary_points)
    # Offset z coordinate by separation
    points_2 = [[p[0], p[1] + separation, p[2]] for p in points_2]
    points_2 = MT.constrain_distance(points_2)
    MT.points_to_surface(points_2)
    cell_min2 = quality["cell_min_factor"] * radius_2
    cell_max2 = quality["cell_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min2, CellSizeMax=cell_max2, DistMax=dist_max)

    # Create background (vacuum) - no wall by default
    bg_cell_min = quality["cell_min_factor"] * vacuum_radius
    bg_cell_max = quality["cell_max_factor"] * vacuum_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=dist_max,
        background_radius=vacuum_radius,
        wall_thickness=None,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["sphere_1", "sphere_2", "vacuum"]
    bounds = {
        "r_min": 0.0,
        "r_max": vacuum_radius,
        "z_min": -vacuum_radius,
        "z_max": vacuum_radius,
    }

    return regions, bounds


def _create_sphere_near_wall(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create sphere near a physical wall.

    Creates geometry with:
    - Sphere (test mass) suspended above the wall
    - Physical wall as a dense subdomain at the bottom
    - Vacuum region between and around
    """
    object_radius = params["object_radius"]
    wall_distance = params["wall_distance"]  # Distance from sphere center to wall surface
    wall_thickness = params.get("wall_thickness", 0.1)  # Default 0.1 if not specified
    vacuum_radius = params["vacuum_radius"]

    # Wall surface is at z=0, wall extends from z=-wall_thickness to z=0
    # Sphere center is at z=wall_distance

    # Create the wall as a half-disk at the bottom
    # Wall region: from z=-wall_thickness to z=0
    wall_disk = MT.create_ellipse(rx=vacuum_radius, ry=vacuum_radius)
    wall_rect = MT.create_rectangle(dx=2 * vacuum_radius, dy=wall_thickness)
    MT.translate_y(wall_rect, dy=-wall_thickness / 2)  # Center at y = -wall_thickness/2
    wall = MT.intersect_shapes(wall_disk, wall_rect)

    # Mark wall as subdomain
    # cell_min based on wall thickness to resolve boundary
    # cell_max based on vacuum_radius so cells can grow large far from wall
    wall_cell_min = quality["cell_min_factor"] * wall_thickness
    wall_cell_max = quality["cell_max_factor"] * vacuum_radius
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=wall_cell_min, CellSizeMax=wall_cell_max, DistMax=dist_max)

    # Create sphere offset above wall surface (z=0)
    sphere = MT.create_ellipse(rx=object_radius, ry=object_radius)
    MT.translate_y(sphere, dy=wall_distance)  # Sphere center at z=wall_distance

    # Mark sphere as subdomain with refinement
    # cell_min based on sphere size to resolve boundary
    # cell_max based on vacuum_radius so cells can grow large far from object
    cell_min = quality["cell_min_factor"] * object_radius
    cell_max = quality["cell_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create vacuum region (upper half-disk, z >= 0)
    vacuum_disk = MT.create_ellipse(rx=vacuum_radius, ry=vacuum_radius)
    upper_rect = MT.create_rectangle(dx=2 * vacuum_radius, dy=vacuum_radius)
    MT.translate_y(upper_rect, dy=vacuum_radius / 2)  # Center at y = vacuum_radius/2
    vacuum = MT.intersect_shapes(vacuum_disk, upper_rect)

    bg_cell_min = quality["cell_min_factor"] * vacuum_radius
    bg_cell_max = quality["cell_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=bg_cell_min, CellSizeMax=bg_cell_max, DistMax=dist_max)

    regions = ["wall", "sphere", "vacuum"]
    bounds = {
        "r_min": 0.0,
        "r_max": vacuum_radius,
        "z_min": -wall_thickness,  # Bottom of wall
        "z_max": vacuum_radius,
    }

    return regions, bounds


def _create_box_3d(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create 3D rectangular box domain."""
    width = params["width"]    # x extent
    height = params["height"]  # y extent
    depth = params["depth"]    # z extent

    # In 3D, create a box
    MT.create_box(dx=width, dy=height, dz=depth)

    # For plain domains, use min dimension directly (no object boundary to resolve)
    char_size = min(width, height, depth)
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
    dist_max = quality["dist_max_factor"] * max(width, height, depth)
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    regions = ["domain"]
    bounds = {
        "x_min": -width / 2,
        "x_max": width / 2,
        "y_min": -height / 2,
        "y_max": height / 2,
        "z_min": -depth / 2,
        "z_max": depth / 2,
    }

    return regions, bounds


def _create_custom_2d(
    MT: MeshingTools,
    params: dict,
    quality: dict,
    symmetry: str = "axial",
) -> tuple[list[str], dict]:
    """Create custom 2D geometry from file or points.

    Supports:
    - shape_file: Path to text file with x,y coordinates per line
    - points: Direct list of [x, y] coordinates (or [r, z] for axisymmetric)

    For axisymmetric (symmetry="axial"), points with x < 0 are discarded.
    """
    import numpy as np

    vacuum_radius = params.get("vacuum_radius", 1.0)
    is_axisymmetric = symmetry == "axial"

    # Load points from file or use direct points
    if "shape_file" in params:
        points_2d = np.loadtxt(params["shape_file"])
    elif "points" in params:
        points_2d = np.array(params["points"])
    else:
        raise ValueError("Must provide either 'shape_file' or 'points'")

    # For axisymmetric, filter out r < 0 points (r = 0 on axis is valid)
    if is_axisymmetric:
        mask = points_2d[:, 0] >= 0  # Keep r >= 0, exclude r < 0
        n_removed = np.sum(~mask)
        if n_removed > 0:
            print(f"Warning: Removed {n_removed} points with r < 0 for axisymmetric mesh")
        points_2d = points_2d[mask]

    # Convert to 3D coordinates (z=0) for SELCIE
    if points_2d.shape[1] == 2:
        points_3d = [[p[0], p[1], 0.0] for p in points_2d]
    else:
        points_3d = points_2d.tolist()

    # Create shape from points
    MT.points_to_surface(points_3d)

    # Mark as subdomain with refinement
    # cell_min based on object size to resolve boundary
    # cell_max based on vacuum_radius so cells can grow large far from object
    char_size = float(np.max(points_2d) - np.min(points_2d))
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * vacuum_radius
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum) if vacuum_radius provided
    if vacuum_radius:
        bg_cell_min = quality["cell_min_factor"] * vacuum_radius
        bg_cell_max = quality["cell_max_factor"] * vacuum_radius
        bg_kwargs = {
            "CellSizeMin": bg_cell_min,
            "CellSizeMax": bg_cell_max,
            "DistMax": dist_max,
            "background_radius": vacuum_radius,
            "wall_thickness": 0.1 * vacuum_radius,
            "refine_outer_wall_boundary": True,  # Use scaled cell sizes for wall
        }
        if is_axisymmetric:
            bg_kwargs["symmetry"] = "vertical"
        MT.create_background_mesh(**bg_kwargs)

    # Calculate actual bounds from points
    x_coords = points_2d[:, 0]
    y_coords = points_2d[:, 1]

    regions = ["object", "vacuum"]
    bounds = {
        "r_min": 0.0,
        "r_max": vacuum_radius,
        "z_min": -vacuum_radius,
        "z_max": vacuum_radius,
        "object_bounds": {
            "x_min": float(np.min(x_coords)),
            "x_max": float(np.max(x_coords)),
            "y_min": float(np.min(y_coords)),
            "y_max": float(np.max(y_coords)),
        }
    }

    return regions, bounds


def _create_custom_3d(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create custom 3D geometry from contours.

    Supports:
    - contours: List of contour point lists, each contour is a list of [x,y,z] points
    - contour_file: Path to file with contours (separate contours by blank lines)
    """
    import numpy as np

    # Load contours
    if "contours" in params:
        contours = params["contours"]
    elif "contour_file" in params:
        # Load contours from file (blank line separates contours)
        with open(params["contour_file"], 'r') as f:
            content = f.read()
        contour_strs = content.strip().split('\n\n')
        contours = []
        for cs in contour_strs:
            lines = cs.strip().split('\n')
            contour = [[float(x) for x in line.split()] for line in lines if line.strip()]
            if contour:
                contours.append(contour)
    else:
        raise ValueError("Must provide either 'contours' or 'contour_file'")

    # Create volume from contours
    MT.points_to_volume(contours)

    # Calculate characteristic size from contours
    all_points = np.array([p for contour in contours for p in contour])
    char_size = float(np.max(all_points) - np.min(all_points))

    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
    dist_max = quality["dist_max_factor"] * char_size
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    regions = ["object"]
    bounds = {
        "x_min": float(np.min(all_points[:, 0])),
        "x_max": float(np.max(all_points[:, 0])),
        "y_min": float(np.min(all_points[:, 1])),
        "y_max": float(np.max(all_points[:, 1])),
        "z_min": float(np.min(all_points[:, 2])),
        "z_max": float(np.max(all_points[:, 2])),
    }

    return regions, bounds


async def handle(args: dict[str, Any]) -> list[TextContent]:
    """Handle create_mesh tool call."""

    geometry = args["geometry"]
    params = args["params"]
    mesh_quality = args.get("mesh_quality", "medium")
    # Determine symmetry
    # Most geometries have fixed symmetry; only CONFIGURABLE_SYMMETRY can be overridden
    requested_symmetry = args.get("symmetry")
    default_symmetry = DEFAULT_SYMMETRY.get(geometry, "axial")
    warnings = []

    if geometry in CONFIGURABLE_SYMMETRY and requested_symmetry and requested_symmetry != "none":
        # Configurable geometry with user override
        symmetry = requested_symmetry
    elif geometry not in CONFIGURABLE_SYMMETRY and requested_symmetry and requested_symmetry != "none" and requested_symmetry != default_symmetry:
        # Fixed symmetry - ignore user override and warn
        symmetry = default_symmetry
        warnings.append(
            f"Symmetry '{requested_symmetry}' ignored for '{geometry}' "
            f"(requires '{default_symmetry}')"
        )
    else:
        # Use default
        symmetry = default_symmetry
    custom_id = args.get("custom_id")

    # Validate geometry is implemented
    implemented = [
        "sphere_in_vacuum", "ellipse_in_vacuum", "disk", "box_2d",
        "shell_in_vacuum", "cylinder_in_vacuum", "sphere_domain", "sphere_in_profile",
        "two_spheres", "sphere_near_wall", "box_3d", "parallel_plates",
        "custom_2d", "custom_3d",
    ]
    if geometry not in implemented:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "NOT_IMPLEMENTED",
                "message": f"Geometry '{geometry}' not yet implemented",
                "implemented": implemented,
            }
        }, indent=2))]

    # Validate required params
    required_params = {
        "sphere_in_vacuum": ["object_radius", "vacuum_radius"],
        "ellipse_in_vacuum": ["rx", "ry", "vacuum_radius"],
        "disk": ["radius"],
        "box_2d": ["width", "height"],
        "shell_in_vacuum": ["inner_radius", "outer_radius", "vacuum_radius"],
        "cylinder_in_vacuum": ["radius", "height", "vacuum_radius"],
        "sphere_domain": ["radius"],
        "sphere_in_profile": ["object_radius", "domain_radius"],
        "two_spheres": ["radius_1", "radius_2", "separation", "vacuum_radius"],
        "sphere_near_wall": ["object_radius", "wall_distance", "vacuum_radius"],
        "box_3d": ["width", "height", "depth"],
        "parallel_plates": ["plate_separation", "plate_thickness"],
    }

    missing = [p for p in required_params.get(geometry, []) if p not in params]
    if missing:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "MISSING_PARAMS",
                "message": f"Missing required parameters: {missing}",
            }
        }, indent=2))]

    # Special validation for custom geometries
    if geometry == "custom_2d":
        if "shape_file" not in params and "points" not in params:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "MISSING_PARAMS",
                    "message": "custom_2d requires either 'shape_file' or 'points' parameter",
                }
            }, indent=2))]
    elif geometry == "custom_3d":
        if "contour_file" not in params and "contours" not in params:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "MISSING_PARAMS",
                    "message": "custom_3d requires either 'contour_file' or 'contours' parameter",
                }
            }, indent=2))]

    # Get session and generate mesh ID
    session = get_session()
    try:
        mesh_id = session.generate_mesh_id(custom_id)
    except ValueError as e:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "INVALID_ID",
                "message": str(e),
            }
        }, indent=2))]

    # Determine dimension based on geometry and symmetry
    if geometry in ["ellipsoid_in_vacuum", "box_3d", "custom_3d"]:
        dimension = 3
    else:
        dimension = 2

    # Get quality settings
    quality = MESH_QUALITY_SETTINGS[mesh_quality].copy()

    # Apply physics-aware refinement if physics_params provided
    physics_params = args.get("physics_params")
    physics_info = None

    # Process physics_params to compute lambda values per region
    lambda_per_region = {}

    if physics_params:
        # Option 1: Direct lambda dict provided
        if physics_params.get("lambda"):
            lambda_per_region = dict(physics_params["lambda"])

        # Option 2: Compute from alpha + density + n
        elif physics_params.get("alpha") is not None and physics_params.get("density"):
            alpha = physics_params["alpha"]
            n = physics_params.get("n", 1)
            density_dict = physics_params["density"]

            for region_name, density_spec in density_dict.items():
                rho = extract_density_value(density_spec, symmetry, geometry)
                if rho is not None and rho > 0:
                    lambda_val = _compute_lambda(alpha, rho, n)
                    lambda_per_region[region_name] = lambda_val

    # Apply physics refinement if we have lambda values
    if lambda_per_region:
        # Find the minimum lambda (densest region = finest mesh needed)
        min_lambda = min(lambda_per_region.values())
        min_lambda_region = min(lambda_per_region, key=lambda k: lambda_per_region[k])

        # Determine subdomain size for physics refinement
        # Use characteristic size based on geometry
        subdomain_size = None

        if geometry == "sphere_in_vacuum":
            subdomain_size = params.get("object_radius")
        elif geometry == "ellipse_in_vacuum":
            subdomain_size = max(params.get("rx", 0), params.get("ry", 0))
        elif geometry == "shell_in_vacuum":
            subdomain_size = params.get("outer_radius")
        elif geometry == "cylinder_in_vacuum":
            subdomain_size = min(params.get("radius", 0), params.get("height", 0) / 2)
        elif geometry == "two_spheres":
            subdomain_size = min(params.get("radius_1", 0), params.get("radius_2", 0))
        elif geometry == "sphere_near_wall":
            subdomain_size = params.get("object_radius")
        elif geometry == "sphere_in_profile":
            subdomain_size = params.get("object_radius")
        elif geometry == "parallel_plates":
            subdomain_size = params.get("plate_thickness")
        elif geometry == "custom_2d":
            if "points" in params:
                import numpy as np
                pts = np.array(params["points"])
                subdomain_size = float(np.max(pts) - np.min(pts)) / 2
            else:
                subdomain_size = None

        if subdomain_size:
            # Use minimum lambda for refinement (most conservative)
            refined_params = {"lambda_subdomain": min_lambda}
            quality = estimate_physics_refinement(subdomain_size, refined_params, quality)

            if quality.get("physics_refined"):
                physics_info = {
                    "lambda_per_region": lambda_per_region,
                    "lambda_min": min_lambda,
                    "lambda_min_region": min_lambda_region,
                    "shell_thickness": quality.get("shell_thickness"),
                    "cell_min": quality["cell_min_factor"] * subdomain_size,
                    "subdomain_size": subdomain_size,
                    "refinement_applied": True,
                }
                # Include input params if lambda was computed from alpha/density
                if physics_params.get("alpha") is not None:
                    physics_info["computed_from"] = {
                        "alpha": physics_params["alpha"],
                        "density": {k: extract_density_value(v, symmetry, geometry) for k, v in physics_params.get("density", {}).items()},
                        "n": physics_params.get("n", 1),
                    }

    # Determine SELCIE symmetry parameter
    selcie_symmetry = "vertical" if symmetry == "axial" else None

    # Create mesh
    mesh_dir = _get_mesh_dir()

    try:
        MT = MeshingTools(dimension=dimension, path=mesh_dir, display_messages=False)

        if geometry == "sphere_in_vacuum":
            regions, bounds = _create_sphere_in_vacuum(MT, params, quality)
        elif geometry == "ellipse_in_vacuum":
            regions, bounds = _create_ellipse_in_vacuum(MT, params, quality)
        elif geometry == "disk":
            regions, bounds = _create_disk(MT, params, quality)
        elif geometry == "box_2d":
            regions, bounds = _create_box_2d(MT, params, quality)
        elif geometry == "shell_in_vacuum":
            regions, bounds = _create_shell_in_vacuum(MT, params, quality)
        elif geometry == "cylinder_in_vacuum":
            regions, bounds = _create_cylinder_in_vacuum(MT, params, quality)
        elif geometry == "sphere_domain":
            regions, bounds = _create_sphere_domain(MT, params, quality)
        elif geometry == "sphere_in_profile":
            regions, bounds = _create_sphere_in_profile(MT, params, quality)
        elif geometry == "two_spheres":
            regions, bounds = _create_two_spheres(MT, params, quality)
        elif geometry == "sphere_near_wall":
            regions, bounds = _create_sphere_near_wall(MT, params, quality)
        elif geometry == "box_3d":
            regions, bounds = _create_box_3d(MT, params, quality)
        elif geometry == "parallel_plates":
            regions, bounds = _create_parallel_plates(MT, params, quality)
        elif geometry == "custom_2d":
            regions, bounds = _create_custom_2d(MT, params, quality, symmetry)
        elif geometry == "custom_3d":
            regions, bounds = _create_custom_3d(MT, params, quality)
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "NOT_IMPLEMENTED",
                    "message": f"Geometry '{geometry}' handler not found",
                }
            }, indent=2))]

        # Generate the mesh (SELCIE saves to "Saved Meshes" subdir)
        MT.generate_mesh(mesh_id, show_mesh=False)

        # Convert to XDMF for FEniCS
        MT.msh_2_xdmf(mesh_id, delete_old_file=True, auto_override=True)

        # Full path to mesh files
        mesh_path = os.path.join(mesh_dir, "Saved Meshes", mesh_id)

        # Get mesh statistics by reading the generated mesh
        n_cells = 0
        n_vertices = 0
        try:
            import meshio
            mesh_file = os.path.join(mesh_path, "mesh.xdmf")
            mesh = meshio.read(mesh_file)
            n_vertices = len(mesh.points)
            for cell_block in mesh.cells:
                n_cells += len(cell_block.data)
        except Exception:
            pass  # meshio may not be installed or file format issue

        # Check cell count limit
        allow_large_mesh = args.get("allow_large_mesh", False)
        if n_cells > DEFAULT_MAX_CELLS and not allow_large_mesh:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "MESH_TOO_LARGE",
                    "message": (
                        f"Mesh has {n_cells:,} cells, exceeding limit of {DEFAULT_MAX_CELLS:,}. "
                        f"Use coarser mesh_quality or set allow_large_mesh=true to override."
                    ),
                    "n_cells": n_cells,
                    "limit": DEFAULT_MAX_CELLS,
                }
            }, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "MESH_GENERATION_FAILED",
                "message": str(e),
            }
        }, indent=2))]

    # Store in session
    mesh_info = MeshInfo(
        mesh_id=mesh_id,
        geometry=geometry,
        dimension=dimension,
        n_cells=n_cells,
        n_vertices=n_vertices,
        symmetry=symmetry,
        mesh_path=mesh_path,
        params=params,
    )

    # Create region name -> marker mapping
    # SELCIE assigns markers in the order subdomains are created (0, 1, 2, ...)
    # The regions list is already in creation order from the geometry functions
    regions_dict = {region: i for i, region in enumerate(regions)}

    mesh_info.regions = regions_dict
    session.add_mesh(mesh_info)

    # Build response
    result = {
        "mesh_id": mesh_id,
        "geometry": geometry,
        "symmetry": symmetry,
        "dimension": dimension,
        "n_cells": n_cells,
        "n_vertices": n_vertices,
        "regions": regions_dict,
        "domain_bounds": bounds,
        "mesh_path": mesh_path,
        "quality": mesh_quality,
    }

    # Add physics refinement info if applied
    if physics_info:
        result["physics_refinement"] = physics_info

    # Add warning for large meshes
    if n_cells > 100000:
        warnings.append(f"Very large mesh ({n_cells:,} cells). Solving may take a long time and use significant memory. Consider using a coarser mesh quality.")
    elif n_cells > 50000:
        warnings.append(f"Large mesh ({n_cells:,} cells). Solving may be slow. Consider using a coarser mesh quality if performance is an issue.")

    # Include any warnings in response
    if warnings:
        result["warnings"] = warnings

    return [TextContent(type="text", text=json.dumps(result, indent=2))]
