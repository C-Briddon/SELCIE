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


# Mesh quality settings for 2D: (CellSizeMin, CellSizeMax, DistMax) relative to object size
# Consistent 2.5x ramp between levels, 4x max/min ratio
MESH_QUALITY_SETTINGS = {
    "very_coarse": {"cell_min_factor": 0.125, "cell_max_factor": 0.5, "dist_max_factor": 2.0},
    "coarse": {"cell_min_factor": 0.05, "cell_max_factor": 0.2, "dist_max_factor": 1.25},
    "medium": {"cell_min_factor": 0.02, "cell_max_factor": 0.08, "dist_max_factor": 0.5},
    "fine": {"cell_min_factor": 0.008, "cell_max_factor": 0.032, "dist_max_factor": 0.2},
    "very_fine": {"cell_min_factor": 0.0032, "cell_max_factor": 0.0128, "dist_max_factor": 0.1},
}

# Mesh quality settings for 3D: more gradual increments since cell count scales as 1/h³
# ~1.6x ramp between levels → ~4x cell count increase per level (1.6³ ≈ 4.1)
MESH_QUALITY_SETTINGS_3D = {
    "very_coarse": {"cell_min_factor": 0.20, "cell_max_factor": 0.8, "dist_max_factor": 2.0},
    "coarse": {"cell_min_factor": 0.125, "cell_max_factor": 0.5, "dist_max_factor": 1.5},
    "medium": {"cell_min_factor": 0.08, "cell_max_factor": 0.32, "dist_max_factor": 1.0},
    "fine": {"cell_min_factor": 0.05, "cell_max_factor": 0.2, "dist_max_factor": 0.6},
    "very_fine": {"cell_min_factor": 0.032, "cell_max_factor": 0.128, "dist_max_factor": 0.4},
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
    "custom_2d_axial": "axial",
    "custom_2d_translation": "translation",
    "custom_3d": "none",  # True 3D mesh
    "custom_step": "none",  # 3D mesh from STEP file import
}

# All geometries have fixed symmetry determined by their type.
# This was previously configurable for custom_2d, but that has been split into
# custom_2d_axial and custom_2d_translation with fixed symmetries.

# Refinement limits to prevent excessive cell counts
REFINEMENT_LIMITS = {
    "min_cell_size_factor": 0.002,  # Minimum cell size as fraction of object size
    "max_refinement_ratio": 50,      # Max ratio of largest to smallest cells
    "target_boundary_cells": 5,      # Target cells across thin shell
}

# Default maximum cell count to prevent accidentally creating huge meshes
DEFAULT_MAX_CELLS = 200000


def validate_geometry_params(geometry: str, params: dict) -> list[str]:
    """Validate geometry parameters for logical consistency.

    Returns a list of error messages. Empty list means validation passed.
    """
    errors = []

    def check_positive(name: str, value: float | None) -> None:
        if value is not None and value <= 0:
            errors.append(f"{name} must be positive (got {value})")

    def check_less_than(name1: str, val1: float | None, name2: str, val2: float | None) -> None:
        if val1 is not None and val2 is not None and val1 >= val2:
            errors.append(f"{name1} ({val1}) must be less than {name2} ({val2})")

    # Common parameter extraction
    object_radius = params.get("object_radius")
    domain_radius = params.get("domain_radius")
    wall_thickness = params.get("wall_thickness")

    # Check all provided numeric params are positive
    positive_params = [
        "object_radius", "domain_radius", "wall_thickness",
        "rx", "ry", "object_height",
        "domain_width", "domain_height", "domain_depth",
        "inner_radius", "outer_radius",
        "radius_1", "radius_2", "separation",
        "wall_distance", "plate_separation", "plate_thickness",
    ]
    for param_name in positive_params:
        if param_name in params:
            check_positive(param_name, params[param_name])

    # Geometry-specific validation
    if geometry == "sphere_in_vacuum":
        check_less_than("object_radius", object_radius, "domain_radius", domain_radius)

    elif geometry == "ellipse_in_vacuum":
        rx, ry = params.get("rx"), params.get("ry")
        if rx is not None and ry is not None and domain_radius is not None:
            max_semi = max(rx, ry)
            if max_semi >= domain_radius:
                errors.append(
                    f"Ellipse semi-axes (rx={rx}, ry={ry}) must fit within "
                    f"domain_radius ({domain_radius})"
                )

    elif geometry == "cylinder_in_vacuum":
        object_height = params.get("object_height")
        check_less_than("object_radius", object_radius, "domain_radius", domain_radius)
        if object_height is not None and domain_radius is not None:
            half_height = object_height / 2
            if half_height >= domain_radius:
                errors.append(
                    f"Cylinder half-height ({half_height}) must be less than "
                    f"domain_radius ({domain_radius})"
                )

    elif geometry == "shell_in_vacuum":
        inner_radius = params.get("inner_radius")
        outer_radius = params.get("outer_radius")
        check_less_than("inner_radius", inner_radius, "outer_radius", outer_radius)
        check_less_than("outer_radius", outer_radius, "domain_radius", domain_radius)

    elif geometry == "two_spheres":
        radius_1 = params.get("radius_1")
        radius_2 = params.get("radius_2")
        separation = params.get("separation")
        if radius_1 is not None and radius_2 is not None and separation is not None:
            if separation < radius_1 + radius_2:
                errors.append(
                    f"separation ({separation}) must be >= radius_1 + radius_2 "
                    f"({radius_1 + radius_2}) to avoid overlap"
                )
        # Check spheres fit in domain (they're centered at z = ±separation/2)
        if separation is not None and domain_radius is not None:
            max_radius = max(radius_1 or 0, radius_2 or 0)
            required_domain = separation / 2 + max_radius
            if required_domain >= domain_radius:
                errors.append(
                    f"Spheres extend to z={required_domain:.3f} but domain_radius "
                    f"is only {domain_radius}"
                )

    elif geometry == "sphere_near_wall":
        wall_distance = params.get("wall_distance")
        wall_thick = params.get("wall_thickness", 0.1)  # Has default
        if object_radius is not None and wall_distance is not None:
            if wall_distance <= object_radius:
                errors.append(
                    f"wall_distance ({wall_distance}) must be greater than "
                    f"object_radius ({object_radius}) so sphere doesn't intersect wall"
                )
        if wall_distance is not None and wall_thick is not None and domain_radius is not None:
            if wall_distance + wall_thick >= domain_radius:
                errors.append(
                    f"wall_distance + wall_thickness ({wall_distance + wall_thick}) "
                    f"must be less than domain_radius ({domain_radius})"
                )

    elif geometry == "sphere_in_profile":
        center_z = params.get("center_z", 0)
        check_less_than("object_radius", object_radius, "domain_radius", domain_radius)
        if object_radius is not None and domain_radius is not None:
            if abs(center_z) + object_radius >= domain_radius:
                errors.append(
                    f"Sphere at center_z={center_z} with radius={object_radius} "
                    f"extends beyond domain_radius ({domain_radius})"
                )

    elif geometry == "parallel_plates":
        plate_separation = params.get("plate_separation")
        domain_height = params.get("domain_height")
        if domain_height is not None and plate_separation is not None:
            if domain_height < plate_separation:
                errors.append(
                    f"domain_height ({domain_height}) must be >= plate_separation "
                    f"({plate_separation})"
                )

    return errors


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
                    "custom_2d_axial", "custom_2d_translation", "custom_3d", "custom_step"
                ],
                "description": (
                    "Geometry template. Choose based on physical setup:\n\n"
                    "OBJECT-IN-VACUUM (screening/force calculations):\n"
                    "- sphere_in_vacuum: Spherical source in vacuum. Regions: object, vacuum [+wall]. "
                    "Fixed symmetry: axial (2D). 'r' = spherical radius.\n"
                    "- ellipse_in_vacuum: Oblate/prolate ellipsoid in vacuum. Regions: object, vacuum [+wall]. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- cylinder_in_vacuum: Cylindrical source in vacuum. Regions: cylinder, vacuum [+wall]. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- shell_in_vacuum: Hollow spherical shell in vacuum. Regions: shell, vacuum [+wall]. "
                    "Fixed symmetry: axial (2D). 'r' = spherical radius.\n"
                    "- two_spheres: Two spheres for force calculations. Regions: sphere_1, sphere_2, vacuum [+wall]. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- sphere_near_wall: Sphere near planar wall. Regions: sphere, wall, vacuum. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- sphere_in_profile: Sphere in spatially-varying density profile (e.g., NFW, isothermal). "
                    "Regions: sphere, background [+wall]. Fixed symmetry: axial (2D). 'r' = spherical radius. "
                    "Use center_z to offset sphere along z-axis.\n\n"
                    "[+wall] = optional wall region added if wall_thickness parameter is set.\n\n"
                    "PLAIN DOMAINS (no interior object):\n"
                    "- sphere_domain: For spherically-symmetric profiles (e.g. NFW, isothermal). Regions: domain [+wall]. "
                    "Fixed symmetry: axial (2D). 'r' = spherical radius.\n"
                    "- disk: For cylindrically-symmetric profiles. Regions: domain [+wall]. "
                    "Fixed symmetry: axial (2D). 'r' = cylindrical radius.\n"
                    "- box_2d: 2D Cartesian rectangle. Regions: domain. Fixed symmetry: translation (2D).\n"
                    "- box_3d: 3D Cartesian box. Regions: domain. Fixed symmetry: none (true 3D).\n"
                    "- parallel_plates: Two parallel plates with vacuum gap. Regions: vacuum, plate. "
                    "Fixed symmetry: translation (2D extended in y). 'x' = perpendicular to plates.\n\n"
                    "CUSTOM SHAPES:\n"
                    "- custom_2d_axial: Arbitrary 2D axisymmetric shape. Points are [r, z] with r >= 0, "
                    "revolved around z-axis. Regions: object, vacuum [+wall]. Fixed symmetry: axial (2D).\n"
                    "- custom_2d_translation: Arbitrary 2D shape with translation symmetry. Points are [x, y], "
                    "extruded in z. Regions: object, vacuum [+wall]. Fixed symmetry: translation (2D).\n"
                    "- custom_3d: Arbitrary 3D shape from contours. Regions: object. Fixed symmetry: none (true 3D).\n"
                    "- custom_step: Import 3D geometry from STEP/IGES/BREP file. Object is centered in spherical vacuum domain. "
                    "Regions: object, vacuum. Fixed symmetry: none (true 3D)."
                ),
            },
            "params": {
                "type": "object",
                "description": "Geometry-specific parameters.",
                "properties": {
                    "object_radius": {"type": "number", "description": "Radius of object inside domain. Used by: sphere_in_vacuum, sphere_in_profile, sphere_near_wall, cylinder_in_vacuum."},
                    "domain_radius": {"type": "number", "description": "Outer boundary radius of simulation domain. Used by: sphere_in_vacuum, ellipse_in_vacuum, cylinder_in_vacuum, shell_in_vacuum, two_spheres, sphere_near_wall, sphere_in_profile, sphere_domain, disk, custom_2d_axial, custom_2d_translation."},
                    "center_z": {"type": "number", "description": "Z-position of sphere center (sphere_in_profile, default 0)"},
                    "wall_thickness": {"type": "number", "description": "Optional outer wall thickness. Adds 'wall' region if set. Supported by: sphere_in_vacuum, ellipse_in_vacuum, cylinder_in_vacuum, shell_in_vacuum, two_spheres, sphere_in_profile, sphere_domain, disk, custom_2d_axial, custom_2d_translation. For sphere_near_wall, wall is always present (defaults to 0.1 if not specified)."},
                    "rx": {"type": "number", "description": "Semi-axis in r/x direction (ellipse_in_vacuum)"},
                    "ry": {"type": "number", "description": "Semi-axis in z/y direction (ellipse_in_vacuum)"},
                    "object_height": {"type": "number", "description": "Height of cylindrical object (cylinder_in_vacuum)"},
                    "domain_width": {"type": "number", "description": "Width of domain in x direction (box_2d, box_3d)"},
                    "domain_height": {"type": "number", "description": "Height of domain in y direction (box_2d, box_3d, parallel_plates)"},
                    "domain_depth": {"type": "number", "description": "Depth of domain in z direction (box_3d)"},
                    "inner_radius": {"type": "number", "description": "Inner radius for shell_in_vacuum"},
                    "outer_radius": {"type": "number", "description": "Outer radius for shell_in_vacuum"},
                    "radius_1": {"type": "number", "description": "First sphere radius (two_spheres)"},
                    "radius_2": {"type": "number", "description": "Second sphere radius (two_spheres)"},
                    "separation": {"type": "number", "description": "Center-to-center separation (two_spheres)"},
                    "wall_distance": {"type": "number", "description": "Distance from sphere center to wall (sphere_near_wall)"},
                    "points": {"type": "array", "description": "Array of [r,z] points for custom_2d_axial or [x,y] points for custom_2d_translation"},
                    "shape_file": {"type": "string", "description": "Path to file with shape points (custom_2d_axial, custom_2d_translation)"},
                    "contour_file": {"type": "string", "description": "Path to 3D contour file (custom_3d)"},
                    "contours": {"type": "array", "description": "List of contour point lists for custom_3d"},
                    "step_file": {"type": "string", "description": "Path to STEP/IGES/BREP file (custom_step)"},
                    "plate_separation": {"type": "number", "description": "Gap between inner surfaces of plates (parallel_plates)"},
                    "plate_thickness": {"type": "number", "description": "Thickness of each plate (parallel_plates)"},
                },
            },
            "mesh_quality": {
                "type": "string",
                "enum": ["very_coarse", "coarse", "medium", "fine", "very_fine"],
                "default": "medium",
                "description": "Mesh resolution.",
            },
            # Note: symmetry is now fixed per geometry type and cannot be overridden.
            # Each geometry description specifies its symmetry (axial, translation, or none).
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
                    "Physics parameters for automatic thin-shell mesh refinement (recommended when available). "
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
    domain_radius = params["domain_radius"]
    wall_thickness = params.get("wall_thickness")

    # Create the sphere using explicit points for smooth boundary
    # (using create_ellipse results in jagged boundaries due to GMSH arc discretization)
    n_boundary_points = 50  # Matches Sphere_Standalone.py
    points = _sphere_points(object_radius, n_boundary_points)
    points = MT.constrain_distance(points)
    MT.points_to_surface(points)

    # Mark as subdomain with refinement
    # cell_min based on object size to resolve boundary
    # cell_max based on domain_radius so cells can grow large far from object
    cell_min = quality["cell_min_factor"] * object_radius
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum)
    bg_cell_min = quality["cell_min_factor"] * domain_radius
    bg_cell_max = quality["cell_max_factor"] * domain_radius
    bg_dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=bg_dist_max,
        background_radius=domain_radius,
        wall_thickness=wall_thickness,  # None if not specified
        #refine_outer_wall_boundary=True,  # Use scaled cell sizes for wall
        symmetry="vertical",  # Axisymmetric - only mesh r >= 0
    )

    regions = ["object", "vacuum"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
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
    domain_radius = params["domain_radius"]
    wall_thickness = params.get("wall_thickness")

    # Create the ellipse using explicit points for smooth boundary
    n_boundary_points = 50
    points = _ellipse_points(rx, ry, n_boundary_points)
    points = MT.constrain_distance(points)
    MT.points_to_surface(points)

    # Mark as subdomain with refinement
    # cell_min based on ellipse size to resolve boundary
    # cell_max based on domain_radius so cells can grow large far from object
    char_size = max(rx, ry)
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum)
    bg_cell_min = quality["cell_min_factor"] * domain_radius
    bg_cell_max = quality["cell_max_factor"] * domain_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=dist_max,
        background_radius=domain_radius,
        wall_thickness=wall_thickness,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["object", "vacuum"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
    }

    return regions, bounds


def _create_disk(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create simple disk domain (no interior object)."""
    domain_radius = params["domain_radius"]
    wall_thickness = params.get("wall_thickness")

    # For plain domains, use domain_radius directly (no object boundary to resolve)
    cell_min = quality["cell_min_factor"] * domain_radius
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_background_mesh(
        CellSizeMin=cell_min,
        CellSizeMax=cell_max,
        DistMax=dist_max,
        background_radius=domain_radius,
        wall_thickness=wall_thickness,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["domain"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
    }

    return regions, bounds


def _create_box_2d(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create 2D rectangular domain."""
    width = params["domain_width"]
    height = params["domain_height"]

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
    Uses translation symmetry - the 2D mesh extends via translation in y.

    Layout (in x-y plane):
        [plate] [vacuum] [plate]
        x: 0 to t | t to t+sep | t+sep to 2t+sep
        where t = plate_thickness, sep = plate_separation

    Note: Both plates share the same region marker. For different plate
    densities, use position-dependent expressions.
    """
    plate_separation = params["plate_separation"]
    plate_thickness = params["plate_thickness"]
    domain_height = params.get("domain_height", plate_separation)

    total_width = 2 * plate_thickness + plate_separation
    dist_max = quality["dist_max_factor"] * plate_separation

    # Cell sizes for plates and vacuum
    cell_min_plate = quality["cell_min_factor"] * plate_thickness
    cell_max_plate = quality["cell_max_factor"] * plate_separation
    cell_min_vac = quality["cell_min_factor"] * plate_separation
    cell_max_vac = quality["cell_max_factor"] * plate_separation

    # Create vacuum region first (inner)
    vac_x_min = plate_thickness
    vac_x_max = plate_thickness + plate_separation
    vacuum = MT.points_to_surface([
        (vac_x_min, 0, 0),
        (vac_x_max, 0, 0),
        (vac_x_max, domain_height, 0),
        (vac_x_min, domain_height, 0),
    ])
    MT.create_subdomain(CellSizeMin=cell_min_vac, CellSizeMax=cell_max_vac, DistMax=dist_max)

    # Create full domain embedding vacuum (plates are the outer region)
    MT.points_to_surface([
        (0, 0, 0),
        (total_width, 0, 0),
        (total_width, domain_height, 0),
        (0, domain_height, 0),
    ], embed=vacuum)
    MT.create_subdomain(CellSizeMin=cell_min_plate, CellSizeMax=cell_max_plate, DistMax=dist_max)

    # Regions: vacuum first, then plate (both plates share same marker)
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
    domain_radius = params["domain_radius"]
    wall_thickness = params.get("wall_thickness")

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
    # cell_max based on domain_radius so cells can grow large far from shell
    char_size = outer_radius - inner_radius
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum)
    bg_cell_min = quality["cell_min_factor"] * domain_radius
    bg_cell_max = quality["cell_max_factor"] * domain_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=dist_max,
        background_radius=domain_radius,
        wall_thickness=wall_thickness,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["shell", "vacuum"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
    }

    return regions, bounds


def _create_cylinder_in_vacuum(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create cylinder in vacuum geometry (2D axisymmetric = rectangle)."""
    object_radius = params["object_radius"]
    object_height = params["object_height"]
    domain_radius = params["domain_radius"]
    wall_thickness = params.get("wall_thickness")

    # In 2D axisymmetric, a cylinder is a rectangle in (r, z)
    rect = MT.create_rectangle(dx=object_radius, dy=object_height)
    MT.translate_x(rect, dx=object_radius / 2)  # Move to positive r

    # Mark as subdomain with refinement
    # cell_min based on cylinder size to resolve boundary
    # cell_max based on domain_radius so cells can grow large far from object
    char_size = min(object_radius, object_height)
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum)
    bg_cell_min = quality["cell_min_factor"] * domain_radius
    bg_cell_max = quality["cell_max_factor"] * domain_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=dist_max,
        background_radius=domain_radius,
        wall_thickness=wall_thickness,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["cylinder", "vacuum"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
    }

    return regions, bounds


def _create_sphere_domain(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create simple spherical domain (no interior object)."""
    domain_radius = params["domain_radius"]
    wall_thickness = params.get("wall_thickness")

    # For plain domains, use domain_radius directly (no object boundary to resolve)
    cell_min = quality["cell_min_factor"] * domain_radius
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_background_mesh(
        CellSizeMin=cell_min,
        CellSizeMax=cell_max,
        DistMax=dist_max,
        background_radius=domain_radius,
        wall_thickness=wall_thickness,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["domain"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
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
    wall_thickness = params.get("wall_thickness")

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
        wall_thickness=wall_thickness,
        symmetry="vertical",
    )

    regions = ["sphere", "background"]
    if wall_thickness:
        regions.append("wall")

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
    domain_radius = params["domain_radius"]
    wall_thickness = params.get("wall_thickness")

    # Create first sphere (source) at origin using explicit points
    # cell_min based on sphere size to resolve boundary
    # cell_max based on domain_radius so cells can grow large far from object
    n_boundary_points = 50
    points_1 = _sphere_points(radius_1, n_boundary_points)
    points_1 = MT.constrain_distance(points_1)
    MT.points_to_surface(points_1)
    cell_min = quality["cell_min_factor"] * radius_1
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create second sphere (test mass) offset along z using explicit points
    points_2 = _sphere_points(radius_2, n_boundary_points)
    # Offset z coordinate by separation
    points_2 = [[p[0], p[1] + separation, p[2]] for p in points_2]
    points_2 = MT.constrain_distance(points_2)
    MT.points_to_surface(points_2)
    cell_min2 = quality["cell_min_factor"] * radius_2
    cell_max2 = quality["cell_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min2, CellSizeMax=cell_max2, DistMax=dist_max)

    # Create background (vacuum)
    bg_cell_min = quality["cell_min_factor"] * domain_radius
    bg_cell_max = quality["cell_max_factor"] * domain_radius
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=dist_max,
        background_radius=domain_radius,
        wall_thickness=wall_thickness,
        refine_outer_wall_boundary=True,  # Use scaled cell sizes if wall added
        symmetry="vertical",  # Axisymmetric
    )

    regions = ["sphere_1", "sphere_2", "vacuum"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
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
    domain_radius = params["domain_radius"]

    # Wall surface is at z=0, wall extends from z=-wall_thickness to z=0
    # Sphere center is at z=wall_distance

    # Create the wall as a half-disk at the bottom
    # Wall region: from z=-wall_thickness to z=0
    wall_disk = MT.create_ellipse(rx=domain_radius, ry=domain_radius)
    wall_rect = MT.create_rectangle(dx=2 * domain_radius, dy=wall_thickness)
    MT.translate_y(wall_rect, dy=-wall_thickness / 2)  # Center at y = -wall_thickness/2
    wall = MT.intersect_shapes(wall_disk, wall_rect)

    # Mark wall as subdomain
    # cell_min based on wall thickness to resolve boundary
    # cell_max based on domain_radius so cells can grow large far from wall
    wall_cell_min = quality["cell_min_factor"] * wall_thickness
    wall_cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=wall_cell_min, CellSizeMax=wall_cell_max, DistMax=dist_max)

    # Create sphere offset above wall surface (z=0)
    sphere = MT.create_ellipse(rx=object_radius, ry=object_radius)
    MT.translate_y(sphere, dy=wall_distance)  # Sphere center at z=wall_distance

    # Mark sphere as subdomain with refinement
    # cell_min based on sphere size to resolve boundary
    # cell_max based on domain_radius so cells can grow large far from object
    cell_min = quality["cell_min_factor"] * object_radius
    cell_max = quality["cell_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create vacuum region (upper half-disk, z >= 0)
    vacuum_disk = MT.create_ellipse(rx=domain_radius, ry=domain_radius)
    upper_rect = MT.create_rectangle(dx=2 * domain_radius, dy=domain_radius)
    MT.translate_y(upper_rect, dy=domain_radius / 2)  # Center at y = domain_radius/2
    vacuum = MT.intersect_shapes(vacuum_disk, upper_rect)

    bg_cell_min = quality["cell_min_factor"] * domain_radius
    bg_cell_max = quality["cell_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=bg_cell_min, CellSizeMax=bg_cell_max, DistMax=dist_max)

    regions = ["wall", "sphere", "vacuum"]
    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -wall_thickness,  # Bottom of wall
        "z_max": domain_radius,
    }

    return regions, bounds


def _create_box_3d(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create 3D rectangular box domain."""
    width = params["domain_width"]    # x extent
    height = params["domain_height"]  # y extent
    depth = params["domain_depth"]    # z extent

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

    Used by:
    - custom_2d_axial: Points are [r, z] with r >= 0, revolved around z-axis
    - custom_2d_translation: Points are [x, y], extruded in z direction

    Supports:
    - shape_file: Path to text file with coordinates per line
    - points: Direct list of coordinate pairs

    For axial symmetry, points with r < 0 are rejected with an error.
    """
    import numpy as np

    domain_radius = params.get("domain_radius", 1.0)
    wall_thickness = params.get("wall_thickness")
    is_axisymmetric = symmetry == "axial"

    # Load points from file or use direct points
    if "shape_file" in params:
        points_2d = np.loadtxt(params["shape_file"])
    elif "points" in params:
        points_2d = np.array(params["points"])
    else:
        raise ValueError("Must provide either 'shape_file' or 'points'")

    # For axisymmetric, validate r >= 0 (r = 0 on axis is valid)
    if is_axisymmetric:
        if np.any(points_2d[:, 0] < 0):
            n_negative = np.sum(points_2d[:, 0] < 0)
            raise ValueError(
                f"custom_2d_axial requires all r values >= 0. "
                f"Found {n_negative} points with r < 0. "
                f"Use custom_2d_translation for shapes with negative x values."
            )

    # Convert to 3D coordinates (z=0) for SELCIE
    if points_2d.shape[1] == 2:
        points_3d = [[p[0], p[1], 0.0] for p in points_2d]
    else:
        points_3d = points_2d.tolist()

    # Create shape from points
    MT.points_to_surface(points_3d)

    # Mark as subdomain with refinement
    # cell_min based on object size to resolve boundary
    # cell_max based on domain_radius so cells can grow large far from object
    char_size = float(np.max(points_2d) - np.min(points_2d))
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * domain_radius
    dist_max = quality["dist_max_factor"] * domain_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum) if domain_radius provided
    if domain_radius:
        bg_cell_min = quality["cell_min_factor"] * domain_radius
        bg_cell_max = quality["cell_max_factor"] * domain_radius
        bg_kwargs = {
            "CellSizeMin": bg_cell_min,
            "CellSizeMax": bg_cell_max,
            "DistMax": dist_max,
            "background_radius": domain_radius,
            "wall_thickness": wall_thickness,
            "refine_outer_wall_boundary": True,  # Use scaled cell sizes for wall
        }
        if is_axisymmetric:
            bg_kwargs["symmetry"] = "vertical"
        MT.create_background_mesh(**bg_kwargs)

    # Calculate actual bounds from points
    x_coords = points_2d[:, 0]
    y_coords = points_2d[:, 1]

    regions = ["object", "vacuum"]
    if wall_thickness:
        regions.append("wall")

    bounds = {
        "r_min": 0.0,
        "r_max": domain_radius,
        "z_min": -domain_radius,
        "z_max": domain_radius,
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


def _create_custom_step(
    params: dict,
    quality: dict,
    mesh_dir: str,
    mesh_id: str,
    physics_info: dict = None,
) -> tuple[list[str], dict, str]:
    """Create mesh from imported STEP/IGES/BREP file.

    This function bypasses MeshingTools entirely since we need direct control
    over gmsh for STEP file import and boolean operations.

    Imports a CAD file, centers it at the origin, embeds it in a spherical
    vacuum domain, and creates a mesh with object and vacuum subdomains.

    Required params:
    - step_file: Path to STEP, IGES, or BREP file
    - domain_radius: Radius of spherical vacuum domain

    Optional params:
    - scale: Scale factor for imported geometry (default 1.0)
    - physics_info: Dict with physics refinement info (lambda_min, etc.)

    Returns:
    - regions: List of region names
    - bounds: Dict with geometry bounds
    - mesh_path: Path to generated mesh files
    """
    import gmsh
    import os

    step_file = params["step_file"]
    domain_radius = params["domain_radius"]
    scale = params.get("scale", 1.0)

    # Validate file exists
    if not os.path.exists(step_file):
        raise FileNotFoundError(f"STEP file not found: {step_file}")

    # Initialize gmsh fresh (don't use MeshingTools)
    gmsh.initialize()
    gmsh.model.add(mesh_id)
    gmsh.option.setNumber("General.Terminal", 0)  # Suppress terminal output

    try:
        # Import the STEP file using OpenCASCADE kernel
        imported_entities = gmsh.model.occ.importShapes(step_file, highestDimOnly=True)
        gmsh.model.occ.synchronize()

        if not imported_entities:
            raise ValueError(f"No geometry entities found in file: {step_file}")

        # Get bounding box of all imported entities
        xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
        xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')
        for dim, tag in imported_entities:
            bx1, by1, bz1, bx2, by2, bz2 = gmsh.model.occ.getBoundingBox(dim, tag)
            xmin, ymin, zmin = min(xmin, bx1), min(ymin, by1), min(zmin, bz1)
            xmax, ymax, zmax = max(xmax, bx2), max(ymax, by2), max(zmax, bz2)

        # Calculate center and characteristic size
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2
        char_size = min(xmax - xmin, ymax - ymin, zmax - zmin)

        # Apply scale if needed
        if scale != 1.0:
            gmsh.model.occ.dilate(imported_entities, center_x, center_y, center_z, scale, scale, scale)
            char_size *= scale
            gmsh.model.occ.synchronize()
            # Recalculate bounds after scaling
            xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
            xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')
            for dim, tag in imported_entities:
                bx1, by1, bz1, bx2, by2, bz2 = gmsh.model.occ.getBoundingBox(dim, tag)
                xmin, ymin, zmin = min(xmin, bx1), min(ymin, by1), min(zmin, bz1)
                xmax, ymax, zmax = max(xmax, bx2), max(ymax, by2), max(zmax, bz2)
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            center_z = (zmin + zmax) / 2

        # Translate to center at origin
        dx, dy, dz = -center_x, -center_y, -center_z
        gmsh.model.occ.translate(imported_entities, dx, dy, dz)
        gmsh.model.occ.synchronize()

        # Update bounds after centering
        xmin, ymin, zmin = xmin + dx, ymin + dy, zmin + dz
        xmax, ymax, zmax = xmax + dx, ymax + dy, zmax + dz

        # Check that object fits within domain
        max_extent = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax), abs(zmin), abs(zmax))
        if max_extent >= domain_radius:
            raise ValueError(
                f"Imported geometry extent ({max_extent:.4f}) exceeds domain_radius ({domain_radius}). "
                f"Increase domain_radius or use scale parameter to shrink geometry."
            )

        # Create spherical vacuum domain
        vacuum_sphere = gmsh.model.occ.addSphere(0, 0, 0, domain_radius)
        gmsh.model.occ.synchronize()

        # Boolean cut: vacuum = sphere - imported object(s)
        # Get all 3D entities from imported shapes
        object_entities = [(dim, tag) for dim, tag in imported_entities if dim == 3]

        if not object_entities:
            raise ValueError("No 3D volumes found in imported file. STEP file may contain only surfaces.")

        # Perform boolean cut: vacuum with object hole
        vacuum_with_hole, vacuum_map = gmsh.model.occ.cut(
            [(3, vacuum_sphere)],
            object_entities,
            removeObject=True,
            removeTool=False  # Keep the object
        )
        gmsh.model.occ.synchronize()

        # Set up mesh sizes
        # cell_min based on object size (for fine resolution near/in object)
        # cell_max based on domain size (for coarse cells in far vacuum)
        cell_min = quality["cell_min_factor"] * char_size
        cell_max = quality["cell_max_factor"] * domain_radius

        # Create physical groups for 3D volumes
        # SELCIE solver expects markers to be 0, 1, 2, ... (not gmsh default 1, 2, 3, ...)
        object_tags = [tag for dim, tag in object_entities if dim == 3]
        gmsh.model.addPhysicalGroup(3, object_tags, tag=0, name="object")

        vacuum_tags = [tag for dim, tag in vacuum_with_hole if dim == 3]
        gmsh.model.addPhysicalGroup(3, vacuum_tags, tag=1, name="vacuum")

        # Create physical groups for 2D boundary surfaces
        # Get all 2D surface entities
        all_surfaces = gmsh.model.getEntities(2)

        # Separate object surfaces from outer boundary
        # The outer boundary is on the vacuum sphere surface
        object_surface_tags = []
        outer_boundary_tags = []

        for dim, tag in all_surfaces:
            # Get bounding box of surface
            bbox = gmsh.model.occ.getBoundingBox(dim, tag)
            # Check if surface touches the outer sphere
            max_dist = max(abs(bbox[0]), abs(bbox[1]), abs(bbox[2]),
                          abs(bbox[3]), abs(bbox[4]), abs(bbox[5]))
            if max_dist > domain_radius * 0.95:
                outer_boundary_tags.append(tag)
            else:
                object_surface_tags.append(tag)

        # Physical groups for surfaces: object=0, outer=1
        if object_surface_tags:
            gmsh.model.addPhysicalGroup(2, object_surface_tags, tag=0, name="object_surface")
        if outer_boundary_tags:
            gmsh.model.addPhysicalGroup(2, outer_boundary_tags, tag=1, name="outer_boundary")

        # Set global mesh size options
        gmsh.option.setNumber("Mesh.MeshSizeMin", cell_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", cell_max)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

        # Apply refinement near object surfaces (always for STEP files)
        # Use physics-based parameters if provided, otherwise use geometry-based defaults
        if object_surface_tags:
            if physics_info and physics_info.get("lambda_min"):
                # Physics-aware refinement based on screening length
                lambda_min = physics_info["lambda_min"]
                shell_thickness = lambda_min
                dist_max = min(3 * shell_thickness, quality["dist_max_factor"] * char_size)
                target_boundary_cells = 5
                cell_min_refined = shell_thickness / target_boundary_cells
            else:
                # Geometry-based refinement using quality settings
                # cell_min_factor controls refinement near object surfaces
                cell_min_refined = quality["cell_min_factor"] * char_size
                dist_max = quality["dist_max_factor"] * char_size

            # Apply limits
            min_cell = char_size * 0.001  # Don't go too small
            cell_min_refined = max(min_cell, cell_min_refined)

            # Ensure refinement ratio isn't too extreme (max 50:1)
            if cell_max / cell_min_refined > 50:
                cell_min_refined = cell_max / 50

            # Cell size inside the object - based on char_size (min dimension)
            cell_size_object = quality["cell_min_factor"] * char_size

            # Create constant field for object - uniform fine cells throughout
            object_field = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(object_field, "VIn", cell_size_object)
            gmsh.model.mesh.field.setNumber(object_field, "VOut", cell_max)
            gmsh.model.mesh.field.setNumbers(object_field, "VolumesList", object_tags)

            # Create distance field from object surfaces (for vacuum grading)
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", object_surface_tags)

            # Create threshold field for vacuum - grades from boundary to far field
            # Use coarser cells in vacuum, transition quickly to cell_max
            vacuum_cell_min = quality["cell_max_factor"] * domain_radius * 0.5
            vacuum_dist_max = quality["dist_max_factor"] * domain_radius * 0.25  # fast transition

            vacuum_thresh_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(vacuum_thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(vacuum_thresh_field, "SizeMin", vacuum_cell_min)
            gmsh.model.mesh.field.setNumber(vacuum_thresh_field, "SizeMax", cell_max)
            gmsh.model.mesh.field.setNumber(vacuum_thresh_field, "DistMin", 0)
            gmsh.model.mesh.field.setNumber(vacuum_thresh_field, "DistMax", vacuum_dist_max)

            # Restrict vacuum threshold to vacuum volumes only
            vacuum_field = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(vacuum_field, "InField", vacuum_thresh_field)
            gmsh.model.mesh.field.setNumbers(vacuum_field, "VolumesList", vacuum_tags)

            # Combine: object field + vacuum field
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [object_field, vacuum_field])

            # Set as background field
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

            # Update global min to allow refined cells
            gmsh.option.setNumber("Mesh.MeshSizeMin", cell_size_object)

        # Generate the 3D mesh
        gmsh.model.mesh.generate(3)

        # Create output directory
        mesh_path = os.path.join(mesh_dir, "Saved Meshes", mesh_id)
        os.makedirs(mesh_path, exist_ok=True)

        # Save as MSH first
        msh_file = os.path.join(mesh_path, f"{mesh_id}.msh")
        gmsh.write(msh_file)

        # Convert to XDMF using meshio
        import meshio
        msh_mesh = meshio.read(msh_file)

        # Get all tetrahedra and their physical group markers
        # Use get_cells_type and cell_data_dict to properly combine all tetra blocks
        tetra_cells = msh_mesh.get_cells_type('tetra')
        if tetra_cells is None or len(tetra_cells) == 0:
            raise ValueError("No tetrahedra found in generated mesh")

        tetra_subdomain_data = msh_mesh.cell_data_dict.get('gmsh:physical', {}).get('tetra')
        if tetra_subdomain_data is None:
            raise ValueError("No physical group data found for tetrahedra")

        # Write mesh.xdmf with subdomain data
        meshio.write(
            os.path.join(mesh_path, "mesh.xdmf"),
            meshio.Mesh(
                points=msh_mesh.points,
                cells=[("tetra", tetra_cells)],
                cell_data={"Subdomain": [tetra_subdomain_data]},
                field_data=msh_mesh.field_data,
            )
        )

        # Write boundaries.xdmf with boundary surface data
        triangle_cells = msh_mesh.get_cells_type('triangle')
        triangle_boundary_data = msh_mesh.cell_data_dict.get('gmsh:physical', {}).get('triangle')

        if triangle_cells is not None and len(triangle_cells) > 0 and triangle_boundary_data is not None:
            meshio.write(
                os.path.join(mesh_path, "boundaries.xdmf"),
                meshio.Mesh(
                    points=msh_mesh.points,
                    cells=[("triangle", triangle_cells)],
                    cell_data={"Boundary": [triangle_boundary_data]},
                    field_data=msh_mesh.field_data,
                )
            )

        # Clean up MSH file
        os.remove(msh_file)

    finally:
        gmsh.finalize()

    # Return regions, bounds, and mesh_path
    regions = ["object", "vacuum"]
    bounds = {
        "object_bounds": {
            "x_min": float(xmin),
            "x_max": float(xmax),
            "y_min": float(ymin),
            "y_max": float(ymax),
            "z_min": float(zmin),
            "z_max": float(zmax),
        },
        "domain_radius": domain_radius,
        "r_min": 0.0,
        "r_max": domain_radius,
        "imported_file": os.path.basename(step_file),
        "char_size": char_size,
    }

    return regions, bounds, mesh_path


async def handle(args: dict[str, Any]) -> list[TextContent]:
    """Handle create_mesh tool call."""

    geometry = args["geometry"]
    params = args["params"]
    mesh_quality = args.get("mesh_quality", "medium")
    # Symmetry is fixed per geometry type
    symmetry = DEFAULT_SYMMETRY.get(geometry, "axial")
    warnings = []
    custom_id = args.get("custom_id")

    # Validate geometry is implemented
    implemented = [
        "sphere_in_vacuum", "ellipse_in_vacuum", "disk", "box_2d",
        "shell_in_vacuum", "cylinder_in_vacuum", "sphere_domain", "sphere_in_profile",
        "two_spheres", "sphere_near_wall", "box_3d", "parallel_plates",
        "custom_2d_axial", "custom_2d_translation", "custom_3d", "custom_step",
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
        "sphere_in_vacuum": ["object_radius", "domain_radius"],
        "ellipse_in_vacuum": ["rx", "ry", "domain_radius"],
        "disk": ["domain_radius"],
        "box_2d": ["domain_width", "domain_height"],
        "shell_in_vacuum": ["inner_radius", "outer_radius", "domain_radius"],
        "cylinder_in_vacuum": ["object_radius", "object_height", "domain_radius"],
        "sphere_domain": ["domain_radius"],
        "sphere_in_profile": ["object_radius", "domain_radius"],
        "two_spheres": ["radius_1", "radius_2", "separation", "domain_radius"],
        "sphere_near_wall": ["object_radius", "wall_distance", "domain_radius"],
        "box_3d": ["domain_width", "domain_height", "domain_depth"],
        "parallel_plates": ["plate_separation", "plate_thickness"],
        "custom_2d_axial": ["domain_radius"],
        "custom_2d_translation": ["domain_radius"],
        "custom_step": ["step_file", "domain_radius"],
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
    if geometry in ("custom_2d_axial", "custom_2d_translation"):
        if "shape_file" not in params and "points" not in params:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "MISSING_PARAMS",
                    "message": f"{geometry} requires either 'shape_file' or 'points' parameter",
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

    # Validate parameter values
    validation_errors = validate_geometry_params(geometry, params)
    if validation_errors:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "INVALID_PARAMS",
                "message": "Invalid geometry parameters",
                "details": validation_errors,
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
    if geometry in ["ellipsoid_in_vacuum", "box_3d", "custom_3d", "custom_step"]:
        dimension = 3
    else:
        dimension = 2

    # Get quality settings - use dedicated 3D settings for 3D meshes
    if dimension == 3:
        quality = MESH_QUALITY_SETTINGS_3D[mesh_quality].copy()
    else:
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
            subdomain_size = min(params.get("object_radius", 0), params.get("height", 0) / 2)
        elif geometry == "two_spheres":
            subdomain_size = min(params.get("radius_1", 0), params.get("radius_2", 0))
        elif geometry == "sphere_near_wall":
            subdomain_size = params.get("object_radius")
        elif geometry == "sphere_in_profile":
            subdomain_size = params.get("object_radius")
        elif geometry == "parallel_plates":
            subdomain_size = params.get("plate_thickness")
        elif geometry in ("custom_2d_axial", "custom_2d_translation"):
            if "points" in params:
                import numpy as np
                pts = np.array(params["points"])
                subdomain_size = float(np.max(pts) - np.min(pts)) / 2
            else:
                subdomain_size = None
        elif geometry in ("custom_step", "custom_3d"):
            # For STEP/custom 3D, we don't know size until import
            # Pass lambda info directly to mesh function which uses char_size
            subdomain_size = None
            physics_info = {
                "lambda_per_region": lambda_per_region,
                "lambda_min": min_lambda,
                "lambda_min_region": min_lambda_region,
                "refinement_applied": True,
            }
            if physics_params.get("alpha") is not None:
                physics_info["computed_from"] = {
                    "alpha": physics_params["alpha"],
                    "density": {k: extract_density_value(v, symmetry, geometry) for k, v in physics_params.get("density", {}).items()},
                    "n": physics_params.get("n", 1),
                }

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

    # custom_step bypasses MeshingTools entirely (direct gmsh for STEP import)
    if geometry == "custom_step":
        try:
            regions, bounds, mesh_path = _create_custom_step(params, quality, mesh_dir, mesh_id, physics_info)
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "MESH_GENERATION_FAILED",
                    "message": str(e),
                }
            }, indent=2))]
    else:
        # Use MeshingTools for all other geometries
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
            elif geometry == "custom_2d_axial":
                regions, bounds = _create_custom_2d(MT, params, quality, symmetry="axial")
            elif geometry == "custom_2d_translation":
                regions, bounds = _create_custom_2d(MT, params, quality, symmetry="translation")
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

        except Exception as e:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "MESH_GENERATION_FAILED",
                    "message": str(e),
                }
            }, indent=2))]

    # Get mesh statistics by reading the generated mesh (for both custom_step and other geometries)
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
    # Most geometry functions return a list (markers assigned as 0, 1, 2, ...)
    # custom_step returns a dict with actual physical group tags from gmsh
    if isinstance(regions, dict):
        regions_dict = regions
    else:
        # SELCIE assigns markers in the order subdomains are created (0, 1, 2, ...)
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
