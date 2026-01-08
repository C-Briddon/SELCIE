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


# Mesh quality settings: (CellSizeMin, CellSizeMax, DistMax) relative to object size
MESH_QUALITY_SETTINGS = {
    "very_coarse": {"cell_min_factor": 0.5, "cell_max_factor": 0.8, "dist_max_factor": 2.0},
    "coarse": {"cell_min_factor": 0.15, "cell_max_factor": 0.4, "dist_max_factor": 0.8},
    "medium": {"cell_min_factor": 0.05, "cell_max_factor": 0.15, "dist_max_factor": 0.5},
    "fine": {"cell_min_factor": 0.02, "cell_max_factor": 0.08, "dist_max_factor": 0.3},
    "very_fine": {"cell_min_factor": 0.005, "cell_max_factor": 0.03, "dist_max_factor": 0.2},
}

# Default symmetry for each geometry
DEFAULT_SYMMETRY = {
    "sphere_in_vacuum": "axial",
    "ellipse_in_vacuum": "axial",
    "ellipsoid_in_vacuum": "none",
    "cylinder_in_vacuum": "axial",
    "shell_in_vacuum": "axial",
    "two_spheres": "axial",
    "sphere_near_wall": "axial",
    "box_2d": "none",
    "box_3d": "none",
    "disk": "axial",
    "sphere_domain": "axial",
    "custom_2d": "axial",
    "custom_3d": "none",
}

# Refinement limits to prevent excessive cell counts
REFINEMENT_LIMITS = {
    "min_cell_size_factor": 0.002,  # Minimum cell size as fraction of object size
    "max_refinement_ratio": 50,      # Max ratio of largest to smallest cells
    "target_boundary_cells": 5,      # Target cells across thin shell
}


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


def _extract_density_value(density_spec) -> float | None:
    """Extract a representative density value from various formats.

    For mesh refinement, we need a single number to compute λ.
    For non-numeric values, we extract/estimate the maximum density.

    Args:
        density_spec: Can be:
            - number: used directly
            - {"expression": str}: evaluate at sample points, take max
            - {"file": str, "skip_header": int}: load profile, take max

    Returns:
        Representative density value, or None if cannot extract
    """
    import numpy as np

    if isinstance(density_spec, (int, float)):
        return float(density_spec)

    if isinstance(density_spec, dict):
        if "expression" in density_spec:
            # Evaluate expression at sample points and take max
            # Use a simple grid to estimate max density
            expr = density_spec["expression"]
            try:
                # Sample points in a reasonable range
                r_vals = np.linspace(0.001, 1.0, 20)
                z_vals = np.linspace(-1.0, 1.0, 20)
                max_rho = 0.0

                for r in r_vals:
                    for z in z_vals:
                        x, y = r, z  # For Cartesian expressions
                        try:
                            val = eval(expr)
                            if isinstance(val, (int, float)) and val > max_rho:
                                max_rho = val
                        except Exception:
                            pass

                return max_rho if max_rho > 0 else None
            except Exception:
                return None

        elif "file" in density_spec:
            # Load profile from file and take max
            try:
                skip_header = density_spec.get("skip_header", 0)
                data = np.loadtxt(density_spec["file"], skiprows=skip_header)
                # Assume density is in second column (first is position)
                if data.ndim == 1:
                    return float(np.max(data))
                else:
                    return float(np.max(data[:, 1]))
            except Exception:
                return None

    return None


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
        "physics_params with alpha and density_contrast to enable automatic mesh refinement "
        "near object boundaries. This ensures the thin shell region is properly resolved."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "geometry": {
                "type": "string",
                "enum": [
                    "sphere_in_vacuum", "ellipse_in_vacuum", "ellipsoid_in_vacuum",
                    "cylinder_in_vacuum", "shell_in_vacuum", "two_spheres", "sphere_near_wall",
                    "box_2d", "box_3d", "disk", "sphere_domain",
                    "custom_2d", "custom_3d"
                ],
                "description": "Geometry template.",
            },
            "params": {
                "type": "object",
                "description": "Geometry-specific parameters.",
                "properties": {
                    "object_radius": {"type": "number", "description": "Radius of spherical source"},
                    "vacuum_radius": {"type": "number", "description": "Outer radius of vacuum region"},
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
                    "contours_file": {"type": "string", "description": "Path to 3D contours file (custom_3d)"},
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
                "enum": ["axial", "none"],
                "description": "Override default symmetry.",
            },
            "custom_id": {
                "type": "string",
                "description": "Custom mesh ID.",
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
                        "description": "Dimensionless density ρ̂ = ρ/ρ₀ per region (ρ₀ is the reference density used to compute α). Each key is a region name, value is: number, {expression: str}, or {file: str, skip_header?: int}. For non-numeric values, max density is used to compute λ.",
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


def _create_sphere_in_vacuum(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create sphere in vacuum geometry."""
    object_radius = params["object_radius"]
    vacuum_radius = params["vacuum_radius"]
    wall_thickness = params.get("wall_thickness")

    # Create the sphere
    MT.create_ellipse(rx=object_radius, ry=object_radius)

    # Mark as subdomain with refinement
    cell_min = quality["cell_min_factor"] * object_radius
    cell_max = quality["cell_max_factor"] * object_radius
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

    # Create the ellipse
    MT.create_ellipse(rx=rx, ry=ry)

    # Mark as subdomain with refinement
    char_size = max(rx, ry)
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
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

    # For plain domains, use radius/5 as base size for reasonable resolution
    char_size = radius / 5
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
    dist_max = quality["dist_max_factor"] * radius
    MT.create_background_mesh(
        CellSizeMin=cell_min,
        CellSizeMax=cell_max,
        DistMax=dist_max,
        background_radius=radius,
        wall_thickness=None,
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

    # For plain domains, use min dimension / 5 for reasonable resolution
    char_size = min(width, height) / 5
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


def _create_shell_in_vacuum(
    MT: MeshingTools,
    params: dict,
    quality: dict,
) -> tuple[list[str], dict]:
    """Create hollow shell in vacuum geometry."""
    inner_radius = params["inner_radius"]
    outer_radius = params["outer_radius"]
    vacuum_radius = params["vacuum_radius"]

    # Create outer sphere and inner sphere, then subtract
    outer = MT.create_ellipse(rx=outer_radius, ry=outer_radius)
    inner = MT.create_ellipse(rx=inner_radius, ry=inner_radius)
    shell = MT.subtract_shapes(outer, inner)

    # Mark as subdomain with refinement
    char_size = outer_radius - inner_radius
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
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
    char_size = min(radius, height)
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
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

    # For plain domains, use radius/5 as base size for reasonable resolution
    char_size = radius / 5
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
    dist_max = quality["dist_max_factor"] * radius
    MT.create_background_mesh(
        CellSizeMin=cell_min,
        CellSizeMax=cell_max,
        DistMax=dist_max,
        background_radius=radius,
        wall_thickness=None,
    )

    regions = ["domain"]
    bounds = {
        "r_min": 0.0,
        "r_max": radius,
        "z_min": -radius,
        "z_max": radius,
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

    # Create first sphere (source) at origin
    s1 = MT.create_ellipse(rx=radius_1, ry=radius_1)
    cell_min = quality["cell_min_factor"] * radius_1
    cell_max = quality["cell_max_factor"] * radius_1
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create second sphere (test mass) offset along z
    s2 = MT.create_ellipse(rx=radius_2, ry=radius_2)
    MT.translate_y(s2, dy=separation)  # Move along z-axis
    cell_min2 = quality["cell_min_factor"] * radius_2
    cell_max2 = quality["cell_max_factor"] * radius_2
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
    wall_cell_min = quality["cell_min_factor"] * wall_thickness
    wall_cell_max = quality["cell_max_factor"] * wall_thickness
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=wall_cell_min, CellSizeMax=wall_cell_max, DistMax=dist_max)

    # Create sphere offset above wall surface (z=0)
    sphere = MT.create_ellipse(rx=object_radius, ry=object_radius)
    MT.translate_y(sphere, dy=wall_distance)  # Sphere center at z=wall_distance

    # Mark sphere as subdomain with refinement
    cell_min = quality["cell_min_factor"] * object_radius
    cell_max = quality["cell_max_factor"] * object_radius
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

    # For 3D plain domains, use min dimension / 3 (3D scales cubically!)
    char_size = min(width, height, depth) / 3
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
) -> tuple[list[str], dict]:
    """Create custom 2D geometry from file or points.

    Supports:
    - shape_file: Path to text file with x,y coordinates per line
    - points: Direct list of [x, y] coordinates
    """
    import numpy as np

    vacuum_radius = params.get("vacuum_radius", 1.0)

    # Load points from file or use direct points
    if "shape_file" in params:
        points_2d = np.loadtxt(params["shape_file"])
    elif "points" in params:
        points_2d = np.array(params["points"])
    else:
        raise ValueError("Must provide either 'shape_file' or 'points'")

    # Convert to 3D coordinates (z=0) for SELCIE
    if points_2d.shape[1] == 2:
        points_3d = [[p[0], p[1], 0.0] for p in points_2d]
    else:
        points_3d = points_2d.tolist()

    # Create shape from points
    MT.points_to_surface(points_3d)

    # Mark as subdomain with refinement
    char_size = float(np.max(points_2d) - np.min(points_2d))
    cell_min = quality["cell_min_factor"] * char_size
    cell_max = quality["cell_max_factor"] * char_size
    dist_max = quality["dist_max_factor"] * vacuum_radius
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create background (vacuum) if vacuum_radius provided
    if vacuum_radius:
        bg_cell_min = quality["cell_min_factor"] * vacuum_radius
        bg_cell_max = quality["cell_max_factor"] * vacuum_radius
        MT.create_background_mesh(
            CellSizeMin=bg_cell_min,
            CellSizeMax=bg_cell_max,
            DistMax=dist_max,
            background_radius=vacuum_radius,
            wall_thickness=0.1 * vacuum_radius,
        )

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
    symmetry = args.get("symmetry") or DEFAULT_SYMMETRY.get(geometry, "axial")
    custom_id = args.get("custom_id")

    # Validate geometry is implemented
    implemented = [
        "sphere_in_vacuum", "ellipse_in_vacuum", "disk", "box_2d",
        "shell_in_vacuum", "cylinder_in_vacuum", "sphere_domain", "two_spheres",
        "sphere_near_wall", "box_3d", "custom_2d", "custom_3d",
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
        "two_spheres": ["radius_1", "radius_2", "separation", "vacuum_radius"],
        "sphere_near_wall": ["object_radius", "wall_distance", "vacuum_radius"],
        "box_3d": ["width", "height", "depth"],
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
                rho = _extract_density_value(density_spec)
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
                        "density": {k: _extract_density_value(v) for k, v in physics_params.get("density", {}).items()},
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
        elif geometry == "two_spheres":
            regions, bounds = _create_two_spheres(MT, params, quality)
        elif geometry == "sphere_near_wall":
            regions, bounds = _create_sphere_near_wall(MT, params, quality)
        elif geometry == "box_3d":
            regions, bounds = _create_box_3d(MT, params, quality)
        elif geometry == "custom_2d":
            regions, bounds = _create_custom_2d(MT, params, quality)
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
        result["warning"] = f"Very large mesh ({n_cells:,} cells). Solving may take a long time and use significant memory. Consider using a coarser mesh quality."
    elif n_cells > 50000:
        result["warning"] = f"Large mesh ({n_cells:,} cells). Solving may be slow. Consider using a coarser mesh quality if performance is an issue."

    return [TextContent(type="text", text=json.dumps(result, indent=2))]
