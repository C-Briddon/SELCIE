#!/usr/bin/env python3
"""
End-to-end example: Custom shape with measuring boundary for fifth force evaluation.

This script demonstrates the measuring_boundary feature using custom_2d_axial:
1. Create mesh with custom_2d_axial geometry and measuring_distance
2. Solve the chameleon field equation
3. Evaluate the maximum gradient at the measuring boundary
4. Compare with max_in_region mode for validation

The custom shape is defined by explicit boundary points, allowing for
arbitrary axisymmetric geometries (ellipses, capsules, complex shapes).

Setup:
- Custom ellipse-like shape (smooth boundary)
- Vacuum chamber radius: 1.0
- Wall thickness: 0.05
- Measuring distance: 0.05 from object surface
- alpha = 1e18, n = 1
- Source density: 1e17
- Vacuum density: 1.0

Usage:
    python measuring_boundary_example.py [--quality QUALITY] [--shape SHAPE]
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import numpy as np

from utils.session import reset_session

# Physical parameters
DOMAIN_RADIUS = 1.0
WALL_THICKNESS = 0.05
MEASURING_DISTANCE = 0.05

SOURCE_DENSITY = 1e17
VACUUM_DENSITY = 1.0
WALL_DENSITY = 1e17  # Dense wall
ALPHA = 1e18
N = 1


def generate_ellipse_points(rx: float, rz: float, n_points: int = 50) -> list:
    """Generate points for an ellipse (half-plane for axisymmetric).

    Args:
        rx: Semi-axis in radial direction
        rz: Semi-axis in z direction
        n_points: Number of boundary points

    Returns:
        List of [r, z] points defining the boundary
    """
    theta = np.linspace(0, np.pi, n_points, endpoint=True)
    r = rx * np.sin(theta)  # radial distance from axis
    z = rz * np.cos(theta)  # height along axis
    return [[float(ri), float(zi)] for ri, zi in zip(r, z)]


def generate_capsule_points(radius: float, height: float, n_points: int = 50) -> list:
    """Generate points for a capsule (cylinder with hemispherical caps).

    Args:
        radius: Radius of the cylinder and hemisphere caps
        height: Height of the cylindrical section (total height = height + 2*radius)
        n_points: Number of boundary points

    Returns:
        List of [r, z] points defining the boundary
    """
    points = []
    half_h = height / 2

    # Top hemisphere (from top to where it meets the cylinder)
    n_cap = n_points // 3
    theta = np.linspace(0, np.pi / 2, n_cap, endpoint=True)
    for t in theta:
        r = radius * np.sin(t)
        z = half_h + radius * np.cos(t)
        points.append([float(r), float(z)])

    # Cylindrical section (right edge, excluding endpoints which are in hemispheres)
    n_cyl = n_points // 3
    for i in range(1, n_cyl - 1):  # Skip first and last to avoid duplicates
        z = half_h - i / (n_cyl - 1) * height
        points.append([float(radius), float(z)])

    # Bottom hemisphere (from where it meets cylinder to bottom)
    n_cap = n_points - len(points)
    theta = np.linspace(np.pi / 2, np.pi, n_cap, endpoint=True)
    for t in theta:
        r = radius * np.sin(t)
        z = -half_h + radius * np.cos(t)
        points.append([float(r), float(z)])

    return points


def generate_teardrop_points(radius: float, length: float, n_points: int = 50) -> list:
    """Generate points for a teardrop shape (sphere with pointed tail).

    Args:
        radius: Maximum radius of the teardrop
        length: Total length from tip to base
        n_points: Number of boundary points

    Returns:
        List of [r, z] points defining the boundary
    """
    points = []

    # Parametric curve: smooth teardrop
    t = np.linspace(0, np.pi, n_points, endpoint=True)

    for ti in t:
        # Teardrop parameterization
        r = radius * np.sin(ti) * (1 - 0.3 * np.cos(ti))
        z = length / 2 * np.cos(ti)
        points.append([float(max(0, r)), float(z)])

    return points


SHAPES = {
    "ellipse": {
        "func": generate_ellipse_points,
        "params": {"rx": 0.15, "rz": 0.1},
        "description": "Ellipse (rx=0.15, rz=0.1)",
    },
    "capsule": {
        "func": generate_capsule_points,
        "params": {"radius": 0.08, "height": 0.15},
        "description": "Capsule (r=0.08, h=0.15)",
    },
    "teardrop": {
        "func": generate_teardrop_points,
        "params": {"radius": 0.12, "length": 0.3},
        "description": "Teardrop (r=0.12, L=0.3)",
    },
    "sphere": {
        "func": generate_ellipse_points,
        "params": {"rx": 0.1337, "rz": 0.1337},
        "description": "Sphere (r=0.1337)",
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SELCIE simulation with measuring boundary using custom_2d_axial."
    )
    parser.add_argument(
        "--quality", "-q",
        choices=["very_coarse", "coarse", "medium", "fine", "very_fine"],
        default="fine",
        help="Mesh quality (default: fine)",
    )
    parser.add_argument(
        "--shape", "-s",
        choices=list(SHAPES.keys()),
        default="ellipse",
        help=f"Shape type (default: ellipse). Options: {', '.join(SHAPES.keys())}",
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=ALPHA,
        help=f"Alpha parameter (default: {ALPHA:.0e})",
    )
    parser.add_argument(
        "--source-density", "-d",
        type=float,
        default=SOURCE_DENSITY,
        help=f"Source density (default: {SOURCE_DENSITY:.0e})",
    )
    parser.add_argument(
        "--measuring-distance", "-m",
        type=float,
        default=MEASURING_DISTANCE,
        help=f"Measuring distance from object surface (default: {MEASURING_DISTANCE})",
    )
    return parser.parse_args()


async def main():
    """Run the measuring boundary example."""
    args = parse_args()

    # Import tool handlers
    from tools.create_mesh import handle as create_mesh
    from tools.solve import handle as solve
    from tools.evaluate import handle as evaluate
    from tools.plot import handle as plot

    # Reset session for fresh state
    reset_session()

    output_dir = Path(__file__).parent.parent / "test_plots"
    output_dir.mkdir(exist_ok=True)

    mesh_id = "custom_measuring"
    solution_id = "custom_measuring_solution"

    # Generate shape points
    shape_config = SHAPES[args.shape]
    points = shape_config["func"](n_points=50, **shape_config["params"])

    # Calculate approximate object size for display
    points_arr = np.array(points)
    max_r = np.max(points_arr[:, 0])
    max_z = np.max(np.abs(points_arr[:, 1]))

    print("=" * 70)
    print("SELCIE MCP Tools: Custom Shape with Measuring Boundary Example")
    print("=" * 70)
    print(f"\nSetup:")
    print(f"  Shape: {shape_config['description']}")
    print(f"  Max radius (r): {max_r:.4f}")
    print(f"  Max extent (z): {max_z:.4f}")
    print(f"  Domain radius: {DOMAIN_RADIUS}")
    print(f"  Wall thickness: {WALL_THICKNESS}")
    print(f"  Measuring distance: {args.measuring_distance}")
    print(f"  alpha = {args.alpha:.0e}")
    print(f"  n = {N}")
    print(f"  Source density: {args.source_density:.0e}")
    print(f"  Vacuum density: {VACUUM_DENSITY:.0e}")
    print(f"  Wall density: {WALL_DENSITY:.0e}")

    # =========================================================================
    # Step 1: Create mesh with measuring boundary using custom_2d_axial
    # =========================================================================
    print("\n" + "-" * 70)
    print("[1/4] Creating mesh with measuring boundary (custom_2d_axial)...")
    print("-" * 70)

    mesh_result = await create_mesh({
        "geometry": "custom_2d_axial",
        "params": {
            "points": points,
            "domain_radius": DOMAIN_RADIUS,
            "wall_thickness": WALL_THICKNESS,
            "measuring_distance": args.measuring_distance,
        },
        "mesh_quality": args.quality,
        "custom_id": mesh_id,
        "physics_params": {
            "alpha": args.alpha,
            "density": {"object": args.source_density, "vacuum": VACUUM_DENSITY},
            "n": N,
        },
    })

    mesh_data = json.loads(mesh_result[0].text)

    if "error" in mesh_data:
        print(f"ERROR: {mesh_data['error']}")
        return

    print(f"  Mesh ID: {mesh_data['mesh_id']}")
    print(f"  Geometry: {mesh_data['geometry']}")
    print(f"  Cells: {mesh_data['n_cells']:,}")
    print(f"  Vertices: {mesh_data['n_vertices']:,}")
    print(f"  Regions: {mesh_data['regions']}")
    print(f"  Path: {mesh_data['mesh_path']}")

    if "physics_refinement" in mesh_data:
        pr = mesh_data["physics_refinement"]
        print(f"  Physics refinement:")
        print(f"    lambda_min = {pr.get('lambda_min', 'N/A'):.3e}")
        print(f"    cell_min_limited: {pr.get('cell_min_limited', False)}")

    # Verify measuring_boundary region exists
    if "measuring_boundary" not in mesh_data["regions"]:
        print("ERROR: measuring_boundary region not found in mesh!")
        return

    print(f"\n  measuring_boundary region created (marker {mesh_data['regions']['measuring_boundary']})")

    # =========================================================================
    # Step 2: Solve chameleon field equation
    # =========================================================================
    print("\n" + "-" * 70)
    print("[2/4] Solving chameleon field equation...")
    print("-" * 70)

    # Note: measuring_boundary density is auto-assigned from vacuum
    solve_result = await solve({
        "mesh_id": mesh_id,
        "alpha": args.alpha,
        "n": N,
        "density": {
            "object": args.source_density,
            "vacuum": VACUUM_DENSITY,
            "wall": WALL_DENSITY,
            # measuring_boundary: auto-assigned from vacuum
        },
        "custom_id": solution_id,
        "max_iter": 200,
    })

    solve_data = json.loads(solve_result[0].text)

    if "error" in solve_data or solve_data.get("status") == "failed":
        print(f"ERROR: {solve_data}")
        return

    print(f"  Solution ID: {solve_data['solution_id']}")
    print(f"  Status: {solve_data['status']}")
    print(f"  Iterations: {solve_data['iterations']}")
    print(f"  Final residual: {solve_data['final_du_norm']:.2e}")

    field_stats = solve_data.get('field_stats', {})
    print(f"  Field range: [{field_stats.get('min', 'N/A'):.6e}, {field_stats.get('max', 'N/A'):.6e}]")

    grad_stats = solve_data.get('gradient_stats', {})
    if grad_stats.get('computed'):
        print(f"  Gradient magnitude range: [{grad_stats.get('magnitude_min', 'N/A'):.6e}, {grad_stats.get('magnitude_max', 'N/A'):.6e}]")

    # =========================================================================
    # Step 3: Evaluate gradient at measuring boundary
    # =========================================================================
    print("\n" + "-" * 70)
    print("[3/4] Evaluating gradient at measuring boundary...")
    print("-" * 70)

    # Method 1: boundary_max mode
    print("\n  Method 1: boundary_max mode on measuring_boundary")
    boundary_max_result = await evaluate({
        "solution_id": solution_id,
        "mode": "boundary_max",
        "params": {
            "region": "measuring_boundary",
        },
    })

    boundary_max_data = json.loads(boundary_max_result[0].text)

    if "error" in boundary_max_data:
        print(f"    ERROR: {boundary_max_data['error']}")
    else:
        data = boundary_max_data["data"]
        print(f"    Max gradient: {data['max_gradient']:.6e}")
        print(f"    Max position: {data['max_position']}")
        print(f"    Mean gradient: {data['mean_gradient']:.6e}")
        print(f"    Min gradient: {data['min_gradient']:.6e}")
        print(f"    N points evaluated: {data['n_points']}")

    # Method 2: max_in_region mode for comparison
    print("\n  Method 2: max_in_region mode on vacuum (with min_distance)")
    max_in_region_result = await evaluate({
        "solution_id": solution_id,
        "mode": "max_in_region",
        "params": {
            "region": "vacuum",
            "min_distance_from": "object",
            "min_distance": args.measuring_distance - 0.01,  # Slightly less to capture boundary
            "n_samples": 5000,
        },
        "quantities": ["gradient_magnitude"],
    })

    max_in_region_data = json.loads(max_in_region_result[0].text)

    if "error" in max_in_region_data:
        print(f"    ERROR: {max_in_region_data['error']}")
    else:
        grad_data = max_in_region_data["data"].get("gradient_magnitude", {})
        print(f"    Max gradient: {grad_data.get('max', 'N/A'):.6e}")
        print(f"    Max position: {grad_data.get('max_location', 'N/A')}")
        print(f"    Mean gradient: {grad_data.get('mean', 'N/A'):.6e}")
        print(f"    N samples: {max_in_region_data.get('n_valid_samples', 'N/A')}")

    # Method 3: Radial profile through the measuring boundary
    print("\n  Method 3: Radial profile through measuring boundary")
    r_inner = max_r * 0.5
    r_outer = max_r + args.measuring_distance + 0.1
    radial_result = await evaluate({
        "solution_id": solution_id,
        "mode": "radial",
        "params": {
            "r_min": r_inner,
            "r_max": r_outer,
            "n_points": 100,
            "log_spacing": False,
            "direction": [1, 0],  # Along r-axis (z=0)
        },
        "quantities": ["field", "gradient_magnitude"],
    })

    radial_data = json.loads(radial_result[0].text)

    if "error" in radial_data:
        print(f"    ERROR: {radial_data['error']}")
    else:
        r_values = np.array(radial_data["data"]["r"])
        grad_values = np.array(radial_data["data"]["gradient_magnitude"])

        # Find max gradient in radial profile
        max_idx = np.argmax(grad_values)
        print(f"    Max gradient in radial profile: {grad_values[max_idx]:.6e}")
        print(f"    At radius: {r_values[max_idx]:.4f}")
        print(f"    (Expected measuring boundary at r ~ {max_r + args.measuring_distance:.4f})")

    # =========================================================================
    # Step 4: Generate plots
    # =========================================================================
    print("\n" + "-" * 70)
    print("[4/4] Generating plots...")
    print("-" * 70)

    # Field 1D profile
    field_1d_path = output_dir / f"{args.shape}_measuring_field_1d.png"
    await plot({
        "solution_id": solution_id,
        "plot_type": "field_1d",
        "output_path": str(field_1d_path),
        "options": {
            "title": f"{args.shape.title()} with Measuring Boundary: Field Profile",
            "n_points": 200,
        },
    })
    print(f"  Field 1D saved to: {field_1d_path}")

    # Gradient 1D profile
    grad_1d_path = output_dir / f"{args.shape}_measuring_gradient_1d.png"
    await plot({
        "solution_id": solution_id,
        "plot_type": "gradient_1d",
        "output_path": str(grad_1d_path),
        "options": {
            "title": f"{args.shape.title()} with Measuring Boundary: Gradient Profile",
            "n_points": 200,
        },
    })
    print(f"  Gradient 1D saved to: {grad_1d_path}")

    # Field 2D
    field_2d_path = output_dir / f"{args.shape}_measuring_field_2d.png"
    await plot({
        "solution_id": solution_id,
        "plot_type": "field_2d",
        "output_path": str(field_2d_path),
        "options": {
            "title": f"{args.shape.title()} with Measuring Boundary: Field (2D)",
            "n_grid": 150,
        },
    })
    print(f"  Field 2D saved to: {field_2d_path}")

    # Gradient 2D
    grad_2d_path = output_dir / f"{args.shape}_measuring_gradient_2d.png"
    await plot({
        "solution_id": solution_id,
        "plot_type": "gradient_2d",
        "output_path": str(grad_2d_path),
        "options": {
            "title": f"{args.shape.title()} with Measuring Boundary: |grad phi| (2D)",
            "n_grid": 150,
        },
    })
    print(f"  Gradient 2D saved to: {grad_2d_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if "error" not in boundary_max_data:
        data = boundary_max_data["data"]
        print(f"\nMaximum gradient at measuring boundary:")
        print(f"  |grad phi|_max = {data['max_gradient']:.6e}")
        print(f"  Position: r = {data['max_position'][0]:.4f}, z = {data['max_position'][1]:.4f}")

        # Calculate distance from axis
        r_measured = (data['max_position'][0]**2 + data['max_position'][1]**2)**0.5
        print(f"\n  Distance from origin: {r_measured:.4f}")

    print(f"\nOutput files saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
