#!/usr/bin/env python3
"""
End-to-end example: STEP file simulation using SELCIE MCP tools.

This script demonstrates the full workflow:
1. Create mesh from STEP file
2. Plot the mesh
3. Solve the chameleon field equation
4. Plot the field profile

Usage:
    python step_example.py [step_file] [--quality QUALITY] [--domain-radius R] [--physics]

Parameters:
- alpha = 1e18
- n = 1
- Source density: 1e17
- Vacuum density: 1.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json

from utils.session import reset_session


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SELCIE simulation on a STEP file geometry."
    )
    parser.add_argument(
        "step_file",
        nargs="?",
        default=None,
        help="Path to STEP file (default: tests/test_data/eotwash_disks.step)",
    )
    parser.add_argument(
        "--quality", "-q",
        choices=["very_coarse", "coarse", "medium", "fine", "very_fine"],
        default="medium",
        help="Mesh quality (default: medium)",
    )
    parser.add_argument(
        "--domain-radius", "-r",
        type=float,
        default=None,
        help="Domain radius (default: auto-calculated from geometry)",
    )
    parser.add_argument(
        "--physics", "-p",
        action="store_true",
        help="Enable physics-aware mesh refinement using solve parameters (alpha, density)",
    )
    parser.add_argument(
        "--use-single-object", "-s",
        action="store_true",
        help="Use single object mode (default: False)",
    )
    return parser.parse_args()


async def main():
    """Run the end-to-end example."""
    args = parse_args()

    # Import tool handlers
    from tools.create_mesh import handle as create_mesh
    from tools.plot_mesh import handle as plot_mesh
    from tools.plot_step import handle as plot_step
    from tools.solve import handle as solve
    from tools.plot import handle as plot

    # Reset session for fresh state
    reset_session()

    # Paths
    test_data_dir = Path(__file__).parent.parent / "tests" / "test_data"
    if args.step_file:
        step_file = Path(args.step_file)
    else:
        step_file = test_data_dir / "eotwash_disks.step"

    if not step_file.exists():
        print(f"ERROR: STEP file not found: {step_file}")
        sys.exit(1)

    output_dir = Path(__file__).parent.parent / "step_plots"
    output_dir.mkdir(exist_ok=True)

    # Derive IDs and output names from filename
    base_name = step_file.stem
    mesh_id = base_name + "_mesh"
    solution_id = base_name + "_solution"

    print("=" * 60)
    print("SELCIE MCP Tools: STEP File Example")
    print("=" * 60)
    print(f"STEP file: {step_file}")
    print(f"Mesh quality: {args.quality}")

    # =========================================================================
    # Step 0: Preview STEP geometry (before meshing)
    # =========================================================================
    print("\n[0/5] Previewing STEP geometry...")

    preview_path = output_dir / f"{base_name}_preview.png"
    preview_result = await plot_step({
        "step_file": str(step_file),
        "output_path": str(preview_path),
        "title": f"{base_name} Geometry Preview",
    })

    preview_data = json.loads(preview_result[0].text)

    if "error" in preview_data:
        print(f"  Warning: {preview_data['error']['message']}")
    else:
        print(f"  Volumes: {preview_data['n_volumes']}")
        print(f"  Extent: x={preview_data['extent']['x']:.3f}, y={preview_data['extent']['y']:.3f}, z={preview_data['extent']['z']:.3f}")
        print(f"  Region names (after meshing): {preview_data['region_names']}")
        print(f"  Preview saved to: {preview_path}")

    # =========================================================================
    # Step 1: Create mesh from STEP file
    # =========================================================================
    print("\n[1/5] Creating mesh from STEP file...")

    # Build params - domain_radius is optional (tool auto-calculates as 1.5x extent if not provided)
    mesh_params = {"step_file": str(step_file)}
    if args.domain_radius is not None:
        mesh_params["domain_radius"] = args.domain_radius
        print(f"  Using domain_radius: {args.domain_radius}")
    else:
        print("  Domain radius: auto (1.5x geometry extent)")

    # Build create_mesh arguments
    create_mesh_args = {
        "geometry": "custom_step",
        "params": mesh_params,
        "mesh_quality": args.quality,
        "custom_id": mesh_id,
        "allow_large_mesh": True,
    }

    # Add physics_params for mesh refinement if --physics flag is set
    # Note: For multi-region STEP files, physics refinement uses max density across all objects
    if args.physics:
        create_mesh_args["physics_params"] = {
            "alpha": 1e18,
            "density": {"object": 1e17, "vacuum": 1.0},  # Uses max object density for refinement
            "n": 1,
        }
        print("  Physics-aware refinement: enabled (uses max object density)")

    mesh_result = await create_mesh(create_mesh_args)

    mesh_data = json.loads(mesh_result[0].text)

    if "error" in mesh_data:
        print(f"ERROR: {mesh_data['error']['message']}")
        return

    print(f"  Mesh ID: {mesh_data['mesh_id']}")
    print(f"  Cells: {mesh_data['n_cells']:,}")
    print(f"  Vertices: {mesh_data['n_vertices']:,}")
    print(f"  Regions: {mesh_data['regions']}")
    print(f"  Path: {mesh_data['mesh_path']}")

    # =========================================================================
    # Step 2: Plot mesh
    # =========================================================================
    print("\n[2/5] Plotting mesh...")

    mesh_plot_path = output_dir / f"{base_name}_mesh.png"
    mesh_plot_result = await plot_mesh({
        "mesh_id": mesh_id,
        "output_path": str(mesh_plot_path),
        "title": f"{base_name} Mesh",
        "clip": {
            "normal": [0, 1, 0],  # Clip in y-direction to see inside
            "origin": [0, 0, 0],
        },
    })

    # Check if plot succeeded
    plot_response = mesh_plot_result[0]
    if hasattr(plot_response, 'text'):
        data = json.loads(plot_response.text)
        if "error" in data:
            print(f"  Warning: {data['error']['message']}")
        else:
            print(f"  Saved to: {mesh_plot_path}")
    else:
        print(f"  Saved to: {mesh_plot_path}")

    # =========================================================================
    # Step 3: Solve chameleon field equation
    # =========================================================================
    print("\n[3/5] Solving chameleon field equation...")

    # Build density dict based on regions in mesh
    # For multi-object STEP files, regions are object_0, object_1, ... (sorted by z-centroid)
    regions = mesh_data["regions"]
    object_regions = [r for r in regions.keys() if r.startswith("object")]

    if len(object_regions) == 1 or args.use_single_object:
        # Single object: use "object"
        density = {
            "object": 1e17,
            "vacuum": 1.0,
        }
    else:
        # Multiple objects: assign different densities to each
        # object_0 is lowest in z, object_1 next, etc.
        density = {"vacuum": 1.0}
        base_density = 1e17
        for i, region in enumerate(sorted(object_regions)):
            # Each successive object gets lower density (for demonstration)
            density[region] = base_density / (10 ** (i * 3))

    print("  Parameters:")
    print("    alpha = 1e18")
    print("    n = 1")
    for region, rho in density.items():
        print(f"    density({region}) = {rho:.0e}")

    solve_result = await solve({
        "mesh_id": mesh_id,
        "alpha": 1e18,
        "n": 1,
        "density": density,
        "custom_id": solution_id,
        "max_iter": 200,
        "display_progress": True,
    })

    solve_data = json.loads(solve_result[0].text)

    if "error" in solve_data:
        print(f"ERROR: {solve_data['error']}")
        return

    print(f"  Solution ID: {solve_data['solution_id']}")
    print(f"  Status: {solve_data['status']}")
    print(f"  Iterations: {solve_data['iterations']}")
    print(f"  Final residual: {solve_data['final_du_norm']:.2e}")
    field_stats = solve_data.get('field_stats', {})
    print(f"  Field range: [{field_stats.get('min', 'N/A'):.3e}, {field_stats.get('max', 'N/A'):.3e}]")

    # =========================================================================
    # Step 4: Plot field profile
    # =========================================================================
    print("\n[4/5] Plotting field profile...")

    # Plot 1D radial profile
    field_1d_path = output_dir / f"{base_name}_field_1d.png"
    field_1d_result = await plot({
        "solution_id": solution_id,
        "plot_type": "field_1d",
        "output_path": str(field_1d_path),
        "options": {
            "title": f"{base_name}: Field Profile",
            "n_points": 200,
        },
    })

    plot_response = field_1d_result[0]
    if hasattr(plot_response, 'text'):
        data = json.loads(plot_response.text)
        if "error" in data:
            print(f"  Warning (1D): {data['error']['message']}")
        else:
            print(f"  Field 1D saved to: {field_1d_path}")
    else:
        print(f"  Field 1D saved to: {field_1d_path}")

    # Plot 2D slice through xz-plane (y=0)
    slice_xz_path = output_dir / f"{base_name}_field_slice_xz.png"
    slice_xz_result = await plot({
        "solution_id": solution_id,
        "plot_type": "slice_xz",
        "output_path": str(slice_xz_path),
        "options": {
            "title": f"{base_name}: Field (xz-plane, y=0)",
            "slice_position": 0,
            "n_grid": 150,
        },
    })

    plot_response = slice_xz_result[0]
    if hasattr(plot_response, 'text'):
        data = json.loads(plot_response.text)
        if "error" in data:
            print(f"  Warning (slice_xz): {data['error']['message']}")
        else:
            print(f"  Field slice (xz) saved to: {slice_xz_path}")
    else:
        print(f"  Field slice (xz) saved to: {slice_xz_path}")

    # Plot 2D slice through xy-plane (z=0)
    slice_xy_path = output_dir / f"{base_name}_field_slice_xy.png"
    slice_xy_result = await plot({
        "solution_id": solution_id,
        "plot_type": "slice_xy",
        "output_path": str(slice_xy_path),
        "options": {
            "title": f"{base_name}: Field (xy-plane, z=0)",
            "slice_position": 0,
            "n_grid": 150,
        },
    })

    plot_response = slice_xy_result[0]
    if hasattr(plot_response, 'text'):
        data = json.loads(plot_response.text)
        if "error" in data:
            print(f"  Warning (slice_xy): {data['error']['message']}")
        else:
            print(f"  Field slice (xy) saved to: {slice_xy_path}")
    else:
        print(f"  Field slice (xy) saved to: {slice_xy_path}")

    # Plot gradient magnitude slice through xz-plane (y=0)
    grad_xz_path = output_dir / f"{base_name}_gradient_slice_xz.png"
    grad_xz_result = await plot({
        "solution_id": solution_id,
        "plot_type": "slice_xz",
        "output_path": str(grad_xz_path),
        "options": {
            "title": f"{base_name}: |∇φ| (xz-plane, y=0)",
            "slice_position": 0,
            "n_grid": 150,
            "quantity": "gradient_magnitude",
        },
    })

    plot_response = grad_xz_result[0]
    if hasattr(plot_response, 'text'):
        data = json.loads(plot_response.text)
        if "error" in data:
            print(f"  Warning (grad_xz): {data['error']['message']}")
        else:
            print(f"  Gradient slice (xz) saved to: {grad_xz_path}")
    else:
        print(f"  Gradient slice (xz) saved to: {grad_xz_path}")

    # Plot gradient magnitude slice through xy-plane (z=0)
    grad_xy_path = output_dir / f"{base_name}_gradient_slice_xy.png"
    grad_xy_result = await plot({
        "solution_id": solution_id,
        "plot_type": "slice_xy",
        "output_path": str(grad_xy_path),
        "options": {
            "title": f"{base_name}: |∇φ| (xy-plane, z=0)",
            "slice_position": 0,
            "n_grid": 150,
            "quantity": "gradient_magnitude",
        },
    })

    plot_response = grad_xy_result[0]
    if hasattr(plot_response, 'text'):
        data = json.loads(plot_response.text)
        if "error" in data:
            print(f"  Warning (grad_xy): {data['error']['message']}")
        else:
            print(f"  Gradient slice (xy) saved to: {grad_xy_path}")
    else:
        print(f"  Gradient slice (xy) saved to: {grad_xy_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - {base_name}_preview.png")
    print(f"  - {base_name}_mesh.png")
    print(f"  - {base_name}_field_1d.png")
    print(f"  - {base_name}_field_slice_xz.png")
    print(f"  - {base_name}_field_slice_xy.png")
    print(f"  - {base_name}_gradient_slice_xz.png")
    print(f"  - {base_name}_gradient_slice_xy.png")


if __name__ == "__main__":
    asyncio.run(main())
