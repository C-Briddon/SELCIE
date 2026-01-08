#!/usr/bin/env python3
"""Compare MCP tools output with Sphere_Standalone.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import numpy as np

from utils.session import reset_session


async def run_comparison():
    """Run MCP tools and compare with Sphere_Standalone.py output."""
    reset_session()

    # Load Sphere_Standalone.py reference data
    ref_csv = Path(__file__).parent.parent.parent / "Examples" / "images" / "sphere_standalone_radial.csv"
    if not ref_csv.exists():
        print(f"Reference CSV not found: {ref_csv}")
        print("Please run Sphere_Standalone.py first")
        return False

    ref_data = np.loadtxt(ref_csv, delimiter=',', skiprows=1)
    ref_r = ref_data[:, 0]
    ref_field = ref_data[:, 1]
    ref_grad = ref_data[:, 2]

    print("Loaded reference data from Sphere_Standalone.py")
    print(f"  r range: {ref_r.min():.4f} to {ref_r.max():.4f}")
    print(f"  field range: {ref_field.min():.2e} to {ref_field.max():.2e}")
    print(f"  gradient range: {ref_grad.min():.2e} to {ref_grad.max():.2e}")
    print()

    # Create mesh (matching Sphere_Standalone.py parameters)
    from tools.create_mesh import handle as create_mesh

    print("Creating mesh...")
    result = await create_mesh({
        "geometry": "sphere_in_vacuum",
        "params": {
            "object_radius": 0.1337,
            "vacuum_radius": 1.0,
            "wall_thickness": 0.05,  # Must match Sphere_Standalone.py!
        },
        "mesh_quality": "coarse",
        "physics_params": {
            "alpha": 1e18,
            "density": {"object": 1e17, "wall": 1e17, "vacuum": 1.0},
            "n": 1,
        },
    })

    mesh_data = json.loads(result[0].text)
    if "error" in mesh_data:
        print(f"Mesh creation failed: {mesh_data['error']}")
        return False

    mesh_id = mesh_data["mesh_id"]
    print(f"  Created {mesh_id} with {mesh_data['n_cells']} cells")

    # Solve
    from tools.solve import handle as solve

    print("Solving...")
    result = await solve({
        "mesh_id": mesh_id,
        "alpha": 1e18,
        "n": 1,
        "density": {
            "object": 1e17,
            "vacuum": 1.0,
            "wall": 1e17,  # Wall has same density as source
        },
    })

    solve_data = json.loads(result[0].text)
    if "error" in solve_data:
        print(f"Solve failed: {solve_data['error']}")
        return False

    solution_id = solve_data["solution_id"]
    print(f"  Created {solution_id}, converged in {solve_data['iterations']} iterations")

    # Evaluate radial profile
    from tools.evaluate import handle as evaluate

    print("Evaluating radial profile...")
    result = await evaluate({
        "solution_id": solution_id,
        "mode": "radial",
        "quantities": ["field", "gradient_magnitude"],
        "params": {
            "n_points": 200,
            "r_min": 0.01,
            "r_max": 0.95,
        },
    })

    eval_data = json.loads(result[0].text)
    if "error" in eval_data:
        print(f"Evaluate failed: {eval_data['error']}")
        return False

    # Extract MCP data (nested under "data")
    data = eval_data["data"]
    mcp_r = np.array(data["r"])
    mcp_field = np.array(data["field"])
    mcp_grad = np.array(data["gradient_magnitude"])

    print(f"  MCP r range: {mcp_r.min():.4f} to {mcp_r.max():.4f}")
    print(f"  MCP field range: {mcp_field.min():.2e} to {mcp_field.max():.2e}")
    print(f"  MCP gradient range: {mcp_grad.min():.2e} to {mcp_grad.max():.2e}")
    print()

    # Compare at common points
    # Interpolate reference data to MCP r values
    from scipy.interpolate import interp1d

    ref_field_interp = interp1d(ref_r, ref_field, kind='linear', bounds_error=False, fill_value='extrapolate')
    ref_grad_interp = interp1d(ref_r, ref_grad, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Find common r range
    r_min = max(ref_r.min(), mcp_r.min())
    r_max = min(ref_r.max(), mcp_r.max())

    mask = (mcp_r >= r_min) & (mcp_r <= r_max)
    r_compare = mcp_r[mask]
    mcp_field_compare = mcp_field[mask]
    mcp_grad_compare = mcp_grad[mask]

    ref_field_compare = ref_field_interp(r_compare)
    ref_grad_compare = ref_grad_interp(r_compare)

    # Compute relative differences
    field_rel_diff = np.abs(mcp_field_compare - ref_field_compare) / np.abs(ref_field_compare + 1e-20)
    grad_rel_diff = np.abs(mcp_grad_compare - ref_grad_compare) / np.abs(ref_grad_compare + 1e-20)

    # Also compute differences in vacuum region only (r > 0.2, outside sphere surface)
    # This is where the physics matters - gradient inside sphere is essentially noise
    vacuum_mask = r_compare > 0.2
    vacuum_field_diff = field_rel_diff[vacuum_mask]
    vacuum_grad_diff = grad_rel_diff[vacuum_mask]

    print(f"\nVacuum region (r > 0.2) comparison:")
    print(f"  Field relative difference: mean={vacuum_field_diff.mean():.2e}, max={vacuum_field_diff.max():.2e}")
    print(f"  Gradient relative difference: mean={vacuum_grad_diff.mean():.2e}, max={vacuum_grad_diff.max():.2e}")

    print("Comparison Results:")
    print(f"  Field relative difference: mean={field_rel_diff.mean():.2e}, max={field_rel_diff.max():.2e}")
    print(f"  Gradient relative difference: mean={grad_rel_diff.mean():.2e}, max={grad_rel_diff.max():.2e}")

    # Plot comparison
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Field comparison
    axes[0, 0].plot(ref_r, ref_field, 'b-', linewidth=2, label='Sphere_Standalone.py')
    axes[0, 0].plot(mcp_r, mcp_field, 'r--', linewidth=2, label='MCP tools')
    axes[0, 0].set_xlabel('r')
    axes[0, 0].set_ylabel('φ')
    axes[0, 0].set_title('Field Profile Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Gradient comparison
    axes[0, 1].plot(ref_r, ref_grad, 'b-', linewidth=2, label='Sphere_Standalone.py')
    axes[0, 1].plot(mcp_r, mcp_grad, 'r--', linewidth=2, label='MCP tools')
    axes[0, 1].set_xlabel('r')
    axes[0, 1].set_ylabel('|∇φ|')
    axes[0, 1].set_title('Gradient Magnitude Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Field difference
    axes[1, 0].plot(r_compare, field_rel_diff, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('r')
    axes[1, 0].set_ylabel('Relative difference')
    axes[1, 0].set_title('Field Relative Difference')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Gradient difference
    axes[1, 1].plot(r_compare, grad_rel_diff, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('Relative difference')
    axes[1, 1].set_title('Gradient Relative Difference')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    output_path = Path(__file__).parent / "mcp_vs_standalone_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nComparison plot saved to: {output_path}")

    # Pass/Fail criteria - use vacuum region where physics matters
    # Inside the sphere, field is nearly constant and gradient is numerical noise
    # Field should match within 1% in vacuum region
    # Gradient should match within 10% in vacuum region (more sensitive to mesh)
    field_ok = vacuum_field_diff.mean() < 0.01  # 1% average in vacuum
    grad_ok = vacuum_grad_diff.mean() < 0.10    # 10% average in vacuum

    print()
    if field_ok and grad_ok:
        print("✓ PASS: MCP tools match Sphere_Standalone.py within tolerance (vacuum region)")
        return True
    else:
        print("✗ FAIL: MCP tools differ from Sphere_Standalone.py in vacuum region")
        if not field_ok:
            print(f"  Field: mean diff {vacuum_field_diff.mean():.2e} > 1%")
        if not grad_ok:
            print(f"  Gradient: mean diff {vacuum_grad_diff.mean():.2e} > 10%")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_comparison())
    sys.exit(0 if success else 1)
