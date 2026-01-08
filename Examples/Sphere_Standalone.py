#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 2025

@author: adammoss

Standalone sphere example with measuring boundary.

Creates:
1. Sphere source from points
2. Measuring boundary at distance d from source
3. Background vacuum chamber with wall

Measures the maximum fifth force on the measuring boundary.

Usage:
    python Sphere_Standalone.py [options]

Options:
    --alpha FLOAT         Coupling constant (default: 1e18)
    --n INT               Potential index (default: 1)
    --volume FLOAT        Target volume (default: 0.01)
    --distance FLOAT      Measuring distance from source (default: 0.05)
    --source-density FLOAT    Source density (default: 1e17)
    --vacuum-density FLOAT    Vacuum density (default: 1.0)

Examples:
    python Sphere_Standalone.py
    python Sphere_Standalone.py --alpha 1e15 --volume 0.005
    python Sphere_Standalone.py --distance 0.1 --source-density 1e10
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import rmtree

from SELCIE import MeshingTools, DensityProfile, FieldSolver
from dolfin import SubMesh, BoundaryMesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fifth force for a sphere with measuring boundary."
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0e18,
        help="Coupling constant (default: 1e18)"
    )
    parser.add_argument(
        "--n", type=int, default=1,
        help="Potential index (default: 1)"
    )
    parser.add_argument(
        "--volume", type=float, default=0.01,
        help="Target volume (default: 0.01)"
    )
    parser.add_argument(
        "--distance", type=float, default=0.05,
        help="Measuring distance from source (default: 0.05)"
    )
    parser.add_argument(
        "--source-density", type=float, default=1.0e17,
        help="Source density (default: 1e17)"
    )
    parser.add_argument(
        "--vacuum-density", type=float, default=1.0,
        help="Vacuum density (default: 1.0)"
    )
    parser.add_argument(
        "--mesh-quality", type=str, default="fast",
        choices=["fast", "balanced", "accurate", "fine"],
        help="Mesh quality: fast, balanced, accurate, fine (default: fast)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate plot of results"
    )
    return parser.parse_args()


# Mesh quality settings: (cell_min, cell_max, bg_cell_min, bg_cell_max, bg_dist_max)
MESH_SETTINGS = {
    'fast':     (5e-3, 0.08, 5e-3, 0.12, 0.3),
    'balanced': (3e-3, 0.06, 3e-3, 0.10, 0.3),
    'accurate': (2e-3, 0.04, 2e-3, 0.08, 0.3),
    'fine':     (1e-3, 0.02, 1e-3, 0.05, 0.3),
}

MESH_NAME = "Sphere_Standalone"
N_POINTS = 50  # Points defining the sphere boundary


# =============================================================================
# Helper functions
# =============================================================================

def sphere_points(radius: float, n_points: int) -> list:
    """Generate points for a sphere (half-plane for axisymmetric)."""
    theta = np.linspace(0, np.pi, n_points, endpoint=True)
    r = radius * np.sin(theta)  # radial distance from axis
    z = radius * np.cos(theta)  # height along axis
    return [[float(ri), float(zi), 0.0] for ri, zi in zip(r, z)]


def rescale_to_volume(points: list, target_volume: float) -> list:
    """Rescale points so the revolved shape has target volume."""
    pts = np.array(points)
    r = pts[:, 0]
    z = pts[:, 1]
    theta = np.arctan2(r, z)
    R = np.sqrt(r**2 + z**2)

    # Volume of revolution around z-axis (y in SELCIE coords)
    # V = pi * integral(r^2 dz) approximated
    dtheta = theta[1] - theta[0] if len(theta) > 1 else np.pi / len(theta)
    nu = np.pi * np.sin(dtheta) * np.sum(R * np.roll(R, 1) * (np.roll(r, 1) + r)) / 3
    eta = (target_volume / nu) ** (1/3)

    return [[p[0] * eta, p[1] * eta, 0.0] for p in points]


def static_density(value: float):
    """Create a density function that returns a constant value."""
    def func(x):
        return value
    return func


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Extract parameters from command line
    alpha = args.alpha
    n_potential = args.n
    target_volume = args.volume
    measuring_distance = args.distance
    source_density = args.source_density
    vacuum_density = args.vacuum_density
    mesh_quality = args.mesh_quality

    # Get mesh settings
    cell_min, cell_max, bg_cell_min, bg_cell_max, bg_dist_max = MESH_SETTINGS[mesh_quality]
    dist_max = 0.1

    print("=" * 60)
    print("SPHERE WITH MEASURING BOUNDARY")
    print("=" * 60)
    print(f"Target volume:       {target_volume}")
    print(f"Measuring distance:  {measuring_distance}")
    print(f"Alpha:               {alpha:.2e}")
    print(f"n:                   {n_potential}")
    print(f"Source density:      {source_density:.2e}")
    print(f"Vacuum density:      {vacuum_density:.2e}")
    print(f"Mesh quality:        {mesh_quality}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Generate sphere points and rescale to target volume
    # -------------------------------------------------------------------------
    print("\n1. Generating sphere geometry...")

    # Initial sphere with arbitrary radius
    initial_radius = 0.1
    points = sphere_points(initial_radius, N_POINTS)

    # Rescale to target volume
    points = rescale_to_volume(points, target_volume)

    # Calculate actual radius after rescaling
    pts_array = np.array(points)
    actual_radius = np.mean(np.sqrt(pts_array[:, 0]**2 + pts_array[:, 1]**2))
    print(f"   Rescaled radius: {actual_radius:.4f}")
    print(f"   Actual volume:   {4/3 * np.pi * actual_radius**3:.4f}")

    # -------------------------------------------------------------------------
    # Step 2: Create mesh with measuring boundary
    # -------------------------------------------------------------------------
    print("\n2. Creating mesh...")

    MT = MeshingTools(dimension=2, display_messages=False)

    # Constrain point distances for mesh quality
    points = MT.constrain_distance(points)

    # Create source shape from points
    source_surface = MT.points_to_surface(points)
    print(f"Source mesh settings: CellSizeMin={cell_min}, CellSizeMax={cell_max}, DistMax={dist_max}")
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    # Create measuring boundary at distance d from source
    # This creates a new subdomain between source and vacuum
    try:
        MT.construct_boundary(
            initial_boundaries=[points],
            d=measuring_distance,
            embed=source_surface,
            symmetry="vertical"
        )
    except TypeError:
        # Older SELCIE versions use 'holes' instead of 'embed'
        MT.construct_boundary(
            initial_boundaries=[points],
            d=measuring_distance,
            holes=source_surface,
            symmetry="vertical"
        )
    print(f"Measuring boundary mesh settings: CellSizeMin={cell_min}, CellSizeMax={cell_max}, DistMax={dist_max}")
    MT.create_subdomain(CellSizeMin=cell_min, CellSizeMax=cell_max, DistMax=dist_max)

    print(f"Background mesh settings: CellSizeMin={bg_cell_min}, CellSizeMax={bg_cell_max}, DistMax={bg_dist_max}")
    # Create background vacuum chamber with wall
    MT.create_background_mesh(
        CellSizeMin=bg_cell_min,
        CellSizeMax=bg_cell_max,
        DistMax=bg_dist_max,
        background_radius=1.0,
        wall_thickness=0.05,
        symmetry="vertical"
    )

    # Generate and convert mesh
    MT.generate_mesh(MESH_NAME, show_mesh=False)
    MT.msh_2_xdmf(MESH_NAME, delete_old_file=True, auto_override=True)
    print(f"   Mesh saved to 'Saved Meshes/{MESH_NAME}/'")

    # -------------------------------------------------------------------------
    # Step 3: Set up density profile and solve
    # -------------------------------------------------------------------------
    print("\n3. Solving chameleon field...")

    # Density functions
    vacuum = static_density(vacuum_density)
    source_wall = static_density(source_density)

    # Subdomains: 0=source, 1=measuring region, 2=vacuum, 3=wall
    p = DensityProfile(
        MESH_NAME,
        dimension=2,
        symmetry='vertical axis-symmetry',
        profiles=[source_wall, vacuum, vacuum, source_wall]
    )

    # Solve
    solver = FieldSolver(alpha, n_potential, density_profile=p)
    print(f"   Mesh cells: {solver.mesh.num_cells()}")
    print(f"   Mesh vertices: {solver.mesh.num_vertices()}")
    solver.picard(
        display_progress=True,
        tol_du=1e-12,
        linear_solver="krylov",
        krylov_method="cg",
        krylov_preconditioner="hypre_amg"
    )
    solver.calc_field_grad_vector()

    # -------------------------------------------------------------------------
    # Step 4: Measure fifth force on measuring boundary
    # -------------------------------------------------------------------------
    print("\n4. Measuring fifth force on boundary...")

    # The measuring region (subdomain 1) has two boundaries:
    #   - Inner boundary: touching the source (at radius ~ actual_radius)
    #   - Outer boundary: at distance d from source (at radius ~ actual_radius + d)
    # We want to measure on the OUTER boundary only

    measuring_subdomain = 1
    measuring_mesh = SubMesh(solver.mesh, solver.subdomains, measuring_subdomain)
    bmesh = BoundaryMesh(measuring_mesh, "exterior")

    # Expected radius of outer measuring boundary
    outer_boundary_radius = actual_radius + measuring_distance
    radius_tolerance = 0.02  # Allow some tolerance for mesh discretization

    fifth_force_max = 0.0
    fifth_force_pos = None
    forces = []
    positions = []

    for pos in bmesh.coordinates():
        dist_from_origin = np.linalg.norm(pos)

        # Only measure on OUTER boundary (at distance ~ actual_radius + d)
        # Skip inner boundary (at distance ~ actual_radius) and chamber wall
        is_outer_boundary = abs(dist_from_origin - outer_boundary_radius) < radius_tolerance
        away_from_axis = abs(pos[0]) > 1e-10

        if is_outer_boundary and away_from_axis:
            grad = solver.field_grad(pos)
            grad_mag = np.linalg.norm(grad)
            forces.append(grad_mag)
            positions.append(pos.copy())

            if grad_mag > fifth_force_max:
                fifth_force_max = grad_mag
                fifth_force_pos = pos.copy()

    # -------------------------------------------------------------------------
    # Step 5: Report results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sphere radius:           {actual_radius:.4f}")
    print(f"Measuring boundary at:   {actual_radius + measuring_distance:.4f} from center")
    print(f"")
    print(f"Fifth force (max):       {fifth_force_max:.6e}")
    if fifth_force_pos is not None:
        r_pos = np.linalg.norm(fifth_force_pos)
        print(f"Position of max:         r={fifth_force_pos[0]:.4f}, z={fifth_force_pos[1]:.4f}")
        print(f"Distance from center:    {r_pos:.4f}")
    print(f"")
    print(f"Field min:               {solver.field.vector().min():.6e}")
    print(f"Field max:               {solver.field.vector().max():.6e}")
    print("=" * 60)

    # Statistics on measuring boundary
    if forces:
        print(f"\nMeasuring boundary statistics ({len(forces)} points):")
        print(f"  Mean force:   {np.mean(forces):.6e}")
        print(f"  Std force:    {np.std(forces):.6e}")
        print(f"  Min force:    {np.min(forces):.6e}")
        print(f"  Max force:    {np.max(forces):.6e}")

    # -------------------------------------------------------------------------
    # Step 6: Plot results
    # -------------------------------------------------------------------------
    if args.plot:
        print("\n5. Generating plot...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Get mesh coordinates for plotting
        coords = solver.mesh.coordinates()
        r_coords = coords[:, 0]
        z_coords = coords[:, 1]

        # Create grid for interpolation
        r_grid = np.linspace(r_coords.min(), min(r_coords.max(), 0.4), 200)
        z_grid = np.linspace(z_coords.min(), z_coords.max(), 200)
        R_grid, Z_grid = np.meshgrid(r_grid, z_grid)

        # Density function for plotting
        def rho_func(x):
            r, z = x[0], x[1]
            dist = np.sqrt(r**2 + z**2)
            return source_density if dist < actual_radius else vacuum_density

        # Panel 1: Source density
        ax1 = axes[0, 0]
        rho_grid = np.zeros_like(R_grid)
        for i in range(R_grid.shape[0]):
            for j in range(R_grid.shape[1]):
                rho_grid[i, j] = rho_func([R_grid[i, j], Z_grid[i, j]])
        im1 = ax1.pcolormesh(R_grid, Z_grid, np.log10(rho_grid + 1),
                             shading='auto', cmap='viridis')
        ax1.set_xlabel('r')
        ax1.set_ylabel('z')
        ax1.set_title('Source Density (log10)')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)

        # Panel 2: Field solution (log scale)
        ax2 = axes[0, 1]
        field_grid = np.zeros_like(R_grid)
        for i in range(R_grid.shape[0]):
            for j in range(R_grid.shape[1]):
                try:
                    field_grid[i, j] = solver.field([R_grid[i, j], Z_grid[i, j]])
                except RuntimeError:
                    field_grid[i, j] = np.nan
        field_grid = np.where(field_grid > 0, field_grid, np.nan)
        im2 = ax2.pcolormesh(R_grid, Z_grid, np.log10(field_grid),
                             shading='auto', cmap='plasma')
        ax2.set_xlabel('r')
        ax2.set_ylabel('z')
        ax2.set_title('Chameleon Field (log10)')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)

        # Panel 3: Field gradient magnitude (log scale)
        ax3 = axes[1, 0]
        grad_grid = np.zeros_like(R_grid)
        for i in range(R_grid.shape[0]):
            for j in range(R_grid.shape[1]):
                try:
                    grad = solver.field_grad([R_grid[i, j], Z_grid[i, j]])
                    grad_grid[i, j] = np.linalg.norm(grad)
                except RuntimeError:
                    grad_grid[i, j] = np.nan
        grad_grid = np.where(grad_grid > 0, grad_grid, np.nan)
        im3 = ax3.pcolormesh(R_grid, Z_grid, np.log10(grad_grid),
                             shading='auto', cmap='hot')
        ax3.set_xlabel('r')
        ax3.set_ylabel('z')
        ax3.set_title('Field Gradient (log10)')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3)

        # Panel 4: Mesh, source, and measuring boundary
        ax4 = axes[1, 1]

        # Plot mesh triangles
        cells = solver.mesh.cells()
        mesh_coords = solver.mesh.coordinates()
        ax4.triplot(mesh_coords[:, 0], mesh_coords[:, 1], cells, 'k-', linewidth=0.2, alpha=0.3)

        # Plot source (sphere)
        theta_src = np.linspace(0, np.pi, 100)
        src_r = actual_radius * np.sin(theta_src)
        src_z = actual_radius * np.cos(theta_src)
        ax4.plot(src_r, src_z, 'b-', linewidth=2, label='Source')
        ax4.fill(src_r, src_z, alpha=0.3, color='blue')

        # Plot measuring boundary
        meas_r = outer_boundary_radius * np.sin(theta_src)
        meas_z = outer_boundary_radius * np.cos(theta_src)
        ax4.plot(meas_r, meas_z, 'g--', linewidth=2, label='Measuring boundary')

        # Plot max force position
        if fifth_force_pos is not None:
            ax4.plot(fifth_force_pos[0], fifth_force_pos[1], 'r*',
                     markersize=15, label=f'Max force: {fifth_force_max:.2e}')

        ax4.set_xlabel('r')
        ax4.set_ylabel('z')
        ax4.set_title('Mesh and Boundaries')
        ax4.set_aspect('equal')
        ax4.legend(loc='upper right')
        ax4.set_xlim(-0.02, 0.4)

        plt.suptitle(f'Sphere R={actual_radius:.4f}, V={target_volume}, d={measuring_distance}',
                     fontsize=14)
        plt.tight_layout()

        # Save to images directory
        images_dir = Path(__file__).parent / "images"
        images_dir.mkdir(exist_ok=True)
        plot_filename = images_dir / "sphere_standalone.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"   Plot saved to: {plot_filename}")
        plt.close()

        # -------------------------------------------------------------------------
        # Additional 1D radial profile plots (for comparison with MCP output)
        # -------------------------------------------------------------------------
        print("   Generating 1D radial profiles...")

        # Sample radial points along r-axis (z=0)
        n_radial = 200
        r_min = 0.01
        r_max = 0.9
        r_radial = np.linspace(r_min, r_max, n_radial)

        field_radial = []
        grad_radial = []
        r_valid = []

        for r in r_radial:
            try:
                phi = solver.field([r, 0.0])
                grad = solver.field_grad([r, 0.0])
                grad_mag = np.linalg.norm(grad)
                field_radial.append(phi)
                grad_radial.append(grad_mag)
                r_valid.append(r)
            except RuntimeError:
                pass  # Point outside mesh

        r_valid = np.array(r_valid)
        field_radial = np.array(field_radial)
        grad_radial = np.array(grad_radial)

        # Create 1D profile figure
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Field radial profile φ(r)
        ax_field = axes2[0]
        ax_field.plot(r_valid, field_radial, 'b-', linewidth=2, label='φ(r)')
        ax_field.axvline(x=actual_radius, color='gray', linestyle='--', alpha=0.7, label=f'Source radius ({actual_radius:.3f})')
        ax_field.set_xlabel('r', fontsize=11)
        ax_field.set_ylabel('φ', fontsize=11)
        ax_field.set_title(f'Radial Field Profile (α={alpha:.2e})', fontsize=12)
        ax_field.set_yscale('log')
        ax_field.grid(True, alpha=0.3)
        ax_field.legend()

        # Panel 2: Gradient magnitude |∇φ|(r)
        ax_grad = axes2[1]
        ax_grad.plot(r_valid, grad_radial, 'r-', linewidth=2, label='|∇φ|(r)')
        ax_grad.axvline(x=actual_radius, color='gray', linestyle='--', alpha=0.7, label=f'Source radius ({actual_radius:.3f})')
        ax_grad.axvline(x=actual_radius + measuring_distance, color='green', linestyle=':', alpha=0.7, label=f'Measuring boundary ({actual_radius + measuring_distance:.3f})')
        ax_grad.set_xlabel('r', fontsize=11)
        ax_grad.set_ylabel('|∇φ|', fontsize=11)
        ax_grad.set_title(f'Radial Gradient Magnitude (α={alpha:.2e})', fontsize=12)
        ax_grad.set_yscale('log')
        ax_grad.grid(True, alpha=0.3)
        ax_grad.legend()

        plt.suptitle(f'1D Radial Profiles - Sphere R={actual_radius:.4f}', fontsize=14)
        plt.tight_layout()

        plot_filename_1d = images_dir / "sphere_standalone_1d.png"
        plt.savefig(plot_filename_1d, dpi=150)
        print(f"   1D profiles saved to: {plot_filename_1d}")
        plt.close()

        # Also save the raw data to CSV for exact comparison
        csv_filename = images_dir / "sphere_standalone_radial.csv"
        np.savetxt(
            csv_filename,
            np.column_stack([r_valid, field_radial, grad_radial]),
            header='r,field,gradient_magnitude',
            delimiter=',',
            comments=''
        )
        print(f"   Radial data saved to: {csv_filename}")

    # Clean up mesh files
    mesh_path = Path("Saved Meshes") / MESH_NAME
    if mesh_path.exists():
        rmtree(mesh_path)
        print(f"\nCleaned up mesh files.")


if __name__ == "__main__":
    main()

