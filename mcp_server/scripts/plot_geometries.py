#!/usr/bin/env python3
"""Generate mesh plots for all implemented geometries."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap

from utils.session import reset_session


# Subdomain colors
SUBDOMAIN_COLORS = {
    0: '#E8F4FD',   # Light blue (vacuum/domain)
    1: '#4A90D9',   # Medium blue (object/inner)
    2: '#2E5A8C',   # Dark blue (shell interior/second object)
    3: '#7CB342',   # Green (third region)
}


def plot_mesh_with_subdomains(mesh_file, ax, title, regions, is_cartesian=False):
    """Plot mesh with colored subdomains."""
    import meshio
    mesh = meshio.read(mesh_file)

    points = mesh.points[:, :2]

    # Handle 3D meshes with tetrahedra
    if "tetra" in mesh.cells_dict and "triangle" not in mesh.cells_dict:
        n_tetra = len(mesh.cells_dict["tetra"])
        ax.text(0.5, 0.5, f"3D mesh\n{n_tetra:,} tetrahedra\n(2D projection not shown)",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    if "triangle" not in mesh.cells_dict:
        ax.text(0.5, 0.5, "No triangles found", ha='center', va='center',
                transform=ax.transAxes)
        return

    triangles = mesh.cells_dict["triangle"]

    # Get cell data (subdomain markers)
    cell_data = None
    if mesh.cell_data:
        for key in mesh.cell_data:
            if "triangle" in mesh.cells_dict:
                # Find the data array for triangles
                for i, cell_block in enumerate(mesh.cells):
                    if cell_block.type == "triangle":
                        if key in mesh.cell_data and len(mesh.cell_data[key]) > i:
                            cell_data = mesh.cell_data[key][i]
                            break
                if cell_data is not None:
                    break

    # Create polygon collection for filled triangles
    verts = points[triangles]

    if cell_data is not None:
        # Color by subdomain
        unique_markers = np.unique(cell_data)
        colors = [SUBDOMAIN_COLORS.get(int(m) % len(SUBDOMAIN_COLORS), '#CCCCCC')
                  for m in cell_data]

        poly = PolyCollection(verts, facecolors=colors, edgecolors='#2C3E50',
                             linewidths=0.3, alpha=0.9)
    else:
        # Single color if no subdomain data
        poly = PolyCollection(verts, facecolors='#E8F4FD', edgecolors='#2C3E50',
                             linewidths=0.3, alpha=0.9)

    ax.add_collection(poly)
    ax.autoscale()
    ax.set_aspect('equal')

    # Title with cell count and regions
    region_str = ", ".join(regions) if regions else "domain"
    ax.set_title(f"{title}\n({len(triangles):,} cells: {region_str})",
                fontsize=10, fontweight='bold')

    if is_cartesian:
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('y', fontsize=9)
    else:
        ax.set_xlabel('r', fontsize=9)
        ax.set_ylabel('z', fontsize=9)

    ax.tick_params(labelsize=8)


async def create_and_plot_mesh(geometry_config, output_dir):
    """Create mesh and save individual plot."""
    from tools.create_mesh import handle

    reset_session()

    # Use coarse for 3D meshes (medium creates too many cells)
    is_3d = geometry_config["geometry"] in ["box_3d", "custom_3d"]
    args = {
        "geometry": geometry_config["geometry"],
        "params": geometry_config["params"],
        "mesh_quality": "coarse" if is_3d else "medium",
    }
    if "physics_params" in geometry_config:
        args["physics_params"] = geometry_config["physics_params"]

    display_name = geometry_config.get("name", geometry_config["geometry"])
    print(f"Creating {display_name}...")
    result = await handle(args)
    data = json.loads(result[0].text)

    if "error" in data:
        print(f"  Error: {data['error']['message']}")
        return None

    print(f"  Created: {data['n_cells']} cells")

    # Create individual plot
    fig, ax = plt.subplots(figsize=(6, 6))

    mesh_file = data["mesh_path"] + "/mesh.xdmf"
    is_cartesian = geometry_config["geometry"] in ["box_2d", "box_3d"]
    regions = data.get("regions", [])

    try:
        plot_mesh_with_subdomains(mesh_file, ax, display_name,
                                  regions, is_cartesian)
    except Exception as e:
        print(f"  Plot error: {e}")
        ax.text(0.5, 0.5, f"Plot error:\n{str(e)[:50]}",
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    # Use custom name if provided, otherwise use geometry type
    filename = geometry_config.get("name", geometry_config["geometry"])
    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


async def main():
    """Generate all geometry plots."""
    geometries = [
        # Object-in-vacuum templates
        {
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.15, "domain_radius": 1.0},
        },
        # Sphere with measuring boundary (for fifth force evaluation at fixed distance)
        {
            "geometry": "sphere_in_vacuum",
            "name": "sphere_measuring_boundary",
            "params": {"object_radius": 0.15, "domain_radius": 1.0, "measuring_distance": 0.1},
        },
        {
            "geometry": "ellipse_in_vacuum",
            "params": {"rx": 0.2, "ry": 0.1, "domain_radius": 1.0},
        },
        # Ellipse with measuring boundary
        {
            "geometry": "ellipse_in_vacuum",
            "name": "ellipse_measuring_boundary",
            "params": {"rx": 0.2, "ry": 0.1, "domain_radius": 1.0, "measuring_distance": 0.1},
        },
        {
            "geometry": "shell_in_vacuum",
            "params": {"inner_radius": 0.1, "outer_radius": 0.2, "domain_radius": 1.0},
        },
        # Shell with larger domain radius (tests cell scaling fix)
        {
            "geometry": "shell_in_vacuum",
            "name": "shell_in_vacuum_r5",
            "params": {"inner_radius": 0.1, "outer_radius": 0.2, "domain_radius": 5.0},
        },
        {
            "geometry": "shell_in_vacuum",
            "name": "shell_in_vacuum_r10",
            "params": {"inner_radius": 0.1, "outer_radius": 0.2, "domain_radius": 10.0},
        },
        # Thin shell with large domain (the problematic case)
        {
            "geometry": "shell_in_vacuum",
            "name": "shell_thin_r10",
            "params": {"inner_radius": 0.96, "outer_radius": 1.0, "domain_radius": 10.0},
        },
        {
            "geometry": "cylinder_in_vacuum",
            "params": {"object_radius": 0.15, "object_height": 0.4, "domain_radius": 1.0},
        },
        # Large domain radius tests (verify cell scaling fix for all geometries)
        {
            "geometry": "sphere_in_vacuum",
            "name": "sphere_in_vacuum_r10",
            "params": {"object_radius": 0.15, "domain_radius": 10.0},
        },
        {
            "geometry": "ellipse_in_vacuum",
            "name": "ellipse_in_vacuum_r10",
            "params": {"rx": 0.2, "ry": 0.1, "domain_radius": 10.0},
        },
        {
            "geometry": "cylinder_in_vacuum",
            "name": "cylinder_in_vacuum_r10",
            "params": {"object_radius": 0.15, "object_height": 0.4, "domain_radius": 10.0},
        },
        {
            "geometry": "two_spheres",
            "name": "two_spheres_r10",
            "params": {"radius_1": 0.12, "radius_2": 0.08, "separation": 0.5, "domain_radius": 10.0},
        },
        {
            "geometry": "sphere_near_wall",
            "name": "sphere_near_wall_r10",
            "params": {"object_radius": 0.1, "wall_distance": 0.15, "wall_thickness": 0.1, "domain_radius": 10.0},
        },
        {
            "geometry": "custom_2d_axial",
            "name": "custom_hexagon_r10",
            "params": {
                "points": [
                    [0.0, 0.2], [0.173, 0.1], [0.173, -0.1], [0.0, -0.2],
                ],
                "domain_radius": 10.0,
            },
        },
        {
            "geometry": "two_spheres",
            "params": {"radius_1": 0.12, "radius_2": 0.08, "separation": 0.5, "domain_radius": 1.0},
        },
        {
            "geometry": "sphere_near_wall",
            "params": {"object_radius": 0.1, "wall_distance": 0.15, "wall_thickness": 0.1, "domain_radius": 1.0},
        },
        # Sphere in density profile (for astrophysical scenarios)
        {
            "geometry": "sphere_in_profile",
            "params": {"object_radius": 0.15, "domain_radius": 1.0},
        },
        {
            "geometry": "sphere_in_profile",
            "name": "sphere_in_profile_offset",
            "params": {"object_radius": 0.1, "domain_radius": 1.0, "center_z": 0.4},
        },
        # Plain domain templates
        {
            "geometry": "sphere_domain",
            "params": {"domain_radius": 1.0},
        },
        {
            "geometry": "disk",
            "params": {"domain_radius": 1.0},
        },
        {
            "geometry": "box_2d",
            "params": {"domain_width": 2.0, "domain_height": 1.5},
        },
        {
            "geometry": "box_3d",
            "params": {"domain_width": 2.0, "domain_height": 1.5, "domain_depth": 1.0},
        },
        # Parallel plates (translation symmetry)
        {
            "geometry": "parallel_plates",
            "params": {"plate_separation": 1.0, "plate_thickness": 0.1},
        },
        {
            "geometry": "parallel_plates",
            "name": "parallel_plates_thick",
            "params": {"plate_separation": 1.0, "plate_thickness": 0.3, "domain_height": 1.5},
        },
        # Custom shapes
        {
            "geometry": "custom_2d_axial",
            "name": "custom_star",
            "params": {
                # Axisymmetric star: only r >= 0 points (revolves to create 3D star-like shape)
                "points": [
                    [0.0, 0.25], [0.059, 0.081], [0.154, 0.077],
                    [0.095, 0.0], [0.154, -0.077], [0.059, -0.081],
                    [0.0, -0.25],
                ],
                "domain_radius": 1.0,
            },
        },
        {
            "geometry": "custom_2d_axial",
            "name": "custom_hexagon",
            "params": {
                # Axisymmetric half-hexagon: only r >= 0 points
                "points": [
                    [0.0, 0.2], [0.173, 0.1], [0.173, -0.1], [0.0, -0.2],
                ],
                "domain_radius": 1.0,
            },
        },
        # Physics-aware refinement examples
        {
            "geometry": "sphere_in_vacuum",
            "name": "sphere_thin_shell",
            "params": {"object_radius": 0.15, "domain_radius": 1.0},
            "physics_params": {"lambda": {"object": 0.01}},
        },
        {
            "geometry": "sphere_in_vacuum",
            "name": "sphere_very_thin_shell",
            "params": {"object_radius": 0.15, "domain_radius": 1.0},
            "physics_params": {"lambda": {"object": 0.002}},
        },
        {
            "geometry": "two_spheres",
            "name": "two_spheres_thin_shell",
            "params": {"radius_1": 0.12, "radius_2": 0.08, "separation": 0.5, "domain_radius": 1.0},
            "physics_params": {"lambda": {"sphere_1": 0.005, "sphere_2": 0.005}},
        },
        {
            "geometry": "custom_2d_axial",
            "name": "custom_star_thin_shell",
            "params": {
                # Axisymmetric star: only r >= 0 points
                "points": [
                    [0.0, 0.25], [0.059, 0.081], [0.154, 0.077],
                    [0.095, 0.0], [0.154, -0.077], [0.059, -0.081],
                    [0.0, -0.25],
                ],
                "domain_radius": 1.0,
            },
            "physics_params": {"lambda": {"object": 0.01}},
        },
        # Test case for translation symmetry
        {
            "geometry": "custom_2d_translation",
            "name": "custom_2d_translation",
            "params": {
                # Full hexagon - all points kept with translation symmetry (extruded in z)
                "points": [
                    [0.2, 0.0], [0.1, 0.173], [-0.1, 0.173],
                    [-0.2, 0.0], [-0.1, -0.173], [0.1, -0.173],
                ],
                "domain_radius": 1.0,
            },
        },
    ]

    # Create plots directory
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    print(f"Saving plots to: {output_dir}\n")

    for geom in geometries:
        await create_and_plot_mesh(geom, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
