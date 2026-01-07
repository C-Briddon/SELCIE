"""Plot mesh tool for SELCIE MCP server."""

import base64
import io
import json
import os
import tempfile

from mcp.types import TextContent, ImageContent, Tool

from utils.session import get_session


# Subdomain colors matching plot_geometries.py
SUBDOMAIN_COLORS = {
    0: '#E8F4FD',   # Light blue (vacuum/domain)
    1: '#4A90D9',   # Medium blue (object/inner)
    2: '#2E5A8C',   # Dark blue (shell interior/second object)
    3: '#7CB342',   # Green (third region)
}


TOOL_DEFINITION = Tool(
    name="plot_mesh",
    description=(
        "Generate a visualization of a mesh showing the geometry and subdomain structure. "
        "Returns a PNG image showing the mesh with different colors for each subdomain region."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "mesh_id": {
                "type": "string",
                "description": "ID of the mesh to plot.",
            },
            "show_edges": {
                "type": "boolean",
                "default": True,
                "description": "Show cell edges. Default: true.",
            },
            "title": {
                "type": "string",
                "description": "Custom plot title. Default: auto-generated from mesh info.",
            },
            "output_path": {
                "type": "string",
                "description": "Save plot to this path. If not provided, returns base64-encoded image.",
            },
            "figsize": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "default": [8, 8],
                "description": "Figure size as [width, height] in inches.",
            },
            "dpi": {
                "type": "integer",
                "default": 150,
                "description": "Resolution in dots per inch.",
            },
        },
        "required": ["mesh_id"],
    },
)


async def handle(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle plot_mesh tool calls."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import numpy as np
    import meshio

    mesh_id = arguments["mesh_id"]
    show_edges = arguments.get("show_edges", True)
    custom_title = arguments.get("title")
    output_path = arguments.get("output_path")
    figsize = arguments.get("figsize", [8, 8])
    dpi = arguments.get("dpi", 150)

    # Get mesh from session
    session = get_session()
    mesh_info = session.get_mesh(mesh_id)

    if mesh_info is None:
        available = list(session.meshes.keys())
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "MESH_NOT_FOUND",
                "message": f"Mesh '{mesh_id}' not found",
                "available_meshes": available,
            }
        }, indent=2))]

    # Load mesh file
    mesh_file = os.path.join(mesh_info.mesh_path, "mesh.xdmf")
    if not os.path.exists(mesh_file):
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "MESH_FILE_NOT_FOUND",
                "message": f"Mesh file not found at {mesh_file}",
            }
        }, indent=2))]

    try:
        mesh = meshio.read(mesh_file)
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "MESH_READ_ERROR",
                "message": f"Failed to read mesh: {str(e)}",
            }
        }, indent=2))]

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    points = mesh.points[:, :2]

    # Handle 3D meshes with tetrahedra
    if "tetra" in mesh.cells_dict and "triangle" not in mesh.cells_dict:
        n_tetra = len(mesh.cells_dict["tetra"])
        ax.text(0.5, 0.5, f"3D mesh\n{n_tetra:,} tetrahedra\n(2D projection not shown)",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(custom_title or f"{mesh_id}: {mesh_info.geometry}")
    elif "triangle" not in mesh.cells_dict:
        ax.text(0.5, 0.5, "No triangles found in mesh",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        triangles = mesh.cells_dict["triangle"]

        # Get cell data (subdomain markers)
        cell_data = None
        if mesh.cell_data:
            for key in mesh.cell_data:
                if "triangle" in mesh.cells_dict:
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
            colors = [SUBDOMAIN_COLORS.get(int(m) % len(SUBDOMAIN_COLORS), '#CCCCCC')
                      for m in cell_data]
        else:
            colors = '#E8F4FD'

        edge_color = '#2C3E50' if show_edges else 'none'
        edge_width = 0.3 if show_edges else 0

        poly = PolyCollection(
            verts,
            facecolors=colors,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=0.9
        )

        ax.add_collection(poly)
        ax.autoscale()
        ax.set_aspect('equal')

        # Set title
        if custom_title:
            title = custom_title
        else:
            title = f"{mesh_id}: {mesh_info.geometry}\n({mesh_info.n_cells:,} cells)"

        ax.set_title(title, fontsize=12, fontweight='bold')

        # Set axis labels based on symmetry
        is_cartesian = mesh_info.geometry in ["box_2d", "box_3d"]
        if is_cartesian:
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('y', fontsize=10)
        else:
            ax.set_xlabel('r', fontsize=10)
            ax.set_ylabel('z', fontsize=10)

    plt.tight_layout()

    # Save or encode
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        return [TextContent(type="text", text=json.dumps({
            "mesh_id": mesh_id,
            "plot_saved": output_path,
            "geometry": mesh_info.geometry,
            "n_cells": mesh_info.n_cells,
        }, indent=2))]
    else:
        # Return as base64-encoded image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        image_base64 = base64.standard_b64encode(buf.read()).decode('utf-8')

        return [
            ImageContent(type="image", data=image_base64, mimeType="image/png"),
            TextContent(type="text", text=json.dumps({
                "mesh_id": mesh_id,
                "geometry": mesh_info.geometry,
                "n_cells": mesh_info.n_cells,
                "symmetry": mesh_info.symmetry,
            }, indent=2))
        ]
