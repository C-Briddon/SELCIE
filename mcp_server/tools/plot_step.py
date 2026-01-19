"""
plot_step tool - Preview STEP file geometry.
"""

import json
import base64
import io
import os

from mcp.types import Tool, TextContent, ImageContent

TOOL_DEFINITION = Tool(
    name="plot_step",
    description="""Preview a STEP/IGES/BREP file geometry.

Useful for verifying geometry before running a full simulation. Shows the CAD geometry with different colors for each volume.

Returns an image of the geometry along with metadata (bounding box, number of volumes, region names, etc.).""",
    inputSchema={
        "type": "object",
        "properties": {
            "step_file": {
                "type": "string",
                "description": "Path to STEP/IGES/BREP file"
            },
            "output_path": {
                "type": "string",
                "description": "Optional path to save image (PNG). If not provided, returns base64 image."
            },
            "title": {
                "type": "string",
                "description": "Optional plot title"
            },
        },
        "required": ["step_file"]
    }
)


async def handle(arguments: dict) -> list[TextContent | ImageContent]:
    """Handle plot_step tool call."""
    import gmsh
    import numpy as np

    step_file = arguments["step_file"]
    output_path = arguments.get("output_path")
    title = arguments.get("title")

    # Validate file exists
    if not os.path.exists(step_file):
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "FILE_NOT_FOUND", "message": f"STEP file not found: {step_file}"}})
        )]

    # Default title from filename
    if title is None:
        title = os.path.basename(step_file)

    # Initialize gmsh and import
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("preview")
        imported = gmsh.model.occ.importShapes(step_file, highestDimOnly=False)
        gmsh.model.occ.synchronize()

        # Get geometry info
        volumes = gmsh.model.getEntities(3)
        surfaces = gmsh.model.getEntities(2)

        # Get bounding box
        if volumes:
            entities_for_bbox = volumes
        elif surfaces:
            entities_for_bbox = surfaces
        else:
            gmsh.finalize()
            return [TextContent(
                type="text",
                text=json.dumps({"error": {"code": "NO_GEOMETRY", "message": "No geometry found in file"}})
            )]

        xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
        xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')

        for dim, tag in entities_for_bbox:
            bx1, by1, bz1, bx2, by2, bz2 = gmsh.model.occ.getBoundingBox(dim, tag)
            xmin, ymin, zmin = min(xmin, bx1), min(ymin, by1), min(zmin, bz1)
            xmax, ymax, zmax = max(xmax, bx2), max(ymax, by2), max(zmax, bz2)

        # Collect volume info
        volume_info = []
        for dim, tag in volumes:
            mass = gmsh.model.occ.getMass(dim, tag)
            bb = gmsh.model.occ.getBoundingBox(dim, tag)
            center_z = (bb[2] + bb[5]) / 2
            volume_info.append({
                "tag": tag,
                "volume": float(mass),
                "center_z": float(center_z),
                "bbox": {
                    "x": [float(bb[0]), float(bb[3])],
                    "y": [float(bb[1]), float(bb[4])],
                    "z": [float(bb[2]), float(bb[5])],
                }
            })

        # Sort by z-centroid to match region naming convention
        volume_info.sort(key=lambda v: v["center_z"])

        # Build mapping from volume tag to sorted index (for consistent coloring)
        volume_tag_to_index = {vol["tag"]: i for i, vol in enumerate(volume_info)}

        # Get boundary surfaces for each volume
        volume_surfaces = {}
        for vol in volume_info:
            vol_tag = vol["tag"]
            # Get boundary surfaces (returns list of (dim, tag) tuples)
            boundary = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
            volume_surfaces[vol_tag] = [abs(tag) for dim, tag in boundary if dim == 2]

        # Create a fine surface mesh for visualization
        char_size = max(xmax - xmin, ymax - ymin, zmax - zmin) / 50
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_size)
        gmsh.model.mesh.generate(2)  # Surface mesh only

        # Get all mesh nodes (globally)
        all_node_ids, all_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = {int(all_node_ids[i]): all_coords[3*i:3*i+3] for i in range(len(all_node_ids))}

        # Extract triangles for each surface using global node IDs
        surface_triangles = {}  # surface_tag -> list of (n1, n2, n3) node IDs
        for dim, surf_tag in surfaces:
            try:
                elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(2, surf_tag)
                if not elem_types:
                    continue

                triangles = []
                for i, elem_type in enumerate(elem_types):
                    if elem_type == 2:  # Triangle
                        nodes = node_tags[i]
                        for j in range(0, len(nodes), 3):
                            triangles.append((int(nodes[j]), int(nodes[j+1]), int(nodes[j+2])))
                surface_triangles[surf_tag] = triangles
            except Exception:
                pass

        # Render with PyVista
        import pyvista as pv

        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True, window_size=[1200, 1000])
        plotter.set_background('white')
        plotter.enable_anti_aliasing('ssaa')

        # Define colors for different volumes
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

        # Build global point array and node ID to index mapping
        all_node_ids_sorted = sorted(node_coords.keys())
        node_id_to_idx = {nid: i for i, nid in enumerate(all_node_ids_sorted)}
        points = np.array([node_coords[nid] for nid in all_node_ids_sorted])

        # Plot colored surfaces for each volume
        for vol in volume_info:
            vol_tag = vol["tag"]
            vol_idx = volume_tag_to_index[vol_tag]
            color = colors[vol_idx % len(colors)]

            # Collect all triangles for this volume's surfaces
            vol_triangles = []
            for surf_tag in volume_surfaces.get(vol_tag, []):
                if surf_tag in surface_triangles:
                    vol_triangles.extend(surface_triangles[surf_tag])

            if vol_triangles:
                # Build faces using global node indices
                faces = []
                for n1, n2, n3 in vol_triangles:
                    faces.extend([3, node_id_to_idx[n1], node_id_to_idx[n2], node_id_to_idx[n3]])

                faces = np.array(faces)
                mesh = pv.PolyData(points.copy(), faces)
                plotter.add_mesh(mesh, color=color, opacity=1.0,
                               show_edges=False, lighting=True,
                               ambient=0.3, diffuse=0.6, specular=0.0)

        plotter.add_title(title, font_size=14)
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.2)

        # Render to buffer
        img = plotter.screenshot(return_img=True)
        plotter.close()

        # Convert to PNG bytes
        from PIL import Image
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        image_data = buf.getvalue()
        renderer_used = "pyvista"

        gmsh.finalize()

        # Build metadata
        metadata = {
            "file": os.path.basename(step_file),
            "n_volumes": len(volumes),
            "n_surfaces": len(surfaces),
            "bounding_box": {
                "x": [float(xmin), float(xmax)],
                "y": [float(ymin), float(ymax)],
                "z": [float(zmin), float(zmax)],
            },
            "extent": {
                "x": float(xmax - xmin),
                "y": float(ymax - ymin),
                "z": float(zmax - zmin),
            },
            "volumes": volume_info,
            "region_names": ["object"] if len(volume_info) == 1 else [f"object_{i}" for i in range(len(volume_info))],
            "renderer": renderer_used,
        }

        # Save or return image
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(image_data)
            return [
                TextContent(type="text", text=json.dumps(metadata, indent=2)),
            ]
        else:
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            return [
                TextContent(type="text", text=json.dumps(metadata, indent=2)),
                ImageContent(type="image", data=image_b64, mimeType="image/png"),
            ]

    except Exception as e:
        gmsh.finalize()
        return [TextContent(
            type="text",
            text=json.dumps({"error": {"code": "PLOT_FAILED", "message": str(e)}})
        )]
