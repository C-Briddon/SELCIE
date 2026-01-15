#!/usr/bin/env python3
"""Tests for plot_mesh tool."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import base64
import pytest

from utils.session import reset_session


class TestPlotMesh:
    """Test plot_mesh tool."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest.mark.asyncio
    async def test_plot_mesh_returns_image(self):
        """Plot mesh should return base64 image."""
        from tools.create_mesh import handle as create_mesh
        from tools.plot_mesh import handle as plot_mesh

        # First create a mesh
        await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
            "mesh_quality": "coarse",
        })

        # Now plot it
        result = await plot_mesh({"mesh_id": "mesh_001"})

        # Should return ImageContent and TextContent
        assert len(result) == 2
        assert result[0].type == "image"
        assert result[0].mimeType == "image/png"
        # Verify it's valid base64
        image_data = base64.b64decode(result[0].data)
        assert len(image_data) > 0
        # PNG magic bytes
        assert image_data[:8] == b'\x89PNG\r\n\x1a\n'

        # TextContent should have mesh info
        data = json.loads(result[1].text)
        assert data["mesh_id"] == "mesh_001"
        assert data["geometry"] == "sphere_in_vacuum"

    @pytest.mark.asyncio
    async def test_plot_mesh_not_found(self):
        """Non-existent mesh should return error."""
        from tools.plot_mesh import handle as plot_mesh

        result = await plot_mesh({"mesh_id": "nonexistent"})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "MESH_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_plot_mesh_save_to_file(self, tmp_path):
        """Plot mesh can save to file."""
        from tools.create_mesh import handle as create_mesh
        from tools.plot_mesh import handle as plot_mesh

        # Create a mesh
        await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
            "mesh_quality": "coarse",
        })

        # Save to file
        output_path = str(tmp_path / "test_mesh.png")
        result = await plot_mesh({
            "mesh_id": "mesh_001",
            "output_path": output_path,
        })

        # Should return TextContent only
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["plot_saved"] == output_path

        # File should exist
        assert Path(output_path).exists()
        # And be a valid PNG
        with open(output_path, "rb") as f:
            assert f.read(8) == b'\x89PNG\r\n\x1a\n'

    @pytest.mark.asyncio
    async def test_plot_mesh_custom_title(self):
        """Plot mesh with custom title."""
        from tools.create_mesh import handle as create_mesh
        from tools.plot_mesh import handle as plot_mesh

        await create_mesh({
            "geometry": "ellipse_in_vacuum",
            "params": {"rx": 0.2, "ry": 0.1, "domain_radius": 1.0},
            "mesh_quality": "coarse",
        })

        result = await plot_mesh({
            "mesh_id": "mesh_001",
            "title": "My Custom Ellipse",
            "show_edges": False,
        })

        assert len(result) == 2
        assert result[0].type == "image"
