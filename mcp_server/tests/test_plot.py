#!/usr/bin/env python3
"""Tests for plot tool.

Note: Like test_evaluate.py, these tests use FEniCS and may have
intermittent segfaults when run in batch with pytest-asyncio.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
import pytest_asyncio

from utils.session import reset_session, get_session


class TestPlot:
    """Test plot tool."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest_asyncio.fixture
    async def solution_id(self):
        """Create a mesh and solve to get a solution for testing."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve

        # Create mesh
        mesh_result = await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.15,
                "domain_radius": 1.0,
            },
            "mesh_quality": "very_coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)
        mesh_id = mesh_data["mesh_id"]

        # Solve
        solve_result = await solve({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": {
                "object": 1e6,
                "vacuum": 1.0,
            },
        })
        solve_data = json.loads(solve_result[0].text)

        if "error" in solve_data and solve_data.get("status") == "failed":
            pytest.fail(f"Solve failed: {solve_data}")

        return solve_data["solution_id"]

    @pytest.mark.asyncio
    async def test_field_1d(self, solution_id):
        """Test field_1d plot type."""
        from tools.plot import handle

        result = await handle({
            "solution_id": solution_id,
            "plot_type": "field_1d",
        })

        # Should return image + text
        assert len(result) == 2
        # First should be image
        assert result[0].type == "image"
        assert result[0].mimeType == "image/png"
        # Second should be metadata
        data = json.loads(result[1].text)
        assert data["plot_type"] == "field_1d"

    @pytest.mark.asyncio
    async def test_force_1d(self, solution_id):
        """Test force_1d plot type (requires gradient loading)."""
        from tools.plot import handle

        result = await handle({
            "solution_id": solution_id,
            "plot_type": "force_1d",
        })

        assert len(result) == 2
        assert result[0].type == "image"
        # Verify no error in metadata
        data = json.loads(result[1].text)
        assert "error" not in data
        assert data["plot_type"] == "force_1d"

    @pytest.mark.asyncio
    async def test_field_2d(self, solution_id):
        """Test field_2d plot type."""
        from tools.plot import handle

        result = await handle({
            "solution_id": solution_id,
            "plot_type": "field_2d",
            "options": {
                "colormap": "plasma"
            }
        })

        assert len(result) == 2
        assert result[0].type == "image"

    @pytest.mark.asyncio
    async def test_force_2d(self, solution_id):
        """Test force_2d plot type (requires gradient loading)."""
        from tools.plot import handle

        result = await handle({
            "solution_id": solution_id,
            "plot_type": "force_2d",
        })

        assert len(result) == 2
        assert result[0].type == "image"
        # Verify no error in metadata
        data = json.loads(result[1].text)
        assert "error" not in data
        assert data["plot_type"] == "force_2d"

    @pytest.mark.asyncio
    async def test_save_to_file(self, solution_id, tmp_path):
        """Test saving plot to file."""
        from tools.plot import handle

        output_file = str(tmp_path / "test_plot.png")

        result = await handle({
            "solution_id": solution_id,
            "plot_type": "field_1d",
            "output_path": output_file,
        })

        # Should return just text when saving to file
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["output_path"] == output_file
        assert Path(output_file).exists()

    @pytest.mark.asyncio
    async def test_solution_not_found(self):
        """Test error when solution doesn't exist."""
        from tools.plot import handle

        result = await handle({
            "solution_id": "nonexistent",
            "plot_type": "field_1d",
        })

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_field_1d_with_options(self, solution_id):
        """Test field_1d with custom options."""
        from tools.plot import handle

        result = await handle({
            "solution_id": solution_id,
            "plot_type": "field_1d",
            "options": {
                "log_r": True,
                "n_points": 50,
                "title": "Custom Title"
            }
        })

        assert len(result) == 2
        assert result[0].type == "image"


class TestPlotComparison:
    """Test comparison plot type."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest_asyncio.fixture
    async def two_solutions(self):
        """Create two solutions with different alpha values."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve

        # Create mesh
        mesh_result = await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.15,
                "domain_radius": 1.0,
            },
            "mesh_quality": "very_coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)
        mesh_id = mesh_data["mesh_id"]

        solution_ids = []

        # Solve with two different alpha values
        for alpha in [0.5, 2.0]:
            solve_result = await solve({
                "mesh_id": mesh_id,
                "alpha": alpha,
                "density": {
                    "object": 1e6,
                    "vacuum": 1.0,
                },
            })
            solve_data = json.loads(solve_result[0].text)
            if "error" not in solve_data or solve_data.get("status") != "failed":
                solution_ids.append(solve_data["solution_id"])

        return solution_ids

    @pytest.mark.asyncio
    async def test_comparison_plot(self, two_solutions):
        """Test comparison plot with multiple solutions."""
        from tools.plot import handle

        if len(two_solutions) < 2:
            pytest.skip("Need at least 2 solutions for comparison")

        result = await handle({
            "solution_id": two_solutions,
            "plot_type": "comparison",
            "options": {
                "quantity": "field",
                "legend_by": "alpha"
            }
        })

        assert len(result) == 2
        assert result[0].type == "image"
        data = json.loads(result[1].text)
        assert data["plot_type"] == "comparison"
        assert len(data["solution_ids"]) == 2

    @pytest.mark.asyncio
    async def test_comparison_needs_multiple_solutions(self):
        """Test that comparison requires at least 2 solutions."""
        from tools.plot import handle
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve

        # Create single solution
        mesh_result = await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.15, "domain_radius": 1.0},
            "mesh_quality": "very_coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)

        solve_result = await solve({
            "mesh_id": mesh_data["mesh_id"],
            "alpha": 1.0,
            "density": {"object": 1e6, "vacuum": 1.0},
        })
        solve_data = json.loads(solve_result[0].text)

        # Try comparison with single solution
        result = await handle({
            "solution_id": [solve_data["solution_id"]],
            "plot_type": "comparison",
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"
