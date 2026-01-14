#!/usr/bin/env python3
"""Tests for evaluate tool.

Note: Some tests using FEniCS may segfault when run in batch due to
FEniCS memory management issues with pytest-asyncio. Tests pass when
run individually or in small batches. Helper function tests (TestEvaluateHelpers)
don't use FEniCS and always pass.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
import pytest_asyncio

from utils.session import reset_session, get_session


class TestEvaluate:
    """Test evaluate tool."""

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
                "vacuum_radius": 1.0,
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
    async def test_radial_mode(self, solution_id):
        """Test radial evaluation mode."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "radial",
            "params": {
                "n_points": 50,
                "r_min": 0.01,
                "r_max": 0.9,
            },
            "quantities": ["field", "gradient_magnitude"],
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["mode"] == "radial"
        assert data["n_points"] == 50
        assert "data" in data
        assert "r" in data["data"]
        assert "field" in data["data"]
        assert "gradient_magnitude" in data["data"]
        assert len(data["data"]["r"]) == 50
        assert len(data["data"]["field"]) == 50
        assert len(data["data"]["gradient_magnitude"]) == 50

    @pytest.mark.asyncio
    async def test_radial_log_spacing(self, solution_id):
        """Test radial mode with log spacing."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "radial",
            "params": {
                "n_points": 20,
                "r_min": 0.01,
                "r_max": 0.9,
                "log_spacing": True,
            },
            "quantities": ["field"],
        })

        data = json.loads(result[0].text)
        assert "error" not in data

        # Check log spacing - ratios should be roughly constant
        r_values = data["data"]["r"]
        ratios = [r_values[i+1] / r_values[i] for i in range(len(r_values)-1)]
        # All ratios should be similar for log spacing
        assert max(ratios) / min(ratios) < 1.1

    @pytest.mark.asyncio
    async def test_line_mode(self, solution_id):
        """Test line evaluation mode."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "line",
            "params": {
                "start": [0.0, 0.0],
                "end": [0.5, 0.5],
                "n_points": 30,
            },
            "quantities": ["field"],
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["mode"] == "line"
        assert "distance" in data["data"]
        assert "coordinates" in data["data"]
        assert len(data["data"]["field"]) == 30

    @pytest.mark.asyncio
    async def test_line_mode_missing_params(self, solution_id):
        """Test line mode error when missing start/end."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "line",
            "params": {
                "n_points": 30,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_points_mode(self, solution_id):
        """Test points evaluation mode."""
        from tools.evaluate import handle

        coords = [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.0, 0.1]]

        result = await handle({
            "solution_id": solution_id,
            "mode": "points",
            "params": {
                "coordinates": coords,
            },
            "quantities": ["field"],
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["mode"] == "points"
        assert len(data["data"]["field"]) == 4
        assert data["data"]["coordinates"] == coords

    @pytest.mark.asyncio
    async def test_points_mode_missing_coords(self, solution_id):
        """Test points mode error when missing coordinates."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "points",
            "params": {},
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_grid_mode(self, solution_id):
        """Test grid evaluation mode."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "grid",
            "params": {
                "r_range": [0.0, 0.5],
                "z_range": [-0.3, 0.3],
                "n_r": 10,
                "n_z": 8,
            },
            "quantities": ["field"],
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["mode"] == "grid"
        assert data["grid_shape"] == [8, 10]
        assert len(data["r_values"]) == 10
        assert len(data["z_values"]) == 8
        assert len(data["data"]["field"]) == 80  # 10 * 8

    @pytest.mark.asyncio
    async def test_grid_mode_missing_params(self, solution_id):
        """Test grid mode error when missing range params."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "grid",
            "params": {
                "n_r": 10,
                "n_z": 8,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_max_in_region_mode(self, solution_id):
        """Test max_in_region evaluation mode."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "max_in_region",
            "params": {
                "region": "vacuum",
                "n_samples": 100,
            },
            "quantities": ["field", "gradient_magnitude"],
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["mode"] == "max_in_region"
        assert data["region"] == "vacuum"
        assert "n_valid_samples" in data
        assert data["n_valid_samples"] > 0

        # Check field stats
        assert "field" in data["data"]
        assert "max" in data["data"]["field"]
        assert "min" in data["data"]["field"]
        assert "mean" in data["data"]["field"]
        assert "max_location" in data["data"]["field"]

        # Check gradient_magnitude stats
        assert "gradient_magnitude" in data["data"]
        assert "max" in data["data"]["gradient_magnitude"]
        assert "min" in data["data"]["gradient_magnitude"]
        assert "mean" in data["data"]["gradient_magnitude"]
        assert "max_location" in data["data"]["gradient_magnitude"]

    @pytest.mark.asyncio
    async def test_max_in_region_with_distance(self, solution_id):
        """Test max_in_region with minimum distance from source."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "max_in_region",
            "params": {
                "region": "vacuum",
                "min_distance_from": "object",
                "min_distance": 0.1,
                "n_samples": 100,
            },
            "quantities": ["field"],
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["min_distance_from"] == "object"
        assert data["min_distance"] == 0.1

    @pytest.mark.asyncio
    async def test_max_in_region_invalid_region(self, solution_id):
        """Test max_in_region error with invalid region."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "max_in_region",
            "params": {
                "region": "nonexistent",
                "n_samples": 100,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_REGION"

    @pytest.mark.asyncio
    async def test_max_in_region_distance_from_all(self, solution_id):
        """Test max_in_region with min_distance_from='all' to exclude all other regions."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "max_in_region",
            "params": {
                "region": "vacuum",
                "min_distance_from": "all",  # Keep distance from all other regions
                "min_distance": 0.05,
                "n_samples": 100,
            },
            "quantities": ["field"],
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["min_distance_from"] == "all"
        assert data["min_distance"] == 0.05
        assert data["n_valid_samples"] > 0

    @pytest.mark.asyncio
    async def test_solution_not_found(self):
        """Test error when solution doesn't exist."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": "nonexistent",
            "mode": "radial",
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "SOLUTION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_default_quantities(self, solution_id):
        """Test default quantities when not specified."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "radial",
            "params": {
                "n_points": 10,
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        # Default should include field and gradient_magnitude
        assert "field" in data["data"]
        assert "gradient_magnitude" in data["data"]

    @pytest.mark.asyncio
    async def test_handles_points_outside_mesh(self, solution_id):
        """Test that points outside mesh return NaN gracefully."""
        from tools.evaluate import handle
        import math

        # Include a point that's definitely outside the mesh
        result = await handle({
            "solution_id": solution_id,
            "mode": "points",
            "params": {
                "coordinates": [[0.1, 0.0], [100.0, 100.0]],  # Second point outside
            },
            "quantities": ["field"],
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["n_valid"] < data["n_points"]
        # One of the field values should be NaN
        field_values = data["data"]["field"]
        assert any(math.isnan(v) if isinstance(v, float) else False for v in field_values)

    @pytest.mark.asyncio
    async def test_integrate_force_on_object(self, solution_id):
        """Test integrate mode for force on object region."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "force",
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["mode"] == "integrate"
        assert data["quantity"] == "force"
        assert data["region"] == "object"
        assert data["symmetry"] == "axial"

        # Check force components are present
        assert "F_r" in data["data"]
        assert "F_z" in data["data"]
        assert "F_magnitude" in data["data"]

        # Check sanity outputs
        assert "mass" in data["data"]
        assert "volume" in data["data"]
        assert "n_cells" in data["data"]

        # Mass should be positive and significant (object density is 1e6)
        assert data["data"]["mass"] > 0
        assert data["data"]["volume"] > 0
        assert data["data"]["n_cells"] > 0

    @pytest.mark.asyncio
    async def test_integrate_mass_only(self, solution_id):
        """Test integrate mode for mass only."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "mass",
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["quantity"] == "mass"

        # Should have mass and volume but not force
        assert "mass" in data["data"]
        assert "volume" in data["data"]
        assert "F_r" not in data["data"]
        assert "F_z" not in data["data"]

    @pytest.mark.asyncio
    async def test_integrate_invalid_region(self, solution_id):
        """Test integrate mode with invalid region."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "integrate",
            "params": {
                "region": "nonexistent",
                "quantity": "force",
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_REGION"

    @pytest.mark.asyncio
    async def test_integrate_invalid_quantity(self, solution_id):
        """Test integrate mode with invalid quantity."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "invalid",
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_QUANTITY"


class TestEvaluateHelpers:
    """Test evaluate helper functions."""

    def test_generate_radial_points_linear(self):
        """Test radial point generation with linear spacing."""
        from tools.evaluate import _generate_radial_points

        params = {
            "n_points": 5,
            "r_min": 0.0,
            "r_max": 1.0,
            "log_spacing": False,
            "direction": [1, 0],
        }
        mesh_bounds = {"r_max": 1.0}

        points, radii = _generate_radial_points(params, mesh_bounds)

        assert len(radii) == 5
        assert radii[0] == 0.0
        assert radii[-1] == 1.0
        # Check linear spacing
        import numpy as np
        assert np.allclose(np.diff(radii), 0.25)

    def test_generate_radial_points_log(self):
        """Test radial point generation with log spacing."""
        from tools.evaluate import _generate_radial_points
        import numpy as np

        params = {
            "n_points": 5,
            "r_min": 0.01,
            "r_max": 1.0,
            "log_spacing": True,
            "direction": [1, 0],
        }
        mesh_bounds = {"r_max": 1.0}

        points, radii = _generate_radial_points(params, mesh_bounds)

        assert len(radii) == 5
        assert np.isclose(radii[0], 0.01)
        assert np.isclose(radii[-1], 1.0)
        # Check log spacing - ratios should be constant
        ratios = radii[1:] / radii[:-1]
        assert np.allclose(ratios, ratios[0])

    def test_generate_line_points(self):
        """Test line point generation."""
        from tools.evaluate import _generate_line_points
        import numpy as np

        params = {
            "start": [0, 0],
            "end": [1, 1],
            "n_points": 3,
        }

        points, distances = _generate_line_points(params)

        assert len(points) == 3
        assert np.allclose(points[0], [0, 0])
        assert np.allclose(points[-1], [1, 1])
        assert np.allclose(points[1], [0.5, 0.5])

    def test_generate_grid_points(self):
        """Test grid point generation."""
        from tools.evaluate import _generate_grid_points
        import numpy as np

        params = {
            "r_range": [0, 1],
            "z_range": [0, 2],
            "n_r": 3,
            "n_z": 2,
        }

        points, r, z = _generate_grid_points(params)

        assert len(r) == 3
        assert len(z) == 2
        assert len(points) == 6  # 3 * 2
        assert np.allclose(r, [0, 0.5, 1])
        assert np.allclose(z, [0, 2])
