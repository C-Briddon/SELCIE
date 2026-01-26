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

    @pytest.mark.asyncio
    async def test_integrate_quantity_all(self, solution_id):
        """Test integrate mode with quantity='all' returns force and torque."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_id,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "all",
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["quantity"] == "all"

        # Force should be present
        assert "F_r" in data["data"]
        assert "F_z" in data["data"]
        assert "F_magnitude" in data["data"]

        # 2D axial mesh: torque is zero by symmetry
        assert data["data"]["tau_x"] == 0.0
        assert data["data"]["tau_y"] == 0.0
        assert data["data"]["tau_z"] == 0.0
        assert "torque_note" in data["data"]
        assert "axial symmetry" in data["data"]["torque_note"]


class TestEvaluateIntegrateTranslation:
    """Test integrate mode with translation symmetry for τ_z calculation."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest_asyncio.fixture
    async def solution_translation(self):
        """Create a 2D mesh with translation symmetry and solve."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve

        # Create custom_2d_translation geometry (square object)
        # Points define a square centered at origin
        mesh_result = await create_mesh({
            "geometry": "custom_2d_translation",
            "params": {
                "points": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
                "domain_radius": 1.0,
            },
            "mesh_quality": "very_coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)

        if "error" in mesh_data:
            pytest.skip(f"Could not create mesh: {mesh_data['error']}")

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
            pytest.skip(f"Solve failed: {solve_data}")

        return solve_data["solution_id"]

    @pytest.mark.asyncio
    async def test_integrate_torque_translation(self, solution_translation):
        """Test integrate mode computes τ_z for translation symmetry."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_translation,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "torque",
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["symmetry"] == "translation"

        # Only τ_z should be present (per unit length)
        assert "tau_z" in data["data"]
        assert "tau_magnitude" in data["data"]
        assert "torque_origin" in data["data"]
        assert "torque_note" in data["data"]
        assert "translation symmetry" in data["data"]["torque_note"]

        # τ_x, τ_y should NOT be present (not computed for translation)
        assert "tau_x" not in data["data"]
        assert "tau_y" not in data["data"]


class TestEvaluateIntegrate3D:
    """Test integrate mode with 3D meshes for torque calculations."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest_asyncio.fixture
    async def solution_3d(self):
        """Create a 3D mesh and solve to get a solution for testing torque."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve

        # Create 3D mesh using custom_step with unit cube
        mesh_result = await create_mesh({
            "geometry": "custom_step",
            "params": {
                "step_file": "tests/test_data/unit_cube.step",
                "domain_radius": 3.0,
            },
            "mesh_quality": "very_coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)

        if "error" in mesh_data:
            pytest.skip(f"Could not create 3D mesh: {mesh_data['error']}")

        mesh_id = mesh_data["mesh_id"]

        # Solve with low alpha to ensure convergence for 3D
        solve_result = await solve({
            "mesh_id": mesh_id,
            "alpha": 0.01,
            "density": {
                "object": 100.0,
                "vacuum": 1.0,
            },
        })
        solve_data = json.loads(solve_result[0].text)

        if "error" in solve_data and solve_data.get("status") == "failed":
            pytest.skip(f"Solve failed: {solve_data}")

        return solve_data["solution_id"]

    @pytest.mark.asyncio
    async def test_integrate_torque_3d(self, solution_3d):
        """Test integrate mode computes torque for 3D mesh."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_3d,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "torque",
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["quantity"] == "torque"

        # Torque components should be present
        assert "tau_x" in data["data"]
        assert "tau_y" in data["data"]
        assert "tau_z" in data["data"]
        assert "tau_magnitude" in data["data"]
        assert "torque_origin" in data["data"]

        # Default origin should be [0, 0, 0]
        assert data["data"]["torque_origin"] == [0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_integrate_torque_custom_origin(self, solution_3d):
        """Test integrate mode with custom torque origin."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_3d,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "torque",
                "torque_origin": [0.5, 0.0, 0.0],
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["data"]["torque_origin"] == [0.5, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_integrate_all_3d(self, solution_3d):
        """Test integrate mode with quantity='all' returns force and torque for 3D."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_3d,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "all",
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["quantity"] == "all"

        # Force should be present
        assert "F_x" in data["data"]
        assert "F_y" in data["data"]
        assert "F_z" in data["data"]
        assert "F_magnitude" in data["data"]

        # Torque should also be present for 3D
        assert "tau_x" in data["data"]
        assert "tau_y" in data["data"]
        assert "tau_z" in data["data"]
        assert "tau_magnitude" in data["data"]

    @pytest.mark.asyncio
    async def test_integrate_with_bounds(self, solution_3d):
        """Test integrate mode with bounds filtering."""
        from tools.evaluate import handle

        # First get full integration to compare
        full_result = await handle({
            "solution_id": solution_3d,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "mass",
            },
        })
        full_data = json.loads(full_result[0].text)
        assert "error" not in full_data
        full_mass = full_data["data"]["mass"]
        full_cells = full_data["data"]["n_cells"]

        # Now integrate with bounds (upper half only)
        bounded_result = await handle({
            "solution_id": solution_3d,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "mass",
                "bounds": {"z_min": 0.0},
            },
        })
        bounded_data = json.loads(bounded_result[0].text)

        assert "error" not in bounded_data
        assert "bounds_applied" in bounded_data
        assert bounded_data["bounds_applied"]["z_min"] == 0.0

        # Bounded region should have fewer cells and less mass
        bounded_mass = bounded_data["data"]["mass"]
        bounded_cells = bounded_data["data"]["n_cells"]

        assert bounded_cells < full_cells
        assert bounded_mass < full_mass

    @pytest.mark.asyncio
    async def test_integrate_bounds_no_cells(self, solution_3d):
        """Test integrate mode with bounds that exclude all cells."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_3d,
            "mode": "integrate",
            "params": {
                "region": "object",
                "quantity": "mass",
                "bounds": {"z_min": 1000.0},  # Way outside object
            },
        })

        data = json.loads(result[0].text)

        assert "error" in data
        assert data["error"]["code"] == "NO_CELLS_IN_BOUNDS"


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


class TestBoundaryMax:
    """Test boundary_max mode for measuring boundary evaluation."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest_asyncio.fixture
    async def solution_with_measuring_boundary(self):
        """Create mesh with measuring_boundary and solve."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve

        # Create sphere with measuring_distance
        mesh_result = await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.1,
                "domain_radius": 1.0,
                "measuring_distance": 0.05,
            },
            "mesh_quality": "coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)

        if "error" in mesh_data:
            pytest.skip(f"Could not create mesh: {mesh_data['error']}")

        mesh_id = mesh_data["mesh_id"]

        # Solve - vacuum density auto-assigned to measuring_boundary
        solve_result = await solve({
            "mesh_id": mesh_id,
            "alpha": 1e-6,
            "density": {
                "object": 1e6,
                "vacuum": 1e-10,
            },
        })
        solve_data = json.loads(solve_result[0].text)

        if "error" in solve_data and solve_data.get("status") == "failed":
            pytest.skip(f"Solve failed: {solve_data}")

        return solve_data["solution_id"]

    @pytest.mark.asyncio
    async def test_boundary_max_basic(self, solution_with_measuring_boundary):
        """Test boundary_max mode returns expected structure."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_with_measuring_boundary,
            "mode": "boundary_max",
            "params": {
                "region": "measuring_boundary",
            },
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["mode"] == "boundary_max"
        assert data["region"] == "measuring_boundary"
        assert data["boundary"] == "outer"  # Default is outer

        # Check data structure
        assert "max_gradient" in data["data"]
        assert "max_position" in data["data"]
        assert "mean_gradient" in data["data"]
        assert "n_points" in data["data"]
        assert "radius_range" in data["data"]

        # Values should be positive
        assert data["data"]["max_gradient"] > 0
        assert data["data"]["mean_gradient"] > 0
        assert data["data"]["n_points"] > 0

        # Position should be a list of coordinates
        assert isinstance(data["data"]["max_position"], list)
        assert len(data["data"]["max_position"]) == 2  # 2D mesh

        # Radius range should be near the outer boundary (object_radius + measuring_distance)
        # object_radius=0.1, measuring_distance=0.05, so outer boundary at ~0.15
        radius_range = data["data"]["radius_range"]
        assert radius_range[0] > 0.14  # Should be on outer boundary, not inner
        assert radius_range[1] < 0.20  # But not too far out

    @pytest.mark.asyncio
    async def test_boundary_max_invalid_region(self, solution_with_measuring_boundary):
        """Test boundary_max with invalid region returns error."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_with_measuring_boundary,
            "mode": "boundary_max",
            "params": {
                "region": "nonexistent",
            },
        })

        data = json.loads(result[0].text)

        assert "error" in data
        assert data["error"]["code"] == "INVALID_REGION"

    @pytest.mark.asyncio
    async def test_boundary_max_default_region(self, solution_with_measuring_boundary):
        """Test boundary_max defaults to measuring_boundary region."""
        from tools.evaluate import handle

        result = await handle({
            "solution_id": solution_with_measuring_boundary,
            "mode": "boundary_max",
            "params": {},  # No region specified
        })

        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["region"] == "measuring_boundary"


class TestMeasuringBoundaryAutoAssign:
    """Test that measuring_boundary density is auto-assigned from vacuum."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest.mark.asyncio
    async def test_solve_without_measuring_boundary_density(self):
        """Solve should succeed without explicit measuring_boundary density."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve

        # Create mesh with measuring_distance
        mesh_result = await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.1,
                "domain_radius": 1.0,
                "measuring_distance": 0.05,
            },
            "mesh_quality": "coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)
        assert "error" not in mesh_data
        assert "measuring_boundary" in mesh_data["regions"]

        mesh_id = mesh_data["mesh_id"]

        # Solve without specifying measuring_boundary density
        solve_result = await solve({
            "mesh_id": mesh_id,
            "alpha": 1e-6,
            "density": {
                "object": 1e6,
                "vacuum": 1e-10,
                # No measuring_boundary - should be auto-assigned
            },
        })
        solve_data = json.loads(solve_result[0].text)

        # Should succeed without error about missing density
        assert solve_data.get("status") != "failed"
        assert "error" not in solve_data or "No density specified" not in str(solve_data.get("error", ""))
        assert solve_data.get("status") in ["converged", "max_iterations_not_converged"]

    @pytest.mark.asyncio
    async def test_solve_with_explicit_measuring_boundary_density(self):
        """Solve should use explicit measuring_boundary density if provided."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle as solve

        # Create mesh with measuring_distance
        mesh_result = await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.1,
                "domain_radius": 1.0,
                "measuring_distance": 0.05,
            },
            "mesh_quality": "coarse",
        })
        mesh_data = json.loads(mesh_result[0].text)
        mesh_id = mesh_data["mesh_id"]

        # Solve with explicit measuring_boundary density (different from vacuum)
        solve_result = await solve({
            "mesh_id": mesh_id,
            "alpha": 1e-6,
            "density": {
                "object": 1e6,
                "vacuum": 1e-10,
                "measuring_boundary": 1.0,  # Explicit, different from vacuum
            },
        })
        solve_data = json.loads(solve_result[0].text)

        # Should succeed
        assert solve_data.get("status") in ["converged", "max_iterations_not_converged"]
