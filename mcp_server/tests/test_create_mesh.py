#!/usr/bin/env python3
"""Tests for create_mesh tool."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import pytest

from utils.session import reset_session


class TestCreateMeshSphereInVacuum:
    """Test sphere_in_vacuum geometry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest.mark.asyncio
    async def test_basic_sphere(self):
        """Create a basic sphere in vacuum mesh."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.1,
                "vacuum_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["mesh_id"] == "mesh_001"
        assert data["geometry"] == "sphere_in_vacuum"
        assert data["symmetry"] == "axial"
        assert data["dimension"] == 2
        assert "object" in data["regions"]
        assert "vacuum" in data["regions"]

    @pytest.mark.asyncio
    async def test_missing_params(self):
        """Missing required parameters should error."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.1,
                # missing vacuum_radius
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "MISSING_PARAMS"

    @pytest.mark.asyncio
    async def test_custom_id(self):
        """Custom mesh ID should be used."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.1,
                "vacuum_radius": 1.0,
            },
            "custom_id": "my_mesh",
        })

        data = json.loads(result[0].text)
        assert data["mesh_id"] == "my_mesh"

    @pytest.mark.asyncio
    async def test_duplicate_id_error(self):
        """Duplicate custom ID should error."""
        from tools.create_mesh import handle

        # First mesh
        await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "vacuum_radius": 1.0},
            "custom_id": "duplicate",
        })

        # Second mesh with same ID
        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.2, "vacuum_radius": 2.0},
            "custom_id": "duplicate",
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_ID"


class TestCreateMeshEllipse:
    """Test ellipse_in_vacuum geometry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_oblate_ellipse(self):
        """Create an oblate ellipse (flattened)."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "ellipse_in_vacuum",
            "params": {
                "rx": 0.1,
                "ry": 0.05,  # flattened in z
                "vacuum_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "ellipse_in_vacuum"


class TestCreateMeshDisk:
    """Test disk geometry (plain domain)."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_disk(self):
        """Create a disk domain."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "disk",
            "params": {"radius": 1.0},
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "disk"
        assert "domain" in data["regions"]


class TestCreateMeshBox2D:
    """Test box_2d geometry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_box_2d(self):
        """Create a 2D box domain."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "box_2d",
            "params": {"width": 2.0, "height": 1.0},
            "mesh_quality": "coarse",
            "symmetry": "none",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "box_2d"
        assert data["symmetry"] == "none"


class TestCreateMeshSphereNearWall:
    """Test sphere_near_wall geometry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_sphere_near_wall(self):
        """Create a sphere near a physical wall."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_near_wall",
            "params": {
                "object_radius": 0.1,
                "wall_distance": 0.15,
                "wall_thickness": 0.1,
                "vacuum_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "sphere_near_wall"
        assert "wall" in data["regions"]  # Physical wall subdomain
        assert "sphere" in data["regions"]
        assert "vacuum" in data["regions"]
        assert data["domain_bounds"]["z_min"] == -0.1  # Bottom of wall


class TestCreateMeshPhysicsRefinement:
    """Test physics-aware mesh refinement."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_physics_params_applies_refinement(self):
        """Physics params should apply refinement for thin shells."""
        from tools.create_mesh import handle

        # With thin shell physics (using lambda dict)
        result_physics = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "vacuum_radius": 1.0},
            "mesh_quality": "very_coarse",
            "physics_params": {"lambda": {"object": 0.01}},
        })
        data_physics = json.loads(result_physics[0].text)

        assert "error" not in data_physics
        assert "physics_refinement" in data_physics
        assert data_physics["physics_refinement"]["refinement_applied"] is True
        assert data_physics["physics_refinement"]["lambda_min"] == 0.01
        # Should produce a reasonable mesh
        assert data_physics["n_cells"] > 1000

    @pytest.mark.asyncio
    async def test_extreme_thin_shell_capped(self):
        """Extreme thin shell should hit limits, not explode."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "vacuum_radius": 1.0},
            "mesh_quality": "coarse",
            "physics_params": {"lambda": {"object": 0.00001}},  # Extremely thin
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        # Should not explode - capped at reasonable cell count
        assert data["n_cells"] < 50000

    @pytest.mark.asyncio
    async def test_lambda_computed_from_alpha_density(self):
        """Lambda should be auto-computed from alpha and density dict."""
        from tools.create_mesh import handle
        import math

        alpha = 1e18
        object_density = 1e17
        n = 1

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "vacuum_radius": 1.0},
            "mesh_quality": "coarse",
            "physics_params": {
                "alpha": alpha,
                "density": {"object": object_density, "vacuum": 1.0},
                "n": n,
            },
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert "physics_refinement" in data
        assert data["physics_refinement"]["refinement_applied"] is True

        # Verify lambda was computed correctly for each region
        # λ = √(α / n(n+1)) × ρ^(-(n+2)/(2(n+1)))
        expected_lambda_object = math.sqrt(alpha / (n * (n + 1))) * (object_density ** (-(n + 2) / (2 * (n + 1))))

        # Check lambda_per_region contains computed values
        assert "lambda_per_region" in data["physics_refinement"]
        actual_lambda_object = data["physics_refinement"]["lambda_per_region"]["object"]
        assert abs(actual_lambda_object - expected_lambda_object) / expected_lambda_object < 0.01  # Within 1%

        # The minimum lambda should be from the densest region (object)
        assert data["physics_refinement"]["lambda_min_region"] == "object"

        # Verify computed_from is included
        assert "computed_from" in data["physics_refinement"]
        assert data["physics_refinement"]["computed_from"]["alpha"] == alpha
        assert data["physics_refinement"]["computed_from"]["density"]["object"] == object_density

    @pytest.mark.asyncio
    async def test_lambda_direct_overrides_computed(self):
        """Direct lambda dict should be used if provided."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "vacuum_radius": 1.0},
            "mesh_quality": "coarse",
            "physics_params": {
                "lambda": {"object": 0.05, "vacuum": 0.1},  # Direct values
            },
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert "physics_refinement" in data
        # Should use the explicit values
        assert data["physics_refinement"]["lambda_per_region"]["object"] == 0.05
        assert data["physics_refinement"]["lambda_per_region"]["vacuum"] == 0.1
        # Minimum should be object (0.05 < 0.1)
        assert data["physics_refinement"]["lambda_min"] == 0.05

    @pytest.mark.asyncio
    async def test_alpha_without_density_no_refinement(self):
        """Alpha alone without density dict should not compute lambda."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "vacuum_radius": 1.0},
            "mesh_quality": "coarse",
            "physics_params": {
                "alpha": 1e18,
                # No density dict
            },
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        # No physics refinement should be applied
        assert "physics_refinement" not in data

    @pytest.mark.asyncio
    async def test_multi_region_refinement(self):
        """Test refinement with multiple dense regions (sphere + wall)."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "vacuum_radius": 1.0, "wall_thickness": 0.05},
            "mesh_quality": "coarse",
            "physics_params": {
                "alpha": 1e18,
                "density": {"object": 1e17, "wall": 1e17, "vacuum": 1.0},
                "n": 1,
            },
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert "physics_refinement" in data

        # Should have lambda for all regions
        lambda_per_region = data["physics_refinement"]["lambda_per_region"]
        assert "object" in lambda_per_region
        assert "wall" in lambda_per_region
        assert "vacuum" in lambda_per_region

        # Object and wall should have same lambda (same density)
        assert abs(lambda_per_region["object"] - lambda_per_region["wall"]) < 1e-10

        # Vacuum should have much larger lambda (lower density)
        assert lambda_per_region["vacuum"] > lambda_per_region["object"] * 100


class TestCreateMeshCustom2D:
    """Test custom_2d geometry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_custom_2d_with_points(self):
        """Create custom 2D shape with inline points."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "custom_2d",
            "params": {
                "points": [
                    [0.2, 0.0], [0.1, 0.173], [-0.1, 0.173],
                    [-0.2, 0.0], [-0.1, -0.173], [0.1, -0.173],
                ],
                "vacuum_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "custom_2d"
        assert "object_bounds" in data["domain_bounds"]

    @pytest.mark.asyncio
    async def test_custom_2d_missing_params(self):
        """Missing points/shape_file should error."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "custom_2d",
            "params": {
                "vacuum_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "MISSING_PARAMS"


class TestCreateMeshNotImplemented:
    """Test unimplemented geometries."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_not_implemented(self):
        """Unimplemented geometry should return error."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "ellipsoid_in_vacuum",
            "params": {
                "rx": 0.1,
                "ry": 0.2,
                "rz": 0.15,
                "vacuum_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "NOT_IMPLEMENTED"
