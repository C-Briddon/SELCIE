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
                "domain_radius": 1.0,
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
                # missing domain_radius
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
                "domain_radius": 1.0,
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
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
            "custom_id": "duplicate",
        })

        # Second mesh with same ID
        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.2, "domain_radius": 2.0},
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
                "domain_radius": 1.0,
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
            "params": {"domain_radius": 1.0},
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
            "params": {"domain_width": 2.0, "domain_height": 1.0},
            "mesh_quality": "coarse",
            # "none" means use default, which is "translation" for box_2d
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "box_2d"
        assert data["symmetry"] == "translation"


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
                "domain_radius": 1.0,
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
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
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
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
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
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
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
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
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
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
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
            "params": {"object_radius": 0.1, "domain_radius": 1.0, "wall_thickness": 0.05},
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
    """Test custom_2d_axial and custom_2d_translation geometries."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_custom_2d_axial_with_points(self):
        """Create custom 2D axisymmetric shape with inline points."""
        from tools.create_mesh import handle

        # Half-hexagon with r >= 0 (revolved to create 3D shape)
        result = await handle({
            "geometry": "custom_2d_axial",
            "params": {
                "points": [
                    [0.0, 0.2], [0.173, 0.1], [0.173, -0.1], [0.0, -0.2],
                ],
                "domain_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "custom_2d_axial"
        assert data["symmetry"] == "axial"
        assert "object_bounds" in data["domain_bounds"]

    @pytest.mark.asyncio
    async def test_custom_2d_translation_with_points(self):
        """Create custom 2D translation shape with inline points."""
        from tools.create_mesh import handle

        # Full hexagon (extruded in z)
        result = await handle({
            "geometry": "custom_2d_translation",
            "params": {
                "points": [
                    [0.2, 0.0], [0.1, 0.173], [-0.1, 0.173],
                    [-0.2, 0.0], [-0.1, -0.173], [0.1, -0.173],
                ],
                "domain_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "custom_2d_translation"
        assert data["symmetry"] == "translation"
        assert "object_bounds" in data["domain_bounds"]

    @pytest.mark.asyncio
    async def test_custom_2d_axial_rejects_negative_r(self):
        """custom_2d_axial should reject points with r < 0."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "custom_2d_axial",
            "params": {
                "points": [
                    [0.2, 0.0], [-0.1, 0.173],  # Negative r value
                ],
                "domain_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert "r < 0" in data["error"]["message"] or "r values >= 0" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_custom_2d_missing_params(self):
        """Missing points/shape_file should error."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "custom_2d_axial",
            "params": {
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "MISSING_PARAMS"


class TestCreateMeshCustomStep:
    """Test custom_step geometry (STEP file import)."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_session()

    @pytest.mark.asyncio
    async def test_custom_step_basic(self):
        """Import a STEP file and create mesh."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "custom_step",
            "params": {
                "step_file": "tests/test_data/eotwash_disks.step",
                "domain_radius": 3.0,
            },
            "mesh_quality": "very_coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data, f"Unexpected error: {data}"
        assert data["geometry"] == "custom_step"
        assert data["symmetry"] == "none"
        assert data["dimension"] == 3
        # Multi-object STEP files get object_0, object_1, etc.
        assert "object_0" in data["regions"]
        assert "object_1" in data["regions"]
        assert "vacuum" in data["regions"]
        assert "object_bounds" in data["domain_bounds"]
        assert data["domain_bounds"]["imported_file"] == "eotwash_disks.step"

    @pytest.mark.asyncio
    async def test_custom_step_single_object(self):
        """Single-object STEP file uses 'object' name for backward compatibility."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "custom_step",
            "params": {
                "step_file": "tests/test_data/unit_cube.step",
            },
            "mesh_quality": "very_coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data, f"Unexpected error: {data}"
        # Single object uses "object" name for backward compatibility
        assert "object" in data["regions"]
        assert "vacuum" in data["regions"]
        assert len(data["regions"]) == 2

    @pytest.mark.asyncio
    async def test_custom_step_multi_region(self):
        """Multi-region STEP file creates separate object regions."""
        from tools.create_mesh import handle as create_mesh

        # Create mesh with two separate objects
        result = await create_mesh({
            "geometry": "custom_step",
            "params": {
                "step_file": "tests/test_data/eotwash_disks.step",
            },
            "mesh_quality": "very_coarse",
        })

        mesh_data = json.loads(result[0].text)
        assert "error" not in mesh_data, f"Mesh error: {mesh_data}"

        # Verify we have two object regions plus vacuum
        assert "object_0" in mesh_data["regions"]
        assert "object_1" in mesh_data["regions"]
        assert "vacuum" in mesh_data["regions"]
        assert len(mesh_data["regions"]) == 3

    @pytest.mark.asyncio
    async def test_custom_step_file_not_found(self):
        """Missing STEP file should error."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "custom_step",
            "params": {
                "step_file": "nonexistent.step",
                "domain_radius": 2.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert "not found" in data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_custom_step_missing_params(self):
        """Missing required params (step_file) should error."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "custom_step",
            "params": {
                # missing step_file - this is required
                "domain_radius": 1.0,
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
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "NOT_IMPLEMENTED"


class TestCreateMeshSphereInProfile:
    """Test sphere_in_profile geometry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest.mark.asyncio
    async def test_basic_sphere_in_profile(self):
        """Create a sphere in density profile mesh."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_profile",
            "params": {
                "object_radius": 0.1,
                "domain_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["geometry"] == "sphere_in_profile"
        assert data["symmetry"] == "axial"
        assert data["dimension"] == 2
        assert "sphere" in data["regions"]
        assert "background" in data["regions"]
        assert data["regions"]["sphere"] == 0  # First subdomain
        assert data["regions"]["background"] == 1  # Second subdomain

    @pytest.mark.asyncio
    async def test_offset_sphere(self):
        """Sphere displaced along z-axis."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_profile",
            "params": {
                "object_radius": 0.1,
                "domain_radius": 2.0,
                "center_z": 0.5,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["regions"]["sphere"] == 0
        assert data["regions"]["background"] == 1

    @pytest.mark.asyncio
    async def test_missing_params(self):
        """Missing required parameters should error."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_profile",
            "params": {
                "object_radius": 0.1,
                # missing domain_radius
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "MISSING_PARAMS"


class TestCreateMeshParallelPlates:
    """Test parallel_plates geometry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest.mark.asyncio
    async def test_basic_parallel_plates(self):
        """Create parallel plates mesh."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "parallel_plates",
            "params": {
                "plate_separation": 1.0,
                "plate_thickness": 0.1,
            },
            "mesh_quality": "coarse",
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        assert "error" not in data
        assert data["geometry"] == "parallel_plates"
        assert data["symmetry"] == "translation"
        assert data["dimension"] == 2
        assert "vacuum" in data["regions"]
        assert "plate" in data["regions"]
        # Region indices follow creation order
        assert data["regions"]["vacuum"] == 0
        assert data["regions"]["plate"] == 1

    @pytest.mark.asyncio
    async def test_parallel_plates_with_domain_height(self):
        """Parallel plates with custom domain height."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "parallel_plates",
            "params": {
                "plate_separation": 1.0,
                "plate_thickness": 0.1,
                "domain_height": 2.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["domain_bounds"]["y_max"] == 2.0

    @pytest.mark.asyncio
    async def test_parallel_plates_missing_params(self):
        """Missing required parameters should error."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "parallel_plates",
            "params": {
                "plate_separation": 1.0,
                # missing plate_thickness
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "MISSING_PARAMS"


class TestParameterValidation:
    """Test geometry parameter validation."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest.mark.asyncio
    async def test_negative_radius_rejected(self):
        """Negative radius should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": -0.1,  # Invalid
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"
        assert "object_radius must be positive" in str(data["error"]["details"])

    @pytest.mark.asyncio
    async def test_zero_radius_rejected(self):
        """Zero radius should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.0,  # Invalid
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_sphere_radius_larger_than_domain(self):
        """Object radius >= domain radius should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 1.5,  # Larger than domain
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"
        assert "must be less than" in str(data["error"]["details"])

    @pytest.mark.asyncio
    async def test_ellipse_semi_axes_larger_than_domain(self):
        """Ellipse semi-axes exceeding domain should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "ellipse_in_vacuum",
            "params": {
                "rx": 1.5,  # Larger than domain
                "ry": 0.5,
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_shell_inner_larger_than_outer(self):
        """Shell inner radius >= outer radius should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "shell_in_vacuum",
            "params": {
                "inner_radius": 0.5,
                "outer_radius": 0.3,  # Smaller than inner
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"
        assert "must be less than" in str(data["error"]["details"])

    @pytest.mark.asyncio
    async def test_shell_outer_larger_than_domain(self):
        """Shell outer radius >= domain radius should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "shell_in_vacuum",
            "params": {
                "inner_radius": 0.5,
                "outer_radius": 1.5,  # Larger than domain
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_cylinder_radius_larger_than_domain(self):
        """Cylinder radius >= domain radius should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "cylinder_in_vacuum",
            "params": {
                "object_radius": 1.5,  # Larger than domain
                "object_height": 0.5,
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_cylinder_height_larger_than_domain(self):
        """Cylinder height/2 >= domain radius should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "cylinder_in_vacuum",
            "params": {
                "object_radius": 0.2,
                "object_height": 3.0,  # height/2 = 1.5 > domain
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_two_spheres_overlapping(self):
        """Two spheres that overlap should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "two_spheres",
            "params": {
                "radius_1": 0.3,
                "radius_2": 0.3,
                "separation": 0.4,  # Spheres overlap (0.3 + 0.3 = 0.6 > 0.4)
                "domain_radius": 2.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"
        assert "overlap" in str(data["error"]["details"]).lower()

    @pytest.mark.asyncio
    async def test_two_spheres_outside_domain(self):
        """Two spheres extending outside domain should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "two_spheres",
            "params": {
                "radius_1": 0.2,
                "radius_2": 0.2,
                "separation": 2.0,  # center + radius extends past domain
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_sphere_near_wall_intersects_wall(self):
        """Sphere intersecting wall should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_near_wall",
            "params": {
                "object_radius": 0.3,
                "wall_distance": 0.2,  # Sphere would intersect wall
                "wall_thickness": 0.1,
                "domain_radius": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"
        assert "intersect" in str(data["error"]["details"]).lower() or "wall_distance" in str(data["error"]["details"])

    @pytest.mark.asyncio
    async def test_sphere_in_profile_outside_domain(self):
        """Sphere in profile extending outside domain should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_profile",
            "params": {
                "object_radius": 0.3,
                "domain_radius": 1.0,
                "center_z": 0.9,  # 0.9 + 0.3 = 1.2 > domain_radius
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_parallel_plates_domain_too_small(self):
        """Parallel plates with domain_height < plate_separation should be rejected."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "parallel_plates",
            "params": {
                "plate_separation": 2.0,
                "plate_thickness": 0.1,
                "domain_height": 1.0,  # Smaller than separation
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "INVALID_PARAMS"

    @pytest.mark.asyncio
    async def test_valid_params_accepted(self):
        """Valid parameters should be accepted."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.1,  # Valid: 0.1 < 1.0
                "domain_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["geometry"] == "sphere_in_vacuum"


class TestFixedSymmetry:
    """Test that all geometries have fixed symmetry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest.mark.asyncio
    async def test_sphere_has_fixed_axial_symmetry(self):
        """sphere_in_vacuum always uses axial symmetry."""
        from tools.create_mesh import handle

        result = await handle({
            "geometry": "sphere_in_vacuum",
            "params": {"object_radius": 0.1, "domain_radius": 1.0},
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["symmetry"] == "axial"

    @pytest.mark.asyncio
    async def test_custom_2d_has_fixed_symmetry(self):
        """custom_2d_axial and custom_2d_translation have fixed symmetry."""
        from tools.create_mesh import handle

        # custom_2d_axial always uses axial symmetry
        result = await handle({
            "geometry": "custom_2d_axial",
            "params": {
                "points": [[0.0, 0.2], [0.2, 0.0], [0.0, -0.2]],
                "domain_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["symmetry"] == "axial"

        reset_session()

        # custom_2d_translation always uses translation symmetry
        result = await handle({
            "geometry": "custom_2d_translation",
            "params": {
                "points": [[0.0, 0.2], [0.2, 0.0], [0.0, -0.2]],
                "domain_radius": 1.0,
            },
            "mesh_quality": "coarse",
        })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["symmetry"] == "translation"
