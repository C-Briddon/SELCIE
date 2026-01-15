#!/usr/bin/env python3
"""Tests for solve tool."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
import pytest_asyncio

from utils.session import reset_session, get_session


class TestSolve:
    """Test solve tool."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    @pytest_asyncio.fixture
    async def mesh_id(self):
        """Create a mesh for testing."""
        from tools.create_mesh import handle as create_mesh

        result = await create_mesh({
            "geometry": "sphere_in_vacuum",
            "params": {
                "object_radius": 0.15,
                "domain_radius": 1.0,
            },
            "mesh_quality": "very_coarse",
        })

        data = json.loads(result[0].text)
        return data["mesh_id"]

    @pytest.mark.asyncio
    async def test_basic_solve(self, mesh_id):
        """Test basic solve with constant density."""
        from tools.solve import handle

        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": {
                "object": 1e6,
                "vacuum": 1.0,
            },
        })

        assert len(result) == 1
        data = json.loads(result[0].text)

        # Check no error
        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed: {data}")

        assert data["solution_id"] is not None
        assert data["mesh_id"] == mesh_id
        assert data["alpha"] == 1.0
        assert data["n"] == 1
        assert "field_stats" in data
        assert data["field_stats"]["min"] is not None
        assert data["field_stats"]["max"] is not None

    @pytest.mark.asyncio
    async def test_solve_with_expression(self, mesh_id):
        """Test solve with expression-based density."""
        from tools.solve import handle

        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": {
                "object": {"expression": "1e6 / (1 + (r/0.1)**2)"},
                "vacuum": 1.0,
            },
        })

        data = json.loads(result[0].text)

        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed: {data}")

        assert data["solution_id"] is not None
        assert "field_stats" in data

    @pytest.mark.asyncio
    async def test_solve_missing_mesh(self):
        """Test error when mesh doesn't exist."""
        from tools.solve import handle

        result = await handle({
            "mesh_id": "nonexistent",
            "alpha": 1.0,
            "density": {
                "object": 1e6,
                "vacuum": 1.0,
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_solve_missing_region(self, mesh_id):
        """Test error when density for a region is missing."""
        from tools.solve import handle

        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": {
                "object": 1e6,
                # Missing vacuum density
            },
        })

        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_solve_custom_id(self, mesh_id):
        """Test custom solution ID."""
        from tools.solve import handle

        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": {
                "object": 1e6,
                "vacuum": 1.0,
            },
            "custom_id": "my_solution",
        })

        data = json.loads(result[0].text)

        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed: {data}")

        assert data["solution_id"] == "my_solution"

    @pytest.mark.asyncio
    async def test_solve_stores_in_session(self, mesh_id):
        """Test that solve stores solution in session."""
        from tools.solve import handle

        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": {
                "object": 1e6,
                "vacuum": 1.0,
            },
        })

        data = json.loads(result[0].text)

        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed: {data}")

        session = get_session()
        solution_id = data["solution_id"]
        solution = session.get_solution(solution_id)

        assert solution is not None
        assert solution.mesh_id == mesh_id
        assert solution.alpha == 1.0
        assert solution.n == 1

    @pytest.mark.asyncio
    async def test_solve_method_selection(self, mesh_id):
        """Test auto method selection."""
        from tools.solve import handle

        # Low alpha should use picard
        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 0.1,
            "density": {
                "object": 1e6,
                "vacuum": 1.0,
            },
            "method": "auto",
        })

        data = json.loads(result[0].text)

        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed: {data}")

        assert data["method_used"] == "picard"

    @pytest.mark.asyncio
    async def test_solve_different_n(self, mesh_id):
        """Test solve with different potential power n."""
        from tools.solve import handle

        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "n": 2,
            "density": {
                "object": 1e6,
                "vacuum": 1.0,
            },
        })

        data = json.loads(result[0].text)

        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed: {data}")

        assert data["n"] == 2


class TestDensityFunctions:
    """Test density function creation."""

    def test_constant_density(self):
        """Test constant density function."""
        from utils.density import create_density_function

        func = create_density_function(1e6, "axial", 2)
        assert func([0.5, 0.5]) == 1e6

    def test_expression_density_axial(self):
        """Test expression density in axial coordinates."""
        from utils.density import create_density_function

        func = create_density_function(
            {"expression": "r + z"},
            "axial", 2
        )
        assert func([0.5, 0.3]) == 0.8

    def test_expression_density_cartesian(self):
        """Test expression density in Cartesian coordinates."""
        from utils.density import create_density_function

        func = create_density_function(
            {"expression": "x + y"},
            "none", 2
        )
        assert func([0.5, 0.3]) == 0.8

    def test_expression_invalid_variable(self):
        """Test that invalid variable in expression raises helpful error."""
        from utils.density import create_density_function
        import pytest

        # For spherical geometry with axial symmetry, 'x' is not valid
        with pytest.raises(ValueError) as exc_info:
            create_density_function(
                {"expression": "x + y"},
                "axial", 2, "sphere_in_vacuum"
            )

        error_msg = str(exc_info.value)
        assert "x" in error_msg  # mentions the invalid variable
        assert "r (spherical radius" in error_msg  # suggests valid variables
        assert "Available variables" in error_msg

    def test_expression_spherical_r_is_spherical(self):
        """Test that r is spherical radius for spherical geometries."""
        from utils.density import create_density_function

        # For sphere_in_vacuum, r should be spherical radius = sqrt(r_cyl^2 + z^2)
        func = create_density_function(
            {"expression": "r"},
            "axial", 2, "sphere_in_vacuum"
        )
        # r_cyl=0.3, z=0.4 -> r_spherical = 0.5
        result = func([0.3, 0.4])
        assert abs(result - 0.5) < 1e-10

    def test_expression_non_spherical_r_is_cylindrical(self):
        """Test that r is cylindrical radius for non-spherical geometries."""
        from utils.density import create_density_function

        # For ellipse_in_vacuum, r should be cylindrical radius = x[0]
        func = create_density_function(
            {"expression": "r"},
            "axial", 2, "ellipse_in_vacuum"
        )
        # r_cyl=0.3, z=0.4 -> r = 0.3 (cylindrical)
        result = func([0.3, 0.4])
        assert abs(result - 0.3) < 1e-10


class TestMethodSelection:
    """Test solver method selection."""

    def test_low_alpha_uses_picard(self):
        """Low alpha should use picard."""
        from tools.solve import _choose_method

        method, relax = _choose_method(0.1, "auto")
        assert method == "picard"
        assert relax == 1.0

    def test_medium_alpha_uses_relaxed_picard(self):
        """Medium alpha should use picard with relaxation."""
        from tools.solve import _choose_method

        method, relax = _choose_method(500, "auto")
        assert method == "picard"
        assert relax < 1.0

    def test_high_alpha_uses_strong_relaxation(self):
        """High alpha should use strong relaxation."""
        from tools.solve import _choose_method

        method, relax = _choose_method(2000, "auto")
        assert method == "picard"
        assert relax == 0.5

    def test_explicit_picard_no_relaxation(self):
        """Explicit picard should use no relaxation."""
        from tools.solve import _choose_method

        method, relax = _choose_method(500, "picard")
        assert method == "picard"
        assert relax == 1.0
