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


class TestSolveDirichletBC:
    """Test the dirichlet_bc outer-boundary condition."""

    UNSCREENED = {"object": 10.0, "vacuum": 1.0}
    SCREENED = {"object": 1e6, "vacuum": 1.0}

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset session before each test."""
        reset_session()

    async def _make_mesh(self, quality="very_coarse", geometry="sphere_in_vacuum"):
        from tools.create_mesh import handle as create_mesh

        params = {"object_radius": 0.15, "domain_radius": 1.0}
        if geometry == "box_2d":
            params = {"domain_width": 1.0, "domain_height": 1.0}
        result = await create_mesh({
            "geometry": geometry,
            "params": params,
            "mesh_quality": quality,
        })
        return json.loads(result[0].text)["mesh_id"]

    async def _boundary_field(self, solution_id):
        from tools.evaluate import handle as evaluate

        result = await evaluate({
            "solution_id": solution_id,
            "mode": "points",
            "params": {"coordinates": [[0.0, 0.999]]},
            "quantities": ["field"],
        })
        return json.loads(result[0].text)["data"]["field"][0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("quality", ["very_coarse", "medium"])
    async def test_unscreened_pin_attained(self, quality):
        """The pinned value must be attained at the outer boundary.

        Covers both mesh qualities: coarse meshes regress the
        check_midpoint=False facet labeling (chord midpoints sag inside
        the radius tolerance and previously left the BC silently unapplied).
        """
        from tools.solve import handle

        mesh_id = await self._make_mesh(quality)
        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.UNSCREENED),
            "dirichlet_bc": 0.5,
        })
        data = json.loads(result[0].text)
        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed: {data}")

        assert data["status"] == "converged"
        assert data["initial_guess"] == "boundary"  # default switches with BC
        assert data["dirichlet_bc"] == 0.5
        assert data["warnings"] is None
        # Natural-BC equilibrium is ~1; the pin must hold the boundary at 0.5
        assert abs(await self._boundary_field(data["solution_id"]) - 0.5) < 0.01

    @pytest.mark.asyncio
    async def test_screened_cold_start_warns_unphysical(self):
        """A screened cold start lands on the negative branch and must warn."""
        from tools.solve import handle

        mesh_id = await self._make_mesh()
        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 0.1,
            "density": dict(self.SCREENED),
            "dirichlet_bc": 1.0,
        })
        data = json.loads(result[0].text)

        if data.get("status") != "converged" or data["field_stats"]["min"] > 0:
            pytest.skip("Negative-branch scenario not reproduced on this mesh")
        assert any("unphysical" in w for w in data["warnings"])
        assert any("initial_guess='previous'" in w for w in data["warnings"])

    @pytest.mark.asyncio
    async def test_screened_two_step_recipe(self):
        """Natural solve then re-solve with the BC stays on the physical branch."""
        from tools.solve import handle

        mesh_id = await self._make_mesh()
        base = {"mesh_id": mesh_id, "alpha": 0.1, "density": dict(self.SCREENED)}

        natural = json.loads((await handle(base))[0].text)
        assert natural["status"] == "converged"

        pinned = json.loads((await handle({
            **base,
            "dirichlet_bc": 1.0,
            "initial_guess": "previous",
        }))[0].text)

        assert pinned["status"] == "converged"
        assert pinned["initial_guess_solution_id"] == natural["solution_id"]
        assert pinned["warnings"] is None
        assert pinned["field_stats"]["min"] > 0
        assert abs(await self._boundary_field(pinned["solution_id"]) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_geometry_without_domain_radius_rejected(self):
        """Geometries with no circular outer boundary must be rejected."""
        from tools.solve import handle

        mesh_id = await self._make_mesh(geometry="box_2d")
        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": {"domain": 1.0},
            "dirichlet_bc": 1.0,
        })
        data = json.loads(result[0].text)
        assert data["error"]["code"] == "DIRICHLET_UNSUPPORTED"

    @pytest.mark.asyncio
    async def test_non_positive_bc_rejected(self):
        """dirichlet_bc must be a positive number."""
        from tools.solve import handle

        mesh_id = await self._make_mesh()
        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.UNSCREENED),
            "dirichlet_bc": -1.0,
        })
        data = json.loads(result[0].text)
        assert data["error"]["code"] == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_boundary_guess_requires_bc(self):
        """initial_guess='boundary' without dirichlet_bc must error."""
        from tools.solve import handle

        mesh_id = await self._make_mesh()
        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.UNSCREENED),
            "initial_guess": "boundary",
        })
        data = json.loads(result[0].text)
        assert data["error"]["code"] == "INVALID_PARAMETER"


class TestSolveInitialGuess:
    """Test initial_guess strategies."""

    DENSITY = {"object": 1e6, "vacuum": 1.0}

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
    async def test_adiabatic_initial_guess_smooth_profile(self):
        """Adiabatic initial guess should converge faster for smooth densities."""
        from tools.create_mesh import handle as create_mesh
        from tools.solve import handle

        result = await create_mesh({
            "geometry": "sphere_domain",
            "params": {"domain_radius": 1.0},
            "mesh_quality": "very_coarse",
        })
        mesh_id = json.loads(result[0].text)["mesh_id"]

        density = {"domain": {"expression": "1e4 / (1 + (r/0.2)**2)**2 + 1"}}

        iterations = {}
        for guess in ["constant", "adiabatic"]:
            result = await handle({
                "mesh_id": mesh_id,
                "alpha": 1e-2,
                "density": density,
                "initial_guess": guess,
            })
            data = json.loads(result[0].text)
            if "error" in data and data.get("status") == "failed":
                pytest.fail(f"Solve with initial_guess={guess} failed: {data}")
            assert data["status"] == "converged"
            assert data["initial_guess"] == guess
            iterations[guess] = data["iterations"]

        # The adiabatic start is close to the solution in this regime, so it
        # must not take more iterations than the constant start
        assert iterations["adiabatic"] <= iterations["constant"]

    @pytest.mark.asyncio
    async def test_diverged_status_reported(self, mesh_id):
        """A non-finite du_norm should be reported as 'diverged', not NaN JSON."""
        from tools.solve import handle

        # Adiabatic start with a discontinuous high-contrast density diverges
        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.DENSITY),
            "initial_guess": "adiabatic",
        })

        data = json.loads(result[0].text, parse_constant=pytest.fail)
        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed outright: {data}")

        if data["status"] == "converged":
            pytest.skip("Solver converged; divergence scenario not reproduced")

        assert data["status"] == "diverged"
        assert data["final_du_norm"] is None
        assert "suggestion" in data

    @pytest.mark.asyncio
    async def test_previous_without_prior_solution_errors(self, mesh_id):
        """initial_guess='previous' with no prior solution should error."""
        from tools.solve import handle

        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.DENSITY),
            "initial_guess": "previous",
        })

        data = json.loads(result[0].text)
        assert data["error"]["code"] == "INITIAL_GUESS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_previous_uses_latest_solution_on_mesh(self, mesh_id):
        """initial_guess='previous' should start from the prior solution."""
        from tools.solve import handle

        first = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.DENSITY),
        })
        first_data = json.loads(first[0].text)
        if "error" in first_data and first_data.get("status") == "failed":
            pytest.fail(f"Initial solve failed: {first_data}")

        second = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.DENSITY),
            "initial_guess": "previous",
        })
        data = json.loads(second[0].text)
        if "error" in data and data.get("status") == "failed":
            pytest.fail(f"Solve failed: {data}")

        assert data["initial_guess"] == "previous"
        assert data["initial_guess_solution_id"] == first_data["solution_id"]
        # Starting from a converged field should not take more iterations
        assert data["iterations"] <= first_data["iterations"]

    @pytest.mark.asyncio
    async def test_previous_deg_v_mismatch_errors(self, mesh_id):
        """initial_guess='previous' with a different deg_V should error."""
        from tools.solve import handle

        await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.DENSITY),
            "deg_V": 2,
        })

        result = await handle({
            "mesh_id": mesh_id,
            "alpha": 1.0,
            "density": dict(self.DENSITY),
            "initial_guess": "previous",
            "deg_V": 1,
        })

        data = json.loads(result[0].text)
        assert data["error"]["code"] == "INITIAL_GUESS_DEGREE_MISMATCH"


class TestSolveUtilities:
    """Test pure solve helpers."""

    def test_prepare_density_spec_does_not_mutate_input(self):
        """Region expansion should leave caller-provided density dict unchanged."""
        from tools.solve import _prepare_density_spec

        density = {"object": 2.0, "vacuum": 1.0}
        regions = {
            "object_0": 0,
            "object_1": 1,
            "vacuum": 2,
            "measuring_boundary": 3,
        }

        prepared = _prepare_density_spec(density, regions)

        assert density == {"object": 2.0, "vacuum": 1.0}
        assert prepared == {
            "object_0": 2.0,
            "object_1": 2.0,
            "vacuum": 1.0,
            "measuring_boundary": 1.0,
        }

