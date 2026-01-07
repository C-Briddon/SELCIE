#!/usr/bin/env python3
"""Tests for calculate_physical_parameters."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from utils.physics import M_PL_EV, calculate_lambda_hat, classify_regime
from utils.units import get_astropy_density_unit, get_astropy_length_unit

from SELCIE.Misc import alpha_calculator_chameleon


class TestLabExperiment:
    """Test case: Lab experiment with aluminum sphere."""

    @pytest.fixture
    def params(self):
        beta = 1e8
        Lambda_eV = 2.4e-3
        n = 1
        rho_0 = 2.7  # g/cm^3
        L = 0.01  # m
        rho_max = 2.7  # g/cm^3
        rho_min = 1e-10  # g/cm^3

        M_eV = M_PL_EV / beta
        rho_unit = get_astropy_density_unit("g/cm^3")
        L_unit = get_astropy_length_unit("m")

        alpha = alpha_calculator_chameleon(
            n=n, M=M_eV, Lam=Lambda_eV, p0=rho_0, L=L,
            p0_NonEVUnits=rho_unit, L_NonEVUnits=L_unit
        )

        rho_max_hat = rho_max / rho_0
        rho_min_hat = rho_min / rho_0
        lambda_max = calculate_lambda_hat(alpha, n, rho_max_hat)
        lambda_min = calculate_lambda_hat(alpha, n, rho_min_hat)

        return {
            "alpha": alpha,
            "lambda_max": lambda_max,
            "lambda_min": lambda_min,
        }

    def test_alpha_value(self, params):
        """Alpha should be ~3.3e-16 for lab experiment."""
        assert params["alpha"] == pytest.approx(3.325586e-16, rel=1e-4)

    def test_lambda_at_rho_max(self, params):
        """Lambda at rho_max should be << 1."""
        assert params["lambda_max"] < 0.1

    def test_lambda_at_rho_min(self, params):
        """Lambda at rho_min should be ~ 1."""
        assert 0.1 < params["lambda_min"] < 10

    def test_regime_is_thin_shell(self, params):
        """Lab experiment should be in thin_shell regime."""
        regime, _, selcie_needed = classify_regime(
            params["lambda_max"], params["lambda_min"]
        )
        assert regime == "thin_shell"
        assert selcie_needed is True


class TestSolarChameleon:
    """Test case: Solar chameleon (beta=1)."""

    @pytest.fixture
    def params(self):
        beta = 1
        Lambda_eV = 2.4e-3
        n = 1
        rho_0 = 150  # g/cm^3
        L = 6.96e8  # m (R_sun)
        rho_max = 150  # g/cm^3
        rho_min = 1e-6  # g/cm^3

        M_eV = M_PL_EV / beta
        rho_unit = get_astropy_density_unit("g/cm^3")
        L_unit = get_astropy_length_unit("m")

        alpha = alpha_calculator_chameleon(
            n=n, M=M_eV, Lam=Lambda_eV, p0=rho_0, L=L,
            p0_NonEVUnits=rho_unit, L_NonEVUnits=L_unit
        )

        rho_max_hat = rho_max / rho_0
        rho_min_hat = rho_min / rho_0
        lambda_max = calculate_lambda_hat(alpha, n, rho_max_hat)
        lambda_min = calculate_lambda_hat(alpha, n, rho_min_hat)

        return {
            "alpha": alpha,
            "lambda_max": lambda_max,
            "lambda_min": lambda_min,
        }

    def test_alpha_value(self, params):
        """Alpha should be ~1.66e-28 for solar case."""
        assert params["alpha"] == pytest.approx(1.657901e-28, rel=1e-4)

    def test_lambdas_both_small(self, params):
        """Both lambdas should be << 1 for adiabatic regime."""
        assert params["lambda_max"] < 0.1
        assert params["lambda_min"] < 0.1

    def test_regime_is_adiabatic(self, params):
        """Solar chameleon should be in adiabatic regime."""
        regime, _, selcie_needed = classify_regime(
            params["lambda_max"], params["lambda_min"]
        )
        assert regime == "adiabatic"
        assert selcie_needed is False


class TestNFWDimensionless:
    """Test case: NFW galaxy halo with dimensionless alpha values."""

    def test_small_alpha_is_adiabatic(self):
        """Small alpha (1e-9) should give adiabatic regime."""
        alpha = 1e-9
        n = 1
        rho_max_hat = 1e6
        rho_min_hat = 1.0

        lambda_max = calculate_lambda_hat(alpha, n, rho_max_hat)
        lambda_min = calculate_lambda_hat(alpha, n, rho_min_hat)
        regime, _, selcie_needed = classify_regime(lambda_max, lambda_min)

        assert lambda_max < 0.1
        assert lambda_min < 0.1
        assert regime == "adiabatic"
        assert selcie_needed is False

    def test_large_alpha_is_transition(self):
        """Large alpha (1e9) should give transition regime."""
        alpha = 1e9
        n = 1
        rho_max_hat = 1e6
        rho_min_hat = 1.0

        lambda_max = calculate_lambda_hat(alpha, n, rho_max_hat)
        lambda_min = calculate_lambda_hat(alpha, n, rho_min_hat)
        regime, _, selcie_needed = classify_regime(lambda_max, lambda_min)

        assert 0.1 < lambda_max < 10  # ~ 1
        assert lambda_min > 1  # >> 1
        assert regime == "transition"
        assert selcie_needed is True
