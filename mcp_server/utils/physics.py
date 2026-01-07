"""Physics functions for chameleon field calculations."""

import math


# Planck mass in eV (for M = M_pl / beta calculation)
M_PL_EV = 2.435e18 * 1e9  # 2.435e18 GeV in eV


def calculate_lambda_hat(alpha: float, n: int, rho_hat: float) -> float:
    """
    Calculate the dimensionless Compton wavelength.

    From SELCIE paper (arXiv:2110.11917):
    λ̂²(ρ̂) = α/(n+1) × ρ̂^{-(n+2)/(n+1)}

    Therefore:
    λ̂(ρ̂) = √(α/(n+1)) × ρ̂^{-(n+2)/(2(n+1))}

    Parameters
    ----------
    alpha : float
        Dimensionless α parameter.
    n : int
        Potential power.
    rho_hat : float
        Dimensionless density ρ̂ = ρ/ρ₀.

    Returns
    -------
    float
        Dimensionless Compton wavelength λ̂.
    """
    return math.sqrt(alpha / (n + 1)) * rho_hat ** (-(n + 2) / (2 * (n + 1)))


def classify_regime(
    lambda_at_rho_max: float, lambda_at_rho_min: float
) -> tuple[str, str, bool]:
    """
    Classify the screening regime based on Compton wavelengths.

    Parameters
    ----------
    lambda_at_rho_max : float
        Compton wavelength at maximum density.
    lambda_at_rho_min : float
        Compton wavelength at minimum density.

    Returns
    -------
    tuple[str, str, bool]
        (regime, explanation, selcie_needed)
    """
    if lambda_at_rho_max > 1 and lambda_at_rho_min > 1:
        return (
            "constant_field",
            f"λ̂(ρ_max) = {lambda_at_rho_max:.2e} >> 1 and "
            f"λ̂(ρ_min) = {lambda_at_rho_min:.2e} >> 1: "
            "field cannot respond to density variations anywhere. φ ≈ constant.",
            False,
        )
    elif lambda_at_rho_max < 0.1 and lambda_at_rho_min < 0.1:
        return (
            "adiabatic",
            f"λ̂(ρ_max) = {lambda_at_rho_max:.2e} << 1 and "
            f"λ̂(ρ_min) = {lambda_at_rho_min:.2e} << 1: "
            "field tracks local minimum everywhere. φ = ρ̂^{-1/(n+1)}.",
            False,
        )
    elif lambda_at_rho_max < 0.1 and lambda_at_rho_min >= 0.5:
        return (
            "thin_shell",
            f"λ̂(ρ_max) = {lambda_at_rho_max:.2e} << 1: field tracks density at high ρ. "
            f"λ̂(ρ_min) = {lambda_at_rho_min:.2e} ~ 1: transition occurs at low ρ. "
            "Expect thin-shell screening.",
            True,
        )
    else:
        return (
            "transition",
            f"λ̂(ρ_max) = {lambda_at_rho_max:.2e}, "
            f"λ̂(ρ_min) = {lambda_at_rho_min:.2e}: "
            "intermediate regime with complex screening behavior.",
            True,
        )
