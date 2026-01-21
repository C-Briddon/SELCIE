"""
calculate_physical_parameters tool.

Convert physical chameleon parameters to SELCIE's dimensionless α and assess
the screening regime using the Compton wavelength criterion.
"""

import json
import math
from typing import Any

from mcp.types import Tool, TextContent

from astropy import units

from SELCIE.Misc import alpha_calculator_chameleon, conv_fifth_force_chameleon

from utils.physics import M_PL_EV, calculate_lambda_hat, classify_regime
from utils.units import get_astropy_density_unit, get_astropy_length_unit

# Standard gravity in m/s^2
G_STANDARD = 9.80665


TOOL_DEFINITION = Tool(
    name="calculate_physical_parameters",
    description=(
        "Convert physical chameleon parameters to SELCIE's dimensionless α and assess "
        "the screening regime using the Compton wavelength criterion. Use this first to "
        "determine if SELCIE is needed or if analytic solutions suffice. Returns the "
        "dimensionless Compton wavelength λ̂(ρ̂) = √(α/(n+1)) × ρ̂^{-(n+2)/(2(n+1))} at "
        "the density extremes. Since λ̂ is in units of L, compare to 1: λ̂ << 1 means adiabatic "
        "(field tracks ρ̂^{-1/(n+1)}), λ̂ >> 1 means field is set by boundaries, λ̂ ~ 1 is the "
        "transition region where SELCIE is needed. Also returns conversion factors: "
        "grad_to_acceleration_g converts dimensionless ∇φ to acceleration in units of g; "
        "mass_scale_kg, force_scale_N, and torque_scale_Nm convert integrate mode outputs to "
        "physical mass (kg), force (N), and torque (N·m)."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "beta": {
                "type": "number",
                "description": (
                    "Matter coupling strength β. M = M_pl/β where M_pl is Planck mass. "
                    "Typical values: β=1 (gravitational strength), β=10^6-10^8 (lab experiments)."
                ),
            },
            "Lambda_eV": {
                "type": "number",
                "description": "Energy scale Λ in eV. Default 2.4e-3 (dark energy scale).",
                "default": 2.4e-3,
            },
            "n": {
                "type": "integer",
                "description": "Potential power in V(φ) = Λ⁴(1 + Λⁿ/φⁿ). Default 1.",
                "default": 1,
            },
            "rho_0": {
                "type": "number",
                "description": (
                    "Reference density scale for non-dimensionalization. Enters α as α ∝ 1/ρ₀. "
                    "Optional: defaults to rho_min if provided. For best solver convergence, use "
                    "the lowest density (e.g., vacuum) or an intermediate value."
                ),
            },
            "rho_0_units": {
                "type": "string",
                "enum": ["g/cm^3", "kg/m^3", "eV^4", "M_sun/kpc^3", "GeV^4"],
                "description": "Units of rho_0.",
            },
            "L": {
                "type": "number",
                "description": (
                    "Characteristic length scale of the system (e.g., object radius, "
                    "domain size). Enters the α calculation as α ∝ 1/L²."
                ),
            },
            "L_units": {
                "type": "string",
                "enum": ["m", "cm", "km", "R_sun", "kpc", "Mpc", "AU"],
                "description": "Units of L.",
            },
            "rho_max": {
                "type": "number",
                "description": (
                    "Maximum density in system, in same units as rho_0. "
                    "Used for regime estimation via Compton wavelength."
                ),
            },
            "rho_min": {
                "type": "number",
                "description": (
                    "Minimum density in system, in same units as rho_0. "
                    "Used for regime estimation via Compton wavelength."
                ),
            },
        },
        "required": ["beta", "rho_0_units", "L", "L_units"],
    },
)


async def handle(args: dict[str, Any]) -> list[TextContent]:
    """Handle calculate_physical_parameters tool call."""

    # Extract required parameters
    beta = args["beta"]
    rho_0_units = args["rho_0_units"]
    L = args["L"]
    L_units = args["L_units"]

    # Extract optional parameters with defaults
    Lambda_eV = args.get("Lambda_eV", 2.4e-3)
    n = args.get("n", 1)
    rho_max = args.get("rho_max")
    rho_min = args.get("rho_min")
    rho_0 = args.get("rho_0")

    # Default rho_0 to rho_min if not provided
    if rho_0 is None:
        if rho_min is not None:
            rho_0 = rho_min
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": {
                    "code": "MISSING_PARAMETER",
                    "message": "Either rho_0 or rho_min must be provided",
                }
            }, indent=2))]

    # Validate inputs
    if beta <= 0:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "INVALID_PARAMETER",
                "message": "beta must be positive",
            }
        }, indent=2))]

    if rho_0 <= 0:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "INVALID_PARAMETER",
                "message": "rho_0 must be positive",
            }
        }, indent=2))]

    if L <= 0:
        return [TextContent(type="text", text=json.dumps({
            "error": {
                "code": "INVALID_PARAMETER",
                "message": "L must be positive",
            }
        }, indent=2))]

    # Calculate M = M_pl / beta (in eV)
    M_eV = M_PL_EV / beta

    # Get astropy units
    rho_unit = get_astropy_density_unit(rho_0_units)
    L_unit = get_astropy_length_unit(L_units)

    # Handle eV^4 and GeV^4 density units (not in astropy)
    if rho_0_units == "eV^4":
        alpha = alpha_calculator_chameleon(
            n=n, M=M_eV, Lam=Lambda_eV, p0=rho_0, L=L,
            L_NonEVUnits=L_unit
        )
        force_conversion = conv_fifth_force_chameleon(
            n=n, M=M_eV, Lam=Lambda_eV, p0=rho_0, L=L,
            L_NonEVUnits=L_unit
        )
    elif rho_0_units == "GeV^4":
        rho_0_eV4 = rho_0 * 1e36
        alpha = alpha_calculator_chameleon(
            n=n, M=M_eV, Lam=Lambda_eV, p0=rho_0_eV4, L=L,
            L_NonEVUnits=L_unit
        )
        force_conversion = conv_fifth_force_chameleon(
            n=n, M=M_eV, Lam=Lambda_eV, p0=rho_0_eV4, L=L,
            L_NonEVUnits=L_unit
        )
    else:
        alpha = alpha_calculator_chameleon(
            n=n, M=M_eV, Lam=Lambda_eV, p0=rho_0, L=L,
            p0_NonEVUnits=rho_unit, L_NonEVUnits=L_unit
        )
        force_conversion = conv_fifth_force_chameleon(
            n=n, M=M_eV, Lam=Lambda_eV, p0=rho_0, L=L,
            p0_NonEVUnits=rho_unit, L_NonEVUnits=L_unit
        )

    # Calculate mass and force scales in SI units
    # Convert rho_0 to kg/m^3 and L to m for SI output
    if rho_0_units == "eV^4":
        # eV^4 to kg/m^3: complex conversion via natural units
        # ρ [kg/m³] = ρ [eV⁴] × (eV/c²)/(ℏc)³
        # Using: 1 eV⁴ ≈ 1.324e-73 kg/m³ (from constants)
        eV4_to_kg_m3 = 1.324e-73
        rho_0_si = rho_0 * eV4_to_kg_m3
    elif rho_0_units == "GeV^4":
        eV4_to_kg_m3 = 1.324e-73
        rho_0_si = rho_0 * 1e36 * eV4_to_kg_m3
    else:
        rho_0_si = (rho_0 * rho_unit).to(units.kg / units.m**3).value

    L_si = (L * L_unit).to(units.m).value

    # Mass scale: M_physical = mass_scale × mass_rescaled [kg]
    mass_scale = rho_0_si * L_si**3

    # Force scale: F_physical = force_scale × force_rescaled [N]
    # This is ρ₀ × L³ × g × conv_fifth_force_chameleon
    force_scale = mass_scale * G_STANDARD * force_conversion

    # Torque scale: τ_physical = torque_scale × torque_rescaled [N·m]
    # Torque has dimensions [Force × Length]
    torque_scale = force_scale * L_si

    # Build result
    result: dict[str, Any] = {
        "rho_0": rho_0,
        "rho_0_units": rho_0_units,
        "rho_0_note": (
            "Reference density for non-dimensionalization. "
            "For solve tool: use ρ̂ = ρ_physical / rho_0 as density values."
        ),
        "L": L,
        "L_units": L_units,
        "L_note": (
            "Reference length for non-dimensionalization. "
            "For mesh/solve tools: use x̂ = x_physical / L as coordinates. All mesh distances and step files should be in units of L."
        ),
        "alpha": alpha,
        "n": n,
        "grad_to_acceleration_g": force_conversion,
        "grad_to_acceleration_note": "Multiply dimensionless grad(phi) by this to get acceleration in units of g (9.81 m/s²)",
        "mass_scale_kg": mass_scale,
        "force_scale_N": force_scale,
        "torque_scale_Nm": torque_scale,
        "integral_scaling_note": (
            "For evaluate(..., mode='integrate'): "
            "M_physical[kg] = mass_scale_kg × mass, "
            "F_physical[N] = force_scale_N × F, "
            "τ_physical[N·m] = torque_scale_Nm × τ"
        ),
    }

    # If rho_max/rho_min not provided, return formulas
    if rho_max is None or rho_min is None:
        lambda_coeff = math.sqrt(alpha / (n + 1))
        phi_exp = -1 / (n + 1)
        lambda_exp = -(n + 2) / (2 * (n + 1))

        result["formulas"] = {
            "phi_min": f"φ̂_min(ρ̂) = ρ̂^{{{phi_exp:.4f}}}",
            "lambda": f"λ̂(ρ̂) = {lambda_coeff:.4e} × ρ̂^{{{lambda_exp:.4f}}}",
        }
        result["regime"] = None
        result["regime_note"] = "Provide rho_max and rho_min for regime estimation"
    else:
        # Calculate dimensionless densities
        rho_max_hat = rho_max / rho_0
        rho_min_hat = rho_min / rho_0

        # Calculate phi_min at density extremes
        phi_min_at_rho_max = rho_max_hat ** (-1 / (n + 1))
        phi_min_at_rho_min = rho_min_hat ** (-1 / (n + 1))

        # Calculate Compton wavelength at density extremes
        lambda_at_rho_max = calculate_lambda_hat(alpha, n, rho_max_hat)
        lambda_at_rho_min = calculate_lambda_hat(alpha, n, rho_min_hat)

        result["at_rho_max"] = {
            "rho_hat": rho_max_hat,
            "phi_min": phi_min_at_rho_max,
            "lambda": lambda_at_rho_max,
        }
        result["at_rho_min"] = {
            "rho_hat": rho_min_hat,
            "phi_min": phi_min_at_rho_min,
            "lambda": lambda_at_rho_min,
        }

        # Classify regime
        regime, explanation, selcie_needed = classify_regime(
            lambda_at_rho_max, lambda_at_rho_min
        )

        result["regime"] = regime
        result["regime_explanation"] = explanation
        result["selcie_needed"] = selcie_needed

        if selcie_needed:
            result["recommended_action"] = f"Use SELCIE solver with α={alpha:.4e}"
        else:
            if regime == "adiabatic":
                result["recommended_action"] = "Use analytic solution: φ̂ = ρ̂^{-1/(n+1)}"
            else:
                result["recommended_action"] = "Field is approximately constant; no solver needed"

    return [TextContent(type="text", text=json.dumps(result, indent=2))]
