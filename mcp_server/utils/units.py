"""Unit conversion utilities for SELCIE MCP server."""

from astropy import units


def get_astropy_density_unit(unit_str: str):
    """
    Convert density unit string to astropy unit.

    Parameters
    ----------
    unit_str : str
        One of: "g/cm^3", "kg/m^3", "M_sun/kpc^3"

    Returns
    -------
    astropy.units.Unit or None
        Astropy unit, or None if not found (e.g., for eV^4, GeV^4).
    """
    unit_map = {
        "g/cm^3": units.g / units.cm**3,
        "kg/m^3": units.kg / units.m**3,
        "M_sun/kpc^3": units.M_sun / units.kpc**3,
    }
    return unit_map.get(unit_str)


def get_astropy_length_unit(unit_str: str):
    """
    Convert length unit string to astropy unit.

    Parameters
    ----------
    unit_str : str
        One of: "m", "cm", "km", "R_sun", "kpc", "Mpc", "AU"

    Returns
    -------
    astropy.units.Unit or None
        Astropy unit, or None if not found.
    """
    unit_map = {
        "m": units.m,
        "cm": units.cm,
        "km": units.km,
        "R_sun": units.R_sun,
        "kpc": units.kpc,
        "Mpc": units.Mpc,
        "AU": units.AU,
    }
    return unit_map.get(unit_str)
