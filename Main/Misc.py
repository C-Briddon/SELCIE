#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:08:36 2021

@author: ppycb3

Code containing miscellaneous functions for aiding the user.
"""

from astropy import units, constants


def alpha_calculator_chameleon(n, M, Lam, p_vac, L, M_NonEVUnits=None,
                               Lam_NonEVUnits=None, p_vac_NonEVUnits=None,
                               L_NonEVUnits=None):

    # Get unit conversion factors.
    kg_2_eV = (constants.c.value**2)/constants.e.value
    _per_m_2_eV = constants.c.value*constants.hbar.value/constants.e.value
    kg_per_m3_2_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(
        constants.e.value**4)

    if M_NonEVUnits:
        M *= M_NonEVUnits.to(units.kg)*kg_2_eV

    if Lam_NonEVUnits:
        Lam *= Lam_NonEVUnits.to(units.m**(-1))*_per_m_2_eV

    if p_vac_NonEVUnits:
        p_vac *= p_vac_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_2_eV4

    if L_NonEVUnits:
        L *= L_NonEVUnits.to(units.m)/_per_m_2_eV

    return (M*Lam/(p_vac*L**2))*(((n*M*Lam**3)/p_vac)**(1/(n+1)))


def calc_field_vacuum_value_chamelon(n, M, Lam, p_vac, Field_NonEVUnits=None,
                                     M_NonEVUnits=None, Lam_NonEVUnits=None,
                                     p_vac_NonEVUnits=None):
    '''
    Calculates the vacuum value for the field with units eV, unless
    specifed to the contrary with 'Field_NonEVUnits'. Then the field will
    be defined in the units provided as 'Field_NonEVUnits'.

    Parameters
    ----------
    M : TYPE float
        The coupling constant of the field to matter.
    Lam : TYPE float
        The energy scale of the field potencial.
    p_vac : TYPE float
        Vacuum density.
    Field_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of the returned field. If set to None then the units are
        taken to be eV. The default is None.
    M_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of M. If set to None then the units are taken to be eV.
        The default is None.
    Lam_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of Lam. If set to None then the units are taken to be eV.
        The default is None.
    p_vac_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of p_vac. If set to None then the units are taken to be eV.
        The default is None.

    Returns
    -------
    field_vacuum_value : TYPE float
        Value of the vacuum value of the field in units eV unless
        'Field_NonEVUnits' is not None. Then return value will be in units
        'Field_NonEVUnits'.

    '''
    kg_2_eV = (constants.c.value**2)/constants.e.value
    _per_m_2_eV = constants.c.value*constants.hbar.value/constants.e.value
    kg_per_m3_2_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(
        constants.e.value**4)

    if M_NonEVUnits:
        M *= M_NonEVUnits.to(units.kg)*kg_2_eV

    if Lam_NonEVUnits:
        Lam *= Lam_NonEVUnits.to(units.m**(-1))*_per_m_2_eV

    if p_vac_NonEVUnits:
        p_vac *= p_vac_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_2_eV4

    field_vacuum_value = Lam*pow(n*M*(Lam**3)/p_vac, 1/(n+1))

    if Field_NonEVUnits:
        field_vacuum_value *= (units.m**(-1)).to(Field_NonEVUnits)/_per_m_2_eV

    return field_vacuum_value


def conv_fifth_force_chameleon(n, M, Lam, p_vac, L, g=9.80665,
                               g_NonEVUnits=units.m/units.s**2,
                               M_NonEVUnits=None, Lam_NonEVUnits=None,
                               p_vac_NonEVUnits=None, L_NonEVUnits=None):
    '''
    Calculates the constant of proportionality between the dimensionless
    field gradient and the fifth force induced.

    Parameters
    ----------
    M : TYPE float
        The coupling constant of the field to matter.
    Lam : TYPE float
        The energy scale of the field potencial.
    p_vac : TYPE float
        Vacuum density.
    L : TYPE float
        Length scale of the system.
    g : TYPE, optional
        Rescaling factor of the fifth force. E.g. fifith force will be in
        units of g. The default is 9.80665.
    g_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of g. If set to None then the units are taken to be eV.
        The default is units.m/units.s**2.
    M_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of M. If set to None then the units are taken to be eV.
        The default is None.
    Lam_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of Lam. If set to None then the units are taken to be eV.
        The default is None.
    p_vac_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of p_vac. If set to None then the units are taken to be eV.
        The default is None.
    L_NonEVUnits : TYPE astropy.units.core.Unit, optional
        Units of L. If set to None then the units are taken to be eV.
        The default is None.

    Returns
    -------
    Conv_Xi_ff : TYPE
        The constant of proportionality between the dimensionless field
        gradient and the fifth force induced.

    '''
    kg_2_eV = (constants.c.value**2)/constants.e.value
    _per_m_2_eV = constants.c.value*constants.hbar.value/constants.e.value
    kg_per_m3_2_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(
        constants.e.value**4)
    m_per_s2_2_eV = constants.hbar.value/(constants.c.value*constants.e.value)

    if M_NonEVUnits:
        M *= M_NonEVUnits.to(units.kg)*kg_2_eV

    if Lam_NonEVUnits:
        Lam *= Lam_NonEVUnits.to(units.m**(-1))*_per_m_2_eV

    if p_vac_NonEVUnits:
        p_vac *= p_vac_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_2_eV4

    if L_NonEVUnits:
        L *= L_NonEVUnits.to(units.m)/_per_m_2_eV

    if g_NonEVUnits:
        g *= g_NonEVUnits.to(units.m/units.s**2)*m_per_s2_2_eV

    Conv_Xi_ff = Lam*pow(n*M*(Lam**3)/p_vac, 1/(n+1))/(M*L*g)

    return Conv_Xi_ff
