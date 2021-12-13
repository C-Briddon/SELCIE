#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:08:36 2021

@author: Chad Briddon

Code containing miscellaneous functions for aiding the user.
"""
import math
import numpy as np
from astropy import units, constants


def alpha_calculator_chameleon(n, M, Lam, p0, L, M_NonEVUnits=None,
                               Lam_NonEVUnits=None, p0_NonEVUnits=None,
                               L_NonEVUnits=None):
    '''
    Calculates the numical value of the alpha-parameter given the other
    parameters of system.

    Parameters
    ----------
    n : int
        Integer value which defines the form of the field potential.
    M : float
        Is equivalent to M_{pl}/beta, where M_{pl} is the planck mass and beta
        is the coupling constant of the field to matter.
    Lam : float
        Energy scale of the field potential.
    p0 : float
        Density rescaling value.
    L : float
        Length rescaling value.
    M_NonEVUnits : astropy.units.core.Unit, optional
        Units of M. If set to None then the units are taken to be eV.
        The default is None.
    Lam_NonEVUnits : astropy.units.core.Unit, optional
        Units of Lam. If set to None then the units are taken to be eV.
        The default is None.
    p0_NonEVUnits : astropy.units.core.Unit, optional
        Units of p0. If set to None then the units are taken to be eV^4.
        The default is None.
    L_NonEVUnits : astropy.units.core.Unit, optional
        Units of L. If set to None then the units are taken to be eV^{-1}.
        The default is None.

    Returns
    -------
    float
        The value of alpha.

    '''

    # Unit conversion factors.
    kg_2_eV = (constants.c.value**2)/constants.e.value
    _per_m_2_eV = constants.c.value*constants.hbar.value/constants.e.value
    kg_per_m3_2_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(
        constants.e.value**4)

    if M_NonEVUnits:
        M *= M_NonEVUnits.to(units.kg)*kg_2_eV

    if Lam_NonEVUnits:
        Lam *= Lam_NonEVUnits.to(units.m**(-1))*_per_m_2_eV

    if p0_NonEVUnits:
        p0 *= p0_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_2_eV4

    if L_NonEVUnits:
        L *= L_NonEVUnits.to(units.m)/_per_m_2_eV

    return (M*Lam/(p0*L**2))*(((n*M*Lam**3)/p0)**(1/(n+1)))


def calc_field_min(n, M, Lam, p0, Field_NonEVUnits=None, M_NonEVUnits=None,
                   Lam_NonEVUnits=None, p0_NonEVUnits=None):
    '''
    Calculates the field value that minimises the effective potential.

    Parameters
    ----------
    n : int
        Integer value which defines the form of the field potential.
    M : float
        Is equivalent to M_{pl}/beta, where M_{pl} is the planck mass and beta
        is the coupling constant of the field to matter.
    Lam : float
        Energy scale of the field potential.
    p0 : float
        Density rescaling value.
    Field_NonEVUnits : astropy.units.core.Unit, optional
        Units of the returned field. If set to None then the units are taken
        to be eV. The default is None.
    M_NonEVUnits : astropy.units.core.Unit, optional
        Units of M. If set to None then the units are taken to be eV.
        The default is None.
    Lam_NonEVUnits : astropy.units.core.Unit, optional
        Units of Lam. If set to None then the units are taken to be eV.
        The default is None.
    p0_NonEVUnits : astropy.units.core.Unit, optional
        Units of p0. If set to None then the units are taken to be eV^4.
        The default is None.

    Returns
    -------
    field_min : float
        Value of the field that minimises the effective potential, in units
        specified by 'Field_NonEVUnits'.

    '''

    # Unit conversion factors.
    kg_2_eV = (constants.c.value**2)/constants.e.value
    _per_m_2_eV = constants.c.value*constants.hbar.value/constants.e.value
    kg_per_m3_2_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(
        constants.e.value**4)

    if M_NonEVUnits:
        M *= M_NonEVUnits.to(units.kg)*kg_2_eV

    if Lam_NonEVUnits:
        Lam *= Lam_NonEVUnits.to(units.m**(-1))*_per_m_2_eV

    if p0_NonEVUnits:
        p0 *= p0_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_2_eV4

    field_min = Lam*pow(n*M*(Lam**3)/p0, 1/(n+1))

    if Field_NonEVUnits:
        field_min *= (units.m**(-1)).to(Field_NonEVUnits)/_per_m_2_eV

    return field_min


def conv_fifth_force_chameleon(n, M, Lam, p0, L, g=9.80665,
                               g_NonEVUnits=units.m/units.s**2,
                               M_NonEVUnits=None, Lam_NonEVUnits=None,
                               p0_NonEVUnits=None, L_NonEVUnits=None):
    '''
    Calculates the constant of proportionality between the dimensionless
    rescaled field gradient and the fifth force in units of g.

    Parameters
    ----------
    n : int
        Integer value which defines the form of the field potential.
    M : float
        Is equivalent to M_{pl}/beta, where M_{pl} is the planck mass and beta
        is the coupling constant of the field to matter.
    Lam : float
        Energy scale of the field potential.
    p0 : float
        Density rescaling value.
    L : float
        Length rescaling value.
    g : float, optional
        Numerical value of g. The default is 9.80665.
    g_NonEVUnits : astropy.units.core.Unit, optional
        Units of g. If set to None then the units are taken to be eV^2.
        The default is units.m/units.s**2.
    M_NonEVUnits : astropy.units.core.Unit, optional
        Units of M. If set to None then the units are taken to be eV.
        The default is None.
    Lam_NonEVUnits : astropy.units.core.Unit, optional
        Units of Lam. If set to None then the units are taken to be eV.
        The default is None.
    p0_NonEVUnits : astropy.units.core.Unit, optional
        Units of p0. If set to None then the units are taken to be eV^4.
        The default is None.
    L_NonEVUnits : astropy.units.core.Unit, optional
        Units of L. If set to None then the units are taken to be eV^{-1}.
        The default is None.

    Returns
    -------
    conv_grad_to_ff : float
        Conversion value relating the dimensionless field gradient and the
        fifth force in units of g.

    '''

    # Unit conversion factors.
    kg_2_eV = (constants.c.value**2)/constants.e.value
    _per_m_2_eV = constants.c.value*constants.hbar.value/constants.e.value
    kg_per_m3_2_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(
        constants.e.value**4)
    m_per_s2_2_eV = constants.hbar.value/(constants.c.value*constants.e.value)

    if M_NonEVUnits:
        M *= M_NonEVUnits.to(units.kg)*kg_2_eV

    if Lam_NonEVUnits:
        Lam *= Lam_NonEVUnits.to(units.m**(-1))*_per_m_2_eV

    if p0_NonEVUnits:
        p0 *= p0_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_2_eV4

    if L_NonEVUnits:
        L *= L_NonEVUnits.to(units.m)/_per_m_2_eV

    if g_NonEVUnits:
        g *= g_NonEVUnits.to(units.m/units.s**2)*m_per_s2_2_eV

    conv_grad_to_ff = Lam*pow(n*M*(Lam**3)/p0, 1/(n+1))/(M*L*g)

    return conv_grad_to_ff


def legendre_coef(k, q, odd_or_even):
    '''
    Returns coefficient of polynomial that is derived from Legendre series.

    Parameters
    ----------
    k : int
        Coefficient is for kth term in polynomial.
    q : int
        If number of Legendre coefficients, N, is even N=2*q. For odd N,
        N=2*q+1.
    odd_or_even : {'odd', 'even'}
        State if number of Legendre coefficients, N, is odd or even.

    Returns
    -------
    float
        Polynomial coefficient for given k and q for Legendre series.

    '''

    if q < k:
        return 0

    else:
        if odd_or_even == 'even':
            return pow(-1, q-k)*math.factorial(
                2*q+2*k)/(pow(4, q) * math.factorial(q-k) *
                          math.factorial(q+k)*math.factorial(2*k))

        elif odd_or_even == 'odd':
            return pow(-1, q-k)*math.factorial(
                2*q+2*k+1)/(pow(4, q) * math.factorial(q-k) *
                            math.factorial(q+k)*math.factorial(2*k+1))

    return None


def legendre_R(theta, a_coef):
    '''
    Calculate the radial distance from the origin of Legendre series.

    Parameters
    ----------
    theta : float
        Angular coordinate.
    a_coef : list of float
        Coefficients of the Legendre series.

    Returns
    -------
    R : float
        Radial value of Legendre series.

    '''

    N = len(a_coef)
    if N % 2 == 0:
        No = int(N/2)
        Ne = int(N/2)
    else:
        No = int((N-1)/2)
        Ne = int((N+1)/2)
    R = 0
    c = np.cos(theta)

    for q in range(Ne):
        for k in range(q+1):
            R += a_coef[2*q]*legendre_coef(k, q, 'even')*pow(c, 2*k)
    for q in range(No):
        for k in range(q+1):
            R += a_coef[2*q+1]*legendre_coef(k, q, 'odd')*pow(c, 2*k+1)

    return R
