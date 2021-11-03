#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:36:41 2020

@author: Chad Briddon

Functions for generating Legendre polynomial shapes for testing solvers.
"""
import math
import numpy as np

import sys
sys.path.append("..")
from Main.MeshingTools import MeshingTools


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


def legendre_shape_components(a_coef, angle=2*np.pi, N=100):
    '''
    Generate lists of points that outline segments of the Legendre polynomial
    shape.

    Parameters
    ----------
    a_coef : list of float
        Coefficients of the Legendre series.
    angle : float, optional
        Angular amount the shape covers in radians. The default is 2*np.pi.
    N : int, optional
        Total number of points. The default is 100.

    Returns
    -------
    shapes_pos : list of list of tuple
        Each list in shapes_pos contains tuple representing a point with a
        positive radial value from the Legendre series. The elements of the
        tuples are the x, y, and z coordinates of the point.
    shapes_neg : list of tuple
        Each list in shapes_neg contains tuple representing a point with a
        negative radial value from the Legendre series. The elements of the
        tuples are the x, y, and z coordinates of the point.

    '''

    theta = np.linspace(0, angle, N, endpoint=False)
    R = legendre_R(theta, a_coef)
    shapes_holes = [[]]

    j = 0
    for i, _ in enumerate(R):
        if R[i]*R[i-1] < 0:
            shapes_holes[j].append((0.0, 0.0, 0.0))
            j += 1
            shapes_holes.append([])

        shapes_holes[j].append((R[i]*np.sin(theta[i]), R[i]*np.cos(theta[i]),
                                0.0))

    if len(shapes_holes) > 1:
        shapes_holes[0] += shapes_holes.pop(-1)

    shapes_pos = shapes_holes[::2]
    shapes_neg = shapes_holes[1::2]

    return shapes_pos, shapes_neg


def construct_legendre_mesh_2D(MT, a_coef, angle=2*np.pi, N=100,
                               include_holes=True):
    '''
    Generate a 2D shape constructed using a Legendre series.

    Parameters
    ----------
    MT : <class 'Main.MeshingTools.MeshingTools'>
        Main.MeshingTools class used to construct mesh.
    a_coef : list of float
        Coefficients of the Legendre series.
    angle : float, optional
        Angular amount the shape covers in radians. The default is 2*np.pi.
    N : int, optional
        Total number of points. The default is 100.
    include_holes : bool, optional
        If True will remove intersections of regions with posative and negative
        radius from shape. Otherwise the shapes are merged so they have no
        holes. The default is True.

    Returns
    -------
    SurfaceDimTag : tuple
        Tuple containing the dimension and tag of the generated surface.

    '''

    shapes_pos, shapes_neg = legendre_shape_components(a_coef, angle, N)

    PosDimTags = []
    NegDimTags = []

    for shape in shapes_pos:
        PosDimTags.append(MT.points_to_surface(points_list=shape))

    for shape in shapes_neg:
        NegDimTags.append(MT.points_to_surface(points_list=shape))

    if PosDimTags and NegDimTags:
        if include_holes:
            SurfaceDimTags = MT.non_intersect_shapes(PosDimTags, NegDimTags)
        else:
            SurfaceDimTags = MT.add_shapes(PosDimTags, NegDimTags)
    else:
        SurfaceDimTags = PosDimTags + NegDimTags

    return SurfaceDimTags


def test():
    '''
    Run test of 'construct_legendre_mesh_2D'.

    Returns
    -------
    None.

    '''

    MT = MeshingTools(dimension=2)

    a_coef = [0.191, 0.0249, 0.6536, 0.9]

    construct_legendre_mesh_2D(MT, a_coef, include_holes=True)

    MT.generate_mesh(show_mesh=True)

    return None


if __name__ == "__main__":
    test()
