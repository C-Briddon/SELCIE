#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:36:41 2020

@author: ppycb3

Functions for generating Legendre polynomial shapes for testing solvers.
"""
import math
import gmsh
import numpy as np


import sys
sys.path.append("..")
from Main.MeshingTools import MeshingTools


def legendre_coef(k, q, odd_or_even):
    'For even N=2*q and for odd N=2*q+1. k is kth term in polynomial.'
    if q < k:
        return 0

    else:
        if odd_or_even == 'even':
            return pow(-1, q-k)*math.factorial(2*q+2*k)/(pow(4, q) *
            math.factorial(q-k)*math.factorial(q+k)*math.factorial(2*k))

        elif odd_or_even == 'odd':
            return pow(-1, q-k)*math.factorial(2*q+2*k+1)/(pow(4, q) *
            math.factorial(q-k)*math.factorial(q+k)*math.factorial(2*k+1))


def legendre_R(theta, a_coef):
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
    theta = np.linspace(0, angle, N, endpoint=False)
    R = legendre_R(theta, a_coef)
    shapes_holes = [[]]

    j = 0
    for i, _ in enumerate(R):
        if R[i]*R[i-1] < 0:
            shapes_holes[j].append([0.0, 0.0, 0.0])
            j += 1
            shapes_holes.append([])

        shapes_holes[j].append([R[i]*np.sin(theta[i]), R[i]*np.cos(theta[i]),
                                0.0])

    if len(shapes_holes) > 1:
        shapes_holes[0] += shapes_holes.pop(-1)

    shapes_pos = shapes_holes[::2]
    shapes_neg = shapes_holes[1::2]

    return shapes_pos, shapes_neg


def construct_legendre_mesh_2D(a_coef, angle=2*np.pi, N=100,
                               include_holes=True):
    MT = MeshingTools(dimension=2)
    shapes_pos, shapes_neg = legendre_shape_components(a_coef, angle, N)

    PosDimTags = []
    NegDimTags = []

    for shape in shapes_pos:
        PosDimTags.append(MT.points_to_surface(Points_list=shape))

    if include_holes:
        if shapes_neg:
            for shape in shapes_neg:
                NegDimTags.append(MT.points_to_surface(Points_list=shape))

            SurfaceDimTags = MT.non_intersect_shapes(PosDimTags, NegDimTags)
        else:
            SurfaceDimTags = PosDimTags
    else:
        SurfaceDimTags = PosDimTags

    return SurfaceDimTags


'''
def construct_legendre_mesh_3D(a_coef, angle=2*np.pi, N=100):
    MT = Meshing_Tools(Dimension=3)
    shapes_pos, shapes_neg = legendre_shape_components(a_coef, angle, N)

    PosDimTags = []
    NegDimTags = []



    return None
'''


def test2D():
    a_coef = [0.191, 0.0249, 0.6536, 0.9]
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)

    SurfaceDimTags = construct_legendre_mesh_2D(a_coef, include_holes=False)
    print(SurfaceDimTags)
    # gmsh.model.occ.remove([SurfaceDimTags[4]])

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=2)

    gmsh.fltk.run()
    gmsh.clear()
    gmsh.finalize()

    return None


def test3D():
    a_coef = [0.191, 0.0249, 0.6536, 0.9]
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)

    SurfaceDimTags = construct_legendre_mesh_2D(a_coef)
    print(SurfaceDimTags)
    # gmsh.model.occ.remove([SurfaceDimTags[4]])

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim=2)

    gmsh.fltk.run()
    gmsh.clear()
    gmsh.finalize()

    return None


if __name__ == "__main__":
    test2D()
