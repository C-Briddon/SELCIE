#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:29:58 2022

@author: Chad Briddon

Example illustrating how to use MeshingTools.MT.points_to_volume().

The input for this function is a list of lists. Each sublist defines a closed
contour that when joined together forms the 3D shape. The elements of these
sublists are lists containing the x, y, and z coordinate of each point.
"""
import numpy as np
from SELCIE import MeshingTools

MT = MeshingTools(dimension=3)

# First will define half of a torus.
N_a = 100
N_t = 20
T = 1.0
a = 1.0

cs = []
for t1 in np.linspace(0, np.pi, N_a, endpoint=False):
    cs.append([])
    for t2 in np.linspace(0, 2*np.pi, N_t, endpoint=False):
        c = a/(np.cosh(T) - np.cos(t2))
        cs[-1].append([c*np.sinh(T)*np.cos(t1),
                       c*np.sinh(T)*np.sin(t1), c*np.sin(t2)])

# Use function to construct 3D shape.
MT.points_to_volume(contour_list=cs)

# Second example is a hourglass shape.
r = 0.1
t = np.linspace(0, 2*np.pi, 20, endpoint=False)
X = np.sin(t)
Y = np.cos(t)

c1 = [[x, y, -1.0] for x, y in zip(X, Y)]
c2 = [[r*x, r*y, 0.0] for x, y in zip(X, Y)]
c3 = [[x, y, +1.0] for x, y in zip(X, Y)]

# Use function to construct 3D shape.
MT.points_to_volume(contour_list=[c1, c2, c3])

# Show result.
MT.generate_mesh(show_mesh=True)
