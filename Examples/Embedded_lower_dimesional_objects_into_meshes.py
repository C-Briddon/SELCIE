#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 11:36:06 2025

@author: chad-briddon

Demonstration of how to use 'MeshingTools.embed_lower_dimension_shape()'.
Generates two spheres with plane through centres and with an embedded sin wave.

"""

import numpy as np
from SELCIE import MeshingTools

MT = MeshingTools(3, display_messages=False)

S1 = MT.create_ellipsoid(rx=1, ry=1, rz=1)
S2 = MT.create_ellipsoid(rx=1, ry=1, rz=1)

MT.translate_x(shapes=S1, dx=+1.05)
MT.translate_x(shapes=S2, dx=-1.05)

R = MT.create_rectangle(4.1, 4.1)
MT.add_points(
    shapes=R, points_list=[[0.0, 1.5, 0.0], [0.1, 1.6, 0.0]])  # embed points.

L = MT.points_to_curve([[x, np.sin(9*x), 0] for x in np.linspace(-3, 3, 1000)])

new_shapes = MT.embed_lower_dimension_shape(shapes=S1+S2, embed=L+R,
                                            remove_trimmings=False)
print(new_shapes)

MT.create_subdomain(CellSizeMin=0.5, CellSizeMax=0.5)
MT.generate_mesh(show_mesh=True)
