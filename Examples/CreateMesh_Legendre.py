#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:59:21 2021

@author: Chad Briddon

Create mesh for a Legndre polynomial shape inside a vacuum chamber of radial
size unity.
"""
import numpy as np
from Tools_Legendre import construct_legendre_mesh_2D

import sys
sys.path.append("..")
from Main.MeshingTools import MeshingTools

# Give input coeficents that define the Legendre polynomial shape.
a_coef = np.array([0.82, 0.02, 0.85, 3.94])/15
# a_coef = np.array([0.97, 0.59, 0.03, 3.99])/15
# a_coef = np.array([1.34, 0.18, 0.41, 2.89])/15
# a_coef = np.array([1.47, 0.19, 0.27, 2.63])/15
# a_coef = np.array([2.00, 0.00, 0.00, 0.00])/15

filename = "Legndre" + str(a_coef)

# Create the mesh in gmsh.
MT = MeshingTools(dimension=2)

construct_legendre_mesh_2D(MT, a_coef, N=1000, include_holes=True)

MT.create_background_mesh(CellSizeMin=1e-4, CellSizeMax=0.1, DistMax=0.4,
                          background_radius=1.0, wall_thickness=0.1)

MT.generate_mesh(filename, show_mesh=True)

MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)
