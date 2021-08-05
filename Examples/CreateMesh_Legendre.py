#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:59:21 2021

@author: ppycb3

Environment - fenics2019

Create mesh for a Legndre polynomial shape inside a vacuum chamber of radial 
size unity.
"""
import gmsh
import numpy as np
from Tools_Legendre import construct_legendre_mesh_2D

import sys
sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools

# Give input coeficents that define the Legendre polynomial shape.
#a_coef = np.array([0.82, 0.02, 0.85, 3.94])/15
#a_coef = np.array([0.97, 0.59, 0.03, 3.99])/15
#a_coef = np.array([1.34, 0.18, 0.41, 2.89])/15
#a_coef = np.array([1.47, 0.19, 0.27, 2.63])/15
a_coef = np.array([2.00, 0.00, 0.00, 0.00])/15


# Create the mesh in gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)

SurfaceDimTags = construct_legendre_mesh_2D(a_coef, N = 100, include_holes = True)

MT = Meshing_Tools()

MT.construct_vacuum_chamber_2D(wall_thickness = 0.1, vacuum_radius = 1)
MT.generate_mesh_2D(SizeMin = 1e-4, SizeMax = 0.05, DistMax = 0.4)
'''
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(dim=2)
'''
# After saving the mesh view it and then clear and close gmsh.
filename = "../Saved Meshes/Legndre" + str(a_coef)
gmsh.write(fileName = filename + ".msh")

gmsh.fltk.run()
gmsh.clear()
gmsh.finalize()