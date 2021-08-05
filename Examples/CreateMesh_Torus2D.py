#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:13:23 2021

@author: ppycb3

Environment - fenics2019

Example of creating a 2D mesh. In this exapmle the mesh is a torus shaped 
source inside an empty vacuum.
"""
import sys
import gmsh

sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools

# Choose source and vacuum radial sizes.
r_inner = 0.05
r_t = 0.1
r_v = 1.0

dx = r_inner + r_t

# Create the mesh in gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)

MT = Meshing_Tools()
c1 = MT.create_disk(rx = r_t, ry = r_t)
MT.translate_x(c1, dx)

c2 = MT.create_disk(rx = r_t, ry = r_t)
MT.translate_x(c2, -dx)

MT.construct_vacuum_chamber_2D(vacuum_radius = r_v, wall_thickness = 0.1)
MT.generate_mesh_2D(SizeMin = 1e-4, SizeMax = 0.05, DistMax = 0.4)

# After saving the mesh view it and then clear and close gmsh.
filename = "../Saved Meshes/Torus_in_Vacuum_2D"
gmsh.write(fileName = filename + ".msh")

gmsh.fltk.run()
gmsh.clear()
gmsh.finalize()