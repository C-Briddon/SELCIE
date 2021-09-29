#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:51:59 2021

@author: ppycb3

Environment - fenics2019

Example of creating a 3D mesh. In this exapmle the mesh is a torus shaped
source inside an empty vacuum.
"""
import sys
import gmsh
import numpy as np
geom = gmsh.model.occ

sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools

# Choose source and vacuum radial sizes.
r_inner = 0.05
r_t = 0.1
r_v = 1.0


# Create the mesh in gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)

MT = Meshing_Tools(Dimension=3)


# Construct Torus.
N_a = 100
N_t = 60
cs1 = []
cs2 = []

for a in np.linspace(-np.pi/2, np.pi/2, N_a, endpoint=True):
    cs1.append([])
    r = r_inner + r_t*(1 + np.sin(a))
    y = r_t*np.cos(a)
    for t in np.linspace(0, 2*np.pi, N_t, endpoint=False):
        cs1[-1].append([r*np.cos(t), y, r*np.sin(t)])

for a in np.linspace(-np.pi/2, np.pi/2, N_a, endpoint=True):
    cs2.append([])
    r = r_inner + r_t*(1 - np.sin(a))
    y = r_t*np.cos(a)
    for t in np.linspace(0, 2*np.pi, N_t, endpoint=False):
        cs2[-1].append([r*np.cos(t), -y, r*np.sin(t)])


s1 = MT.points_to_volume(Contour_list=cs1)
s2 = MT.points_to_volume(Contour_list=cs2)

s3 = MT.add_shapes(s1, s2)

# Place mesh torus inside vacuum chamber.
MT.create_background_mesh(CellSizeMin=5e-3, CellSizeMax=0.05, DistMax=0.1,
                          background_radius=r_v, wall_thickness=0.05)
MT.generate_mesh()


# After saving the mesh view it and then clear and close gmsh.
filename = "../Saved Meshes/Torus_in_Vacuum_3D"
gmsh.write(fileName=filename+".msh")

gmsh.fltk.run()
gmsh.clear()
gmsh.finalize()
