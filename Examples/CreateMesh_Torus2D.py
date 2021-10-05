#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:13:23 2021

@author: ppycb3

Example of creating a 2D mesh. In this exapmle the mesh is a torus shaped
source inside an empty vacuum.
"""
import sys
sys.path.append("..")
from Main.MeshingTools import MeshingTools

# Choose source and vacuum radial sizes.
r_inner = 0.05
r_t = 0.1
r_v = 1.0
dx = r_inner + r_t

filename = "../Saved Meshes/Torus_in_Vacuum_2D"

# Create the mesh in gmsh.
MT = MeshingTools(dimension=2)

c1 = MT.create_disk(rx=r_t, ry=r_t)
MT.translate_x(c1, dx)

c2 = MT.create_disk(rx=r_t, ry=r_t)
MT.translate_x(c2, -dx)

MT.create_subdomain(CellSizeMin=1e-4, CellSizeMax=0.05, DistMax=0.1)

MT.create_background_mesh(CellSizeMin=1e-4, CellSizeMax=0.05, DistMax=0.1,
                          background_radius=r_v, wall_thickness=0.05)

MT.generate_mesh(filename, show_mesh=True)

MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)
