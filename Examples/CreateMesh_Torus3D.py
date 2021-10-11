#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:51:59 2021

@author: ppycb3

Example of creating a 3D mesh. In this exapmle the mesh is a torus shaped
source inside an empty vacuum.
"""
import sys
sys.path.append("..")
from Main.MeshingTools import MeshingTools

# Choose source and vacuum radial sizes.
r_inner = 0.05
r_t = 0.1
r_v = 1.0

filename = "../Saved Meshes/Torus_in_Vacuum_3D_2"

# Create the mesh in gmsh.
MT = MeshingTools(dimension=3)

# Construct Torus.
MT.create_torus(r_hole=r_inner, r_tube=r_t)

# Place mesh torus inside vacuum chamber.
MT.create_background_mesh(CellSizeMin=1e-2, CellSizeMax=0.1, DistMax=0.2,
                          background_radius=r_v, wall_thickness=0.05)

MT.generate_mesh(filename=filename, show_mesh=True)

MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)
