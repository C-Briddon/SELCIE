#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:49:00 2021

@author: Chad Briddon

Example of creating a mesh. In this example the mesh is a circular source
inside an empty vacuum.
"""
from SELCIE import MeshingTools

# Choose source and vacuum radial sizes.
r_s = 0.005
r_v = 1.0

filename = "Circle_in_Vacuum_r" + str(r_s)

# Construct mesh.
MT = MeshingTools(dimension=2)

MT.create_ellipse(rx=r_s, ry=r_s)
MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.05, DistMax=0.4)

MT.create_background_mesh(CellSizeMin=1e-3, CellSizeMax=0.05, DistMax=0.4,
                          background_radius=r_v, wall_thickness=0.1)

MT.generate_mesh(filename, show_mesh=True)

MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)
