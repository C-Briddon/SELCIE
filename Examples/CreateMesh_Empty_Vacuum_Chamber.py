#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:28:23 2021

@author: Chad Briddon

Example of creating a mesh. In this example the mesh is an empty vacuum chamber
with a radial size of unity.
"""
import sys
sys.path.append("..")
from SELCIE.MeshingTools import MeshingTools

# Choose source and vacuum radial sizes.
r = 1.0
wt = 0.1

filename = "Circle_Empty_Vacuum_chamber"

# Construct mesh.
MT = MeshingTools(dimension=2)

MT.create_background_mesh(CellSizeMin=1e-3, CellSizeMax=0.01, DistMax=1.0,
                          background_radius=r, wall_thickness=wt)

MT.generate_mesh(filename, show_mesh=True)

MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)
