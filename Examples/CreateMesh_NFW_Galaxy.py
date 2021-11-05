#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:17:40 2021

@author: Chad Briddon

Generate mesh for NFW galexy cluster profile with a core region.
"""
from SELCIE import MeshingTools

# Define Cutoff and Domain radius.
r_cutoff = 1e-6
domain_size = 10

filename = "NFW_Galaxy"

# Create the mesh in gmsh.
MT = MeshingTools(dimension=2)

MT.create_ellipse(rx=r_cutoff, ry=r_cutoff)
MT.create_subdomain(CellSizeMin=2e-7, CellSizeMax=0.1, DistMax=0.1)

MT.create_background_mesh(CellSizeMin=0.1, CellSizeMax=0.1, DistMax=0.1,
                          background_radius=domain_size)

MT.generate_mesh(filename, show_mesh=True)

MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)
