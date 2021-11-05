#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:27:24 2021

@author: Chad Briddon

Example of creating a mesh. In this example the mesh is an ellipical source
inside an empty vacuum.
"""
import numpy as np
from SELCIE import MeshingTools

# Choose source and vacuum radial sizes.
Xi = 1.01
r0 = 0.005
r_v = 1.0
a = r0/(Xi*(Xi**2 - 1))**(1/3)

filename = "Ellipse_in_Vacuum_r%f_Xi%f" % (r0, Xi)

# Construct mesh.
MT = MeshingTools(dimension=2)

MT.create_ellipse(rx=a*Xi, ry=a*np.sqrt(Xi**2 - 1))
MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.05, DistMax=0.4,
                    NumPointsPerCurve=10000)

MT.create_background_mesh(CellSizeMin=5e-3, CellSizeMax=0.05, DistMax=0.4,
                          background_radius=r_v, wall_thickness=0.1)

MT.generate_mesh(filename, show_mesh=True)

MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)
