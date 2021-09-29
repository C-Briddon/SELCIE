#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:27:24 2021

@author: ppycb3

Environment - fenics2019

Example of creating a mesh. In this exapmle the mesh is an ellipical source 
inside an empty vacuum.
"""
import sys
import gmsh
import numpy as np

sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools

# Choose source and vacuum radial sizes.
Xi = 1.01
r0 = 0.005
a = r0/(Xi*(Xi**2 - 1))**(1/3)
r_v = 1.0

# Create the mesh in gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)

MT = Meshing_Tools(Dimension=2)

MT.create_disk(rx=a*Xi, ry=a*np.sqrt(Xi**2 - 1))
MT.create_subdomain(CellSizeMin=5e-5, CellSizeMax=0.05, DistMax=0.4,
                    NumPointsPerCurve=10000)

MT.create_background_mesh(CellSizeMin=5e-3, CellSizeMax=0.05, DistMax=0.4,
                          background_radius=r_v, wall_thickness=0.1)
MT.generate_mesh()


# After saving the mesh view it and then clear and close gmsh.
filename = "../Saved Meshes/Ellipse_in_Vacuum_r%f_Xi%f" % (r0, Xi)
gmsh.write(fileName=filename+".msh")

gmsh.fltk.run()
gmsh.clear()
gmsh.finalize()
