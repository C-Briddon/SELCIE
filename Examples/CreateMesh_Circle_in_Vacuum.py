#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:49:00 2021

@author: ppycb3

Environment - fenics2019

Example of creating a mesh. In this exapmle the mesh is a circular source
inside an empty vacuum.
"""
import sys
import gmsh

sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools

# Choose source and vacuum radial sizes.
r_s = 0.005
r_v = 1.0

# Create the mesh in gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)

MT = Meshing_Tools(Dimension=2)

MT.create_disk(rx=r_s, ry=r_s)
MT.create_subdomain(CellSizeMin=5e-5, CellSizeMax=0.05, DistMax=0.4)

MT.create_background_mesh(CellSizeMin=5e-3, CellSizeMax=0.05, DistMax=0.4,
                          background_radius=r_v, wall_thickness=0.1)
MT.generate_mesh()


# After saving the mesh view it and then clear and close gmsh.
filename = "../Saved Meshes/Circle_in_Vacuum_r" + str(r_s)
gmsh.write(fileName=filename+".msh")

gmsh.fltk.run()
gmsh.clear()
gmsh.finalize()
