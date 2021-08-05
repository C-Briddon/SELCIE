#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:28:23 2021

@author: ppycb3

Environment - fenics2019

Example of creating a mesh. In this exapmle the mesh is an empty vacuum chamber
with a radial size of unity.
"""
import sys
import gmsh
geom = gmsh.model.occ

sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools

# Choose source and vacuum radial sizes.
r = 1.0
wt = 0.1

# Create the mesh in gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)

MT = Meshing_Tools()
MT.vacuum = MT.create_disk(rx = r, ry = r)
MT.inner_wall_boundary = geom.getEntities(dim = 1)


MT.wall, _ = geom.cut(objectDimTags = MT.create_disk(rx = r + wt, ry = r + wt), 
                      toolDimTags = MT.vacuum, removeObject = True, removeTool = False)

MT.outer_wall_boundary = [b for b in geom.getEntities(dim = 1)
                          if b not in MT.inner_wall_boundary]

geom.synchronize()
MT.generate_mesh_2D(SizeMin = 1e-3, SizeMax = 0.01, DistMax = 1)

# After saving the mesh view it and then clear and close gmsh.
filename = "../Saved Meshes/Circle_Empty_Vacuum_chamber"
gmsh.write(fileName = filename + ".msh")


gmsh.fltk.run()
gmsh.clear()
gmsh.finalize()
