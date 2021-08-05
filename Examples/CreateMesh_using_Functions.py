#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:39:11 2021

@author: ppycb3

Environment - fenics2019

Example of creating a mesh. In this exapmle we demonstrating using 
Meshing_Tools functions to construct shapes.
"""
import sys
import gmsh


sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools
MT = Meshing_Tools()


# Create the mesh in gmsh.
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)


s1 = MT.create_rectangle(dx = 0.8, dy = 0.2)
s1 = MT.rotate_z(s1, rot_fraction = 0.05)

s2 = MT.create_rectangle(dx = 0.2, dy = 0.8)
s2 = MT.rotate_z(s2, rot_fraction = -0.05)

s12 = MT.add_shapes(s1, s2)

s3 = MT.create_disk()

s123 = MT.subtract_shapes(s12, s3)

s4 = MT.create_disk(0.1, 0.2)
s4 = MT.translate_x(s4, dx = 0.3)

s5 = MT.create_disk(0.1, 0.2)
s5 = MT.translate_x(s5, dx = -0.3)

s1234 = MT.add_shapes(s123, s4)
s12345 = MT.add_shapes(s1234, s5)



gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(dim=2)

gmsh.fltk.run()
gmsh.clear()
gmsh.finalize()

