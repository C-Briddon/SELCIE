#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:16:57 2021

@author: ppycb3

Environment - fenics2019

Test genetic algorithm using [] to reproduce legendre polynomial shapes.
"""
import gmsh
geom = gmsh.model.occ
import numpy as np
from Tools_Legendre import construct_legendre_mesh_2D


import sys
sys.path.append("../Main")
from Shape_GA import Shape_GA
from Fitness_Functions import evalSimularity

@ evalSimularity
def legendre_shape():
    a_coef = np.array([0.82, 0.02, 0.85, 3.94])/15
    #a_coef = np.array([0.97, 0.59, 0.03, 3.99])/15
    #a_coef = np.array([1.34, 0.18, 0.41, 2.89])/15
    #a_coef = np.array([1.47, 0.19, 0.27, 2.63])/15
    #a_coef = np.array([2.00, 0.00, 0.00, 0.00])/15
    
    SurfaceDimTags = construct_legendre_mesh_2D(a_coef, N = 200, include_holes = False)
    #SurfaceDimTags = [(2, geom.addDisk(xc=0, yc=0, zc=0, rx=0.3, ry=0.2))]
    return SurfaceDimTags


"Setup genetic algorithm."
GA = Shape_GA(legendre_shape, Np = 500, initial_tree_depth_min = 20, 
              initial_tree_depth_max = 30, MAX_HEIGHT = 30)
GA.generation_max = 1000
GA.prob_X = 0.6
GA.prob_M = 0.4


"Use Genetic Algorithm."
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 1)

GA.run_GA()

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(dim=2)

gmsh.fltk.run()
gmsh.clear()
gmsh.finalize()


