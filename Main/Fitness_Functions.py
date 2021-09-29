#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:12:07 2021

@author: ppycb3

Environment - fenics2019

Collection of Fitness Functions to be used with the Genetic Algorithm.
"""
import gmsh
from deap import gp

from Meshing_Tools import Meshing_Tools
MT = Meshing_Tools()


def evalSimularity(TargetShapeFunction):
    def wrapper(individual, pset):
        'Create target shape.'
        ts = TargetShapeFunction()

        'Compare with shape from indivigual tree.'
        run = gp.compile(individual, pset)
        mass = MT.shape_similarity(ts, run)

        'Clear GMSH of target and generated shapes.'
        gmsh.clear()
        return mass,
    return wrapper
