#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:46:07 2022

@author: Chad Briddon

Example illustrating how to use user defined initial conditions for solver.

To generate mesh run 'CreateMesh_Legendre.py'.
"""
import numpy as np
from SELCIE import FieldSolver
from SELCIE import DensityProfile


# Define density profile functions.
def source_wall(x):
    return 1.0e17


def vacuum(x):
    return 1.0


def f1(x):
    return 1e-9


def f2(x):
    d = 2/15
    r = np.linalg.norm(x)
    return max(1e-9, (r-d)*(1.0-r)*1e-6)


# Define parameters.
n = 1
alpha = 1e18

# Define the density profile of the mesh using its subdomains.
filename = "Legndre[0.13333333 0.         0.         0.        ]"
p = DensityProfile(filename, dimension=2, symmetry='vertical axis-symmetry',
                   profiles=[source_wall, vacuum, source_wall])

# Setup problem with custom initial conditions and solve.
s = FieldSolver(alpha, n, density_profile=p,
                initial_field_profiles=[f1, f2, f1])

s.plot_results(field_scale='log')

# Using non-default initial field profile can cause faster convergence.
s.picard()
s.plot_results(field_scale='log')
