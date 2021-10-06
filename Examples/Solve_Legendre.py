#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:16:36 2021

@author: ppycb3

Solve the chameleon field around a Legndre polynomial shape.
"""
import numpy as np
from astropy import units
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from Main.MeshingTools import MeshingTools
from Main.SolverChameleon import FieldSolver
from Main.DensityProfiles import DensityProfile
from Main.Misc import conv_fifth_force_chameleon


# Define density profile functions.
def source_wall(x):
    return 1.0e17


def vacuum(x):
    return 1.0


# Import mesh and convert from .msh to .xdmf.
a_coef = np.array([0.82, 0.02, 0.85, 3.94])/15
# a_coef = np.array([0.97, 0.59, 0.03, 3.99])/15
# a_coef = np.array([1.34, 0.18, 0.41, 2.89])/15
# a_coef = np.array([1.47, 0.19, 0.27, 2.63])/15
# a_coef = np.array([2.00, 0.00, 0.00, 0.00])/15

MT = MeshingTools(dimension=2)


# Set model parameters.
n = 1
alpha = 6.115821312572006e+18   # James's value.
#alpha = 2.3238240196e19         # James's result with M = 2.435e18 eV.
#alpha = 1e+18
alpha = 1e6
ps = 1e17

d = 0.5/15
d = 0.01
d_tol = 1e-4


# Define the density profile of the mesh using its subdomains.
p = DensityProfile(filename="../Saved Meshes/Legndre" + str(a_coef),
                   dimension=2, symmetry='vertical axis-symmetry',
                   profiles=[source_wall, vacuum, source_wall], degree=0)

# Setup problem.
s = FieldSolver(alpha, n, density_profile=p)


# Set tolerance on field solutions and solve for above problems.
s.picard()
s.calc_field_grad_mag()
s.calc_field_residual()

field_grad, probe_point = s.measure_fifth_force(boundary_distance=d, tol=d_tol)

s.plot_results(field_scale='log', grad_scale='log')
plt.plot(probe_point.x(), probe_point.y(), 'rx')


# Rescale fifth_force_max into units of g.
M = 1e27                            # eV
Lam = 1.0e-3                        # eV
p_vac = 43.10130531853594           # eV4
L = 15.0                            # cm
Xi_2_ff = conv_fifth_force_chameleon(n, M, Lam, p_vac, L,
                                     L_NonEVUnits=units.cm)


fifth_force = Xi_2_ff*field_grad
print("FifthForce =", fifth_force)


'''
    a_coef                          ff(with holes)          ff(without holes)
[0.82, 0.02, 0.85, 3.94],    2.7189761609283968e-11     2.6428497767994413e-11
[0.97, 0.59, 0.03, 3.99],    2.6646134651394897e-11     2.637557210081549e-11
[1.34, 0.18, 0.41, 2.89],    2.7063742787884858e-11     2.743048735808965e-11
[1.47, 0.19, 0.27, 2.63],    2.75899642201886e-11       2.7449752532234442e-11
[2.00, 0.00, 0.00, 0.00],    2.7121768517406257e-11     2.7119103904381668e-11
'''

'''
# At boundary.
4.2065780630507226e-11


'''
