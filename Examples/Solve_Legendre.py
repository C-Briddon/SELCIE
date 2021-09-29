#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:16:36 2021

@author: ppycb3

Environment - fenics2019

Solve the chameleon field around a Legndre polynomial shape.
"""
import numpy as np
from astropy import units
import matplotlib.pyplot as plt
from Density_profiles import source_wall, vacuum

import sys
sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools
from Main.Solver_Chameleon import Field_Solver
from Main.Density_Profiles import Density_Profile
from Main.Misc import conv_fifth_force_Chameleon

# Import mesh and convert from .msh to .xdmf.
a_coef = np.array([0.82, 0.02, 0.85, 3.94])/15
# a_coef = np.array([0.97, 0.59, 0.03, 3.99])/15
# a_coef = np.array([1.34, 0.18, 0.41, 2.89])/15
# a_coef = np.array([1.47, 0.19, 0.27, 2.63])/15
# a_coef = np.array([2.00, 0.00, 0.00, 0.00])/15

MT = Meshing_Tools(Dimension=2)
filename = "../Saved Meshes/Legndre" + str(a_coef)
mesh, subdomains, boundaries = MT.msh_2_xdmf(filename)


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
p = Density_Profile(mesh=mesh, subdomain_markers=subdomains,
                    mesh_symmetry='horizontal axis-symmetry',
                    profiles=[source_wall, vacuum, source_wall], degree=0)


# Setup problem.
s = Field_Solver(alpha, n, density_profile=p)


# Set tolerance on field solutions and solve for above problems.
s.picard()
s.calc_field_grad_mag()

field_grad, probe_point = s.measure_fifth_force(boundary_distance=d, tol=d_tol)

s.plot_results(field_scale='log', grad_scale='log', res_scale='log')
plt.plot(probe_point.x(), probe_point.y(), 'rx')

'''
import dolfin as d
plt.figure()
plt.ylabel('Y')
plt.xlabel('X')
d.plot(subdomains)
'''


# Rescale fifth_force_max into units of g.
M = 1e27                        # eV
Lam = 1.0e-3                        # eV
p_vac = 43.10130531853594           # eV4
L = 15.0                            # cm
Xi_2_ff = conv_fifth_force_Chameleon(n, M, Lam, p_vac, L,
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
