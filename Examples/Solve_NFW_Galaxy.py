#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:36:07 2021

@author: ppycb3

Code to solve for the NFW density profile.
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from Main.MeshingTools import MeshingTools
from Main.SolverChameleon import FieldSolver
from Main.DensityProfiles import DensityProfile


# Define piece-wise density profiles.
r_cutoff = 1e-6
domain_size = 10
critical_density = 1e6
background_density = 1

# Set model parameters.
n = 1
alpha = 1e+9
# alpha = 1e-9


def core(x):
    return critical_density/(r_cutoff*(1+r_cutoff)**2)


def NFW_profile(x):
    r = np.linalg.norm(x)
    return max(critical_density/(r*(1+r)**2), 1)


def field_min(r):
    phi_s = critical_density**(-1/(n+1))

    phi_cutoff = phi_s*((r_cutoff*(1+r_cutoff)**2)**(1/(n+1)))
    phi_background = background_density**(-1/(n+1))

    phi = phi_s*((r*(1+r)**2)**(1/(n+1)))

    # Apply cutoffs.
    phi[x < r_cutoff] = phi_cutoff
    phi[phi > phi_background] = phi_background

    return phi


def NFW_solution(r, g):
    c = critical_density/alpha

    r_c = max(0, (c/g) - 1)
    print('r_c =', r_c)

    phi = field_min(r)

    for i, r_i in enumerate(r):
        if r_i > r_c:
            phi[i] = g*(1 - (r_c/r_i)) + c*(1/r_i)*np.log((1+r_c)/(1+r_i))

    return phi


# Import mesh and convert from .msh to .xdmf.
MT = MeshingTools(dimension=2)


# Define the density profile of the mesh using its subdomains.
p = DensityProfile(filename="../Saved Meshes/NFW_Galaxy",
                   dimension=2, symmetry='vertical axis-symmetry',
                   profiles=[core, NFW_profile, NFW_profile], degree=0)


# Setup problem.
s = FieldSolver(alpha, n, density_profile=p)
# s.tol_du = 1e-11

# Set tolerance on field solutions and solve for above problems.
s.picard()


# Plot results.
dx = 0.01

field_values = s.probe_function(s.field, gradient_vector=np.array([dx, 0]),
                                radial_limit=domain_size)

x = np.array([i*dx for i, _ in enumerate(field_values)])


# Get analytic result.
phi_bg = s.field(domain_size, 0) + 0.1*(critical_density /
                                        alpha)*np.log(1+domain_size)

# field_anal = NFW_solution(r=x, g=1)
field_anal = NFW_solution(r=x, g=phi_bg)

plt.figure()
plt.ylabel(r'$\hat{\phi}$')
plt.xlabel('x')
plt.plot(x, field_values, 'b-', label='Calculated')
plt.plot(x[1:], field_anal[1:], 'r--', label='Analytic')
plt.legend()

# Plot error.
er = (field_values - field_anal)/field_anal

plt.figure()
plt.yscale('log')
plt.ylabel(r'$\delta \hat{\phi}/\hat{\phi}$')
plt.xlabel('x')
plt.plot(x[1:], er[1:])
