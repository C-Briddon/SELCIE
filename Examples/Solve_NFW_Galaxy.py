#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:36:07 2021

@author: Chad Briddon

Code to solve the chameleon field for the NFW density profile.
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from SELCIE.MeshingTools import MeshingTools
from SELCIE.SolverChameleon import FieldSolver
from SELCIE.DensityProfiles import DensityProfile


# Define functions for analytic solution.
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


def NFW_solution(critical_density, alpha, phi_bg, r):
    c = critical_density/alpha

    r_c = max(0, (c/phi_bg) - 1)

    phi = field_min(r)

    for i, r_i in enumerate(r):
        if r_i > r_c:
            phi[i] = phi_bg*(1 - (r_c/r_i)) + c*(1/r_i)*np.log((1+r_c)/(1+r_i))

    return phi


# Set model parameters.
r_cutoff = 1e-6
domain_size = 10
critical_density = 1e6
background_density = 1

n = 1
alpha = [1e-9, 1e+9]


# Import mesh and convert from .msh to .xdmf.
MT = MeshingTools(dimension=2)

p = DensityProfile(filename="NFW_Galaxy", dimension=2,
                   symmetry='vertical axis-symmetry',
                   profiles=[core, NFW_profile, NFW_profile])


# Run problem for both alpha values.
s0 = FieldSolver(alpha[0], n, density_profile=p)
s1 = FieldSolver(alpha[1], n, density_profile=p)

s0.picard()
s1.picard()

dx = 0.01

field_values_0 = s0.probe_function(s0.field, gradient_vector=np.array([dx, 0]),
                                   radial_limit=domain_size)

field_values_1 = s1.probe_function(s1.field, gradient_vector=np.array([dx, 0]),
                                   radial_limit=domain_size)


# Get analytic result.
x = np.array([i*dx for i, _ in enumerate(field_values_0)])

phi_bg = [1, s1.field(domain_size, 0) + 0.1*(critical_density /
                                             alpha[1])*np.log(1+domain_size)]

field_anal = [NFW_solution(critical_density, alpha[0], phi_bg[0], r=x),
              NFW_solution(critical_density, alpha[1], phi_bg[1], r=x)]


# Plot analytic against calculated results.
plt.rc('axes', titlesize=10)                # fontsize of the axes title
plt.rc('axes', labelsize=14)                # fontsize of the x and y labels
plt.rc('legend', fontsize=13)               # legend fontsize

plt.figure(figsize=[5.8, 4.0], dpi=150)
plt.title(r'$\alpha = 10^{%i}$' % np.log10(alpha[0]))
plt.ylabel(r'$\hat{\phi}$')
plt.xlabel(r'$\hat{r}$')
plt.plot(x, field_values_0, 'b-', label='Calculated')
plt.plot(x[1:], field_anal[0][1:], 'r--', label='Analytic')
plt.legend()

plt.figure(figsize=[5.8, 4.0], dpi=150)
plt.title(r'$\alpha = 10^{%i}$' % np.log10(alpha[1]))
plt.ylabel(r'$\hat{\phi}$')
plt.xlabel(r'$\hat{r}$')
plt.plot(x, field_values_1, 'b-', label='Calculated')
plt.plot(x[1:], field_anal[1][1:], 'r--', label='Analytic')
plt.legend()


# Plot error.
er_0 = (field_values_0 - field_anal[0])/field_anal[0]

er_1 = (field_values_1 - field_anal[1])/field_anal[1]

plt.figure()
plt.title(r'$\alpha = 10^{%i}$' % np.log10(alpha[0]))
plt.ylabel(r'$\delta \hat{\phi}/\hat{\phi}$')
plt.yscale('log')
plt.xlabel('x')
plt.plot(x[1:], er_0[1:])

plt.figure()
plt.title(r'$\alpha = 10^{%i}$' % np.log10(alpha[1]))
plt.ylabel(r'$\delta \hat{\phi}/\hat{\phi}$')
plt.yscale('log')
plt.xlabel('x')
plt.plot(x[1:], er_1[1:])
