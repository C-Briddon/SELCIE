#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:16:36 2021

@author: Chad Briddon

Solve the chameleon field around a shape constructed from Legndre polynomials,
inside a vacuum chamber. The values in this example were taken from
'arXiv:1711.02065'.

To generate mesh run 'CreateMesh_Legendre.py'.
"""
import numpy as np
import matplotlib.pyplot as plt

from astropy import units
from SELCIE import FieldSolver
from SELCIE import DensityProfile
from SELCIE.Misc import conv_fifth_force_chameleon, alpha_calculator_chameleon


# Define density profile functions.
def source_wall(x):
    return 1.0e17


def vacuum(x):
    return 1.0


def constraint(x):
    return np.linalg.norm(x) < 0.9


# Set model parameters.
M = 1e27                            # eV
Lam = 1.0e-3                        # eV
p_vac = 43.10130531853594           # eV4
L = 15.0                            # cm
n = 1

alpha = alpha_calculator_chameleon(n, M, Lam, p_vac, L, L_NonEVUnits=units.cm)

a_coef = np.array([0.82, 0.02, 0.85, 3.94])/15
# a_coef = np.array([0.97, 0.59, 0.03, 3.99])/15
# a_coef = np.array([1.34, 0.18, 0.41, 2.89])/15
# a_coef = np.array([1.47, 0.19, 0.27, 2.63])/15
# a_coef = np.array([2.00, 0.00, 0.00, 0.00])/15

# Import mesh and setup density field.
p = DensityProfile(filename="Legndre" + str(a_coef),
                   dimension=2, symmetry='vertical axis-symmetry',
                   profiles=[source_wall, vacuum, vacuum, source_wall])


# Setup problem.
s = FieldSolver(alpha, n, density_profile=p)

s.calc_density_field()
s.plot_results(density_scale='linear')


# Solve for field and find position on boundary where fifth force is maximised.
s.picard()
s.calc_field_grad_mag()
field_grad, probe_point = s.maximise_function(s.field_grad_mag, subdomain=2,
                                              check_boundary_only=True,
                                              constraint=constraint)

s.plot_results(field_scale='log', grad_scale='log')
plt.plot(probe_point[0], probe_point[1], 'rx')
plt.ylim([-0.5, 0.5])
plt.xlim([-0.5, 0.5])


# Rescale fifth_force_max into units of g.
Xi_2_ff = conv_fifth_force_chameleon(n, M, Lam, p_vac, L,
                                     L_NonEVUnits=units.cm)

fifth_force = Xi_2_ff*field_grad
print("FifthForce =", fifth_force)
