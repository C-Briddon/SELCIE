#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:00:59 2021

@author: Chad Briddon

Example solving the chamelon field for an empty vacuum chamber for a range of
alpha values to test the alpha dependence of the field.

To generate mesh run 'CreateMesh_Empty_Vacuum_Chamber.py'.
"""
import numpy as np
import matplotlib.pyplot as plt

from SELCIE import FieldSolver
from SELCIE import DensityProfile


# Define density profile functions.
def source_wall(x):
    return 1.0e17


def vacuum(x):
    return 1.0


# Set model parameters.
n = 1
alpha = [1e6, 1e12, 1e18]


# Define the density profile of the mesh using its subdomains.
p = DensityProfile(filename="Circle_Empty_Vacuum_chamber",
                   dimension=2, symmetry='vertical axis-symmetry',
                   profiles=[vacuum, source_wall])

exit()


# Plot rescaled field for range of alpha values.
dx = 0.01
R_max = 1.1 - 1e-4

phi_0 = []
lbs = ['r--', 'g--', 'k--']

plt.rc('axes', titlesize=10)        # fontsize of the axes title
plt.rc('axes', labelsize=14)        # fontsize of the x and y labels
plt.rc('legend', fontsize=13)       # legend fontsize

plt.figure(figsize=[5.8, 4.0], dpi=150)
plt.ylabel(r"$\hat{\varphi}$")
plt.xlabel(r"$\hat{r}$")

for i, a in enumerate(alpha):
    # Setup problem.
    s = FieldSolver(alpha=a, n=n, density_profile=p)

    # Set tolerance on field solutions and solve for above problems.
    s.tol_du = 1e-10
    s.picard()

    for t in np.linspace(0, 2*np.pi, 100, endpoint=False):
        dr = dx*np.array([np.cos(t), np.sin(t)])

        calculated_field = s.probe_function(function=s.field,
                                            gradient_vector=dr,
                                            radial_limit=R_max)

        X = np.array([i*dx for i, _ in enumerate(calculated_field)])
        rescaled_field = pow(s.alpha, 1/(s.n+2))*calculated_field

        plt.plot(X, rescaled_field, lbs[i],
                 label=r"$\alpha = 10^{%i}$" % int(np.log10(a)),
                 linewidth=1, markersize=3)

    phi_0.append(pow(s.alpha, 1/(s.n+2))*s.field(0.0, 0.0))


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Print rescaled field values at centre of the chamber.
print(phi_0)
