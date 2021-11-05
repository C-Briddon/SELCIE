#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:25:00 2021

@author: Chad Briddon

Solve the chameleon field for a sphere and infinitely long cylinder inside a
vacuum.
"""
import numpy as np
import dolfin as d
from scipy.special import kn
import matplotlib.pyplot as plt

from SELCIE import FieldSolver
from SELCIE import DensityProfile


# Analytic solutions.
def solution_sphere(alpha, n, source_density, R, r):
    source_field_min = pow(source_density, -1/(n+1))
    R_roll = R - alpha*(1 - source_field_min)/source_density
    sol = []

    for r_i in r:
        if r_i < R_roll:
            sol.append(source_field_min)
        elif r_i > R:
            sol.append(1 - (R/r_i)*np.exp(-(r_i-R)*np.sqrt((n+1)/alpha)))
        else:
            sol.append((source_density/(3*alpha))*((r_i**2)/2
                                                   + (R_roll**3)/r_i
                                                   - 1.5*R_roll**2)
                       + source_field_min)

    return np.array(sol)


def solution_cylinder(alpha, n, source_density, R, r):
    source_field_min = pow(source_density, -1/(n+1))
    sol = []

    for r_i in r:
        if r_i <= R:
            sol.append(source_field_min)
        elif r_i > R:
            sol.append(1+kn(0, r_i*np.sqrt((n+1)/alpha))/np.log(R*np.sqrt(
                (n+1)/alpha)/2))

    return np.array(sol)


# Define density profile functions.
def source_wall(x):
    return 1.0e17


def vacuum(x):
    return 1.0


# Set model parameters.
n = 1
alpha = 0.1
R = 0.005


# Define the density profile of the mesh using its subdomains.
p_sphere = DensityProfile(filename="Circle_in_Vacuum_r" + str(R),
                          dimension=2, symmetry='vertical axis-symmetry',
                          profiles=[source_wall, vacuum, vacuum])

p_cylinder = DensityProfile(filename="Circle_in_Vacuum_r" + str(R),
                            dimension=2, symmetry='cylinder slice',
                            profiles=[source_wall, vacuum, vacuum])


# Setup problem.
s_sphere = FieldSolver(alpha, n, density_profile=p_sphere)
s_cylinder = FieldSolver(alpha, n, density_profile=p_cylinder)


# Set tolerance on field solutions and solve for above problems.
s_sphere.tol_du = 1e-10
s_cylinder.tol_du = 1e-10

s_sphere.picard()
s_cylinder.picard()


# Plot calculated value against the analytic solution and measure difference.
dx = 0.01
R_max = 1.0 - 1e-4

calculated_field_sphere = []
calculated_field_cylinder = []
N = 100

for t in np.linspace(0, 2*np.pi, N, endpoint=False):
    dr = dx*np.array([np.cos(t), np.sin(t)])

    calculated_field_sphere.append(
        s_sphere.probe_function(function=s_sphere.field,
                                gradient_vector=dr,
                                radial_limit=R_max))

    calculated_field_cylinder.append(
        s_cylinder.probe_function(function=s_cylinder.field,
                                  gradient_vector=dr,
                                  radial_limit=R_max))

X = np.array([i*dx for i in range(N)])
analytic_field_sphere = solution_sphere(alpha, n, source_wall(0), R, r=X)
analytic_field_cylinder = solution_cylinder(alpha, n, source_wall(0), R, r=X)


# Plot field profiles against analytic solutions.
plt.rc('axes', titlesize=10)                # fontsize of the axes title
plt.rc('axes', labelsize=14)                # fontsize of the x and y labels
plt.rc('legend', fontsize=13)               # legend fontsize

plt.figure(figsize=[5.8, 4.0], dpi=150)
plt.ylabel(r"$\hat{\phi}$")
plt.xlabel(r"$\hat{r}$")

for cs in calculated_field_sphere[1:]:
    plt.plot(X, cs, 'k.')

for cc in calculated_field_cylinder[1:]:
    plt.plot(X, cc, 'kx')

plt.plot(X, calculated_field_sphere[0], 'k.',
         label=r"Calculated $\hat{\phi}_{Sphere}$")
plt.plot(X, analytic_field_sphere, 'b-',
         label=r"Analytic $\hat{\phi}_{Sphere}$")
plt.plot(X, calculated_field_cylinder[0], 'kx',
         label=r"Calculated $\hat{\phi}_{Cylinder}$")
plt.plot(X, analytic_field_cylinder, 'r-',
         label=r"Analytic $\hat{\phi}_{Cylinder}$")
plt.legend()


# Plot relative errors between analytic and calculated solutions.
plt.figure()
plt.ylabel(r"$\delta \hat{\phi}/\hat{\phi}$")
plt.xlabel(r"$\hat{r}$")

for cs in calculated_field_sphere[1:]:
    err_sphere = (cs - analytic_field_sphere)/analytic_field_sphere
    plt.plot(X, err_sphere, 'r-')

for cc in calculated_field_cylinder[1:]:
    err_cylinder = (cc - analytic_field_cylinder)/analytic_field_cylinder
    plt.plot(X, err_sphere, 'r-')


err_sphere = (calculated_field_sphere[0]
              - analytic_field_sphere)/analytic_field_sphere

err_cylinder = (calculated_field_cylinder[0]
                - analytic_field_cylinder)/analytic_field_cylinder

plt.plot(X, err_sphere, 'r-', label="Sphere")
plt.plot(X, err_cylinder, 'b-', label="Cylinder")
plt.legend()


# Make image of subdomains of the mesh with zoomed in image of the source.
D = 1e-2
G = 1.15

fig = plt.figure(figsize=(6, 4))

ax_pos_0 = [0.10, 0.10, 0.85, 0.85]
ax_pos_1 = [0.55, 0.65, 0.25, 0.25]

ax0 = fig.add_axes(ax_pos_0)
plt.ylim([-G, G])
plt.xlim([-G, G])
plt.ylabel('y')
plt.xlabel('x')
d.plot(p_sphere.subdomains)
plt.plot([-D, D, D, -D, -D], [-D, -D, D, D, -D], 'r-')
plt.plot([-D, 0.265], [+D, 1.00], 'r-')
plt.plot([+D, 0.95], [-D, 0.325], 'r-')

ax1 = fig.add_axes(ax_pos_1)
plt.ylim([-D, D])
plt.xlim([-D, D])
d.plot(p_sphere.subdomains)
