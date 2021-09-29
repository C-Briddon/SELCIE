#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:29:06 2021

@author: ppycb3

Environment - fenics2019

Solve the chameleon field around an ellipse and compare to the approximate
analytic solution.
"""
import numpy as np
import matplotlib.pyplot as plt
from Density_profiles import source_wall, vacuum

import sys
sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools
from Main.Solver_Chameleon import Field_Solver
from Main.Density_Profiles import Density_Profile


# Legendre functions of the first and second kind.
def Q0(x):
    return 0.5*np.log((x+1)/(x-1))


def Q2(x):
    return 0.25*(3*x**2 - 1)*np.log((x+1)/(x-1)) - 1.5*x


def P2(x):
    return 0.5*(3*x**2 - 1)


# Analytic solutions.
def solution_ellipse(Xi0, Xi, eta, alpha, n):
    phi_bg = min(0.6940480354297778*pow(alpha, -1/(n+2)), 1)
    phi = phi_bg*(1 - (Q0(Xi) - P2(eta)*Q2(Xi))/Q0(Xi0))
    return phi


# Import mesh and convert from .msh to .xdmf.
Xi0 = 1.01
r0 = 0.005
a = r0/((Xi0*(Xi0**2 - 1))**(1/3))
MT = Meshing_Tools(Dimension=2)
filename = "../Saved Meshes/Ellipse_in_Vacuum_r%f_Xi%f" % (r0, Xi0)
mesh, subdomains, boundaries = MT.msh_2_xdmf(filename)


# Define the density profile of the mesh using its subdomains.
p = Density_Profile(mesh=mesh, subdomain_markers=subdomains,
                    mesh_symmetry='horizontal axis-symmetry',
                    profiles=[source_wall, vacuum, source_wall], degree=0)


# Setup problem.
alpha = 1.0e3
n = 1
s = Field_Solver(alpha, n, density_profile=p)

# Set tolerance on field solutions and solve for above problems.
s.tol_du = 1.0e-14
s.picard()


# Plot calculated value against the analytic solution and measure difference.
XiMax = 1/a
Xi = np.linspace(Xi0, XiMax, 1000)

calculated_field_ellipse = []
colors = ['g', 'b', 'r', 'k', 'c']
ls = 1
ms = 1

Eta = np.linspace(0, 1, 3)
xi_max_plot = 40

for eta in Eta:
    calculated_field_ellipse.append([])

    x = a*np.sqrt((Xi**2 - 1)*(1 - eta**2))
    z = a*eta*Xi

    for x_i, z_i in zip(x, z):
        calculated_field_ellipse[-1].append(s.field(z_i, x_i))

plt.figure()
plt.title(r'$\xi_0$ = %.2f & $\alpha$ = %.e' % (Xi0, alpha))
plt.ylabel(r"$\hat{\phi}$")
plt.xlabel(r"$\xi$")
plt.ylim([0, 0.07])
plt.xlim([Xi0, xi_max_plot])
plt.xscale('log')

i = 0
for line, eta in zip(calculated_field_ellipse, Eta):
    plt.plot(Xi, line, '-', markersize=ls, color=colors[i],
             label=r"Calculated $\eta$ = %.2f" % eta)
    analytic_field_ellipse = solution_ellipse(Xi0, Xi, eta, alpha, n)
    plt.plot(Xi, analytic_field_ellipse, '--', markersize=ms,
             color=colors[i], label=r"Analytic $\eta$ = %.2f" % eta)
    i += 1

plt.legend()

plt.figure()
plt.title(r'$\xi_0$ = %.2f & $\alpha$ = %.e' % (Xi0, alpha))
plt.ylabel("relative error")
plt.xlabel(r"$\xi$")
plt.ylim([0, 0.1])
plt.xlim([Xi0, xi_max_plot])
plt.xscale('log')

i = 0
for line, eta in zip(calculated_field_ellipse, Eta):
    analytic_field_ellipse = solution_ellipse(Xi0, Xi, eta, alpha, n)
    error = abs(line - analytic_field_ellipse)/line
    plt.plot(Xi, error, color=colors[i], label=r"Error $\eta$ = %f" % eta)
    i += 1
plt.legend()


# Get fifth force measure.
s.calc_field_grad_mag()
field_grad, probe_point = s.measure_fifth_force(boundary_distance=0.01,
                                                tol=1e-4)

s.plot_results(field_scale='linear')
plt.plot(probe_point.x(), probe_point.y(), 'rx')
plt.ylim([-0.05, 0.05])
plt.xlim([-0.05, 0.05])


# Add marker to plot indicating location of maximum fifth force.
plt.plot(probe_point.x(), probe_point.y(), 'rx')
print('Maximum field gradient is %f and is at position (%f, %f).'
      % (field_grad, probe_point.x(), probe_point.y()))
