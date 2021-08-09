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

import sys
sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools
from Main.Solver_Chameleon import Field_Solver
from Main.Density_Profiles import vacuum_chamber_density_profile


# Define the approximate analytic solution.
def Q0(x):
    return 0.5*np.log((x+1)/(x-1))


def Q2(x):
    return 0.25*(3*x**2 - 1)*np.log((x+1)/(x-1)) - 1.5*x


def P2(x):
    return 0.5*(3*x**2 - 1)


def analytic_solution_ellipse(Xi0, Xi, eta, alpha, n):
    phi_bg = min(0.6940480354297778*pow(alpha, -1/(n+2)), 1)
    phi = phi_bg*(1 - (Q0(Xi) - P2(eta)*Q2(Xi))/Q0(Xi0))
    return phi


# Import mesh and convert from .msh to .xdmf.
Xi0 = 1.01
r0 = 0.005
a = r0/((Xi0*(Xi0**2 - 1))**(1/3))
MT = Meshing_Tools()
filename = "../Saved Meshes/Ellipse_in_Vacuum_r%f_Xi%f" %(r0, Xi0)
mesh, subdomains, boundaries = MT.msh_2_xdmf(filename, dim=2)


# Set model parameters.
n = 1
alpha = 1e6
ps = 1e17


# Define the density profile of the mesh using its subdomains.
p = vacuum_chamber_density_profile(mesh = mesh, 
                                   subdomain_markers = subdomains, 
                                   source_density = ps, 
                                   vacuum_density = 1, 
                                   wall_density = ps, 
                                   mesh_symmetry = 'horizontal axis-symmetry', 
                                   degree = 0)


# Setup problem.
s = Field_Solver("name", alpha = alpha, n = n, density_profile = p)


# Set tolerance on field solutions and solve for above problems.
#s.tol_du = 1e-5

s.picard()



# Plot calculated value against the analytic solution and measure difference.
XiMax = 1/a
Xi = np.linspace(Xi0, XiMax, 1000)

calculated_field_ellipse = []

Eta = np.linspace(0, 1, 5)

for eta in Eta:
    calculated_field_ellipse.append([])
    
    x = a*np.sqrt((Xi**2 - 1)*(1 - eta**2))
    z = a*eta*Xi
    
    for x_i, z_i in zip(x, z):
        calculated_field_ellipse[-1].append(s.field(z_i, x_i))


plt.figure()
plt.title(r'$\xi_0$ = %f & $\alpha$ = %.e' %(Xi0,alpha))
plt.ylabel("$\hat{\phi}$")
plt.xlabel(r"$\xi$")
#plt.ylim([0,1e-6])

for line, eta in zip(calculated_field_ellipse, Eta):
    plt.plot(Xi, line, label = r"Calculated $\eta$ = %f" %eta)
    
    analytic_field_ellipse = analytic_solution_ellipse(Xi0, Xi, eta, alpha, n)
    plt.plot(Xi, analytic_field_ellipse, marker = '.', 
             label = r"Analytic $\eta$ = %f" %eta)

plt.legend()

plt.figure()
plt.title(r'$\xi_0$ = %f & $\alpha$ = %.e' %(Xi0,alpha))
plt.ylabel("error")
plt.xlabel(r"$\xi$")
plt.ylim([0,0.1])
plt.xlim([0,XiMax/2])
for line, eta in zip(calculated_field_ellipse, Eta):
    analytic_field_ellipse = analytic_solution_ellipse(Xi0, Xi, eta, alpha, n)
    error = abs(line - analytic_field_ellipse)/line
    plt.plot(Xi, error, label = r"Error $\eta$ = %f" %eta)
plt.legend()

