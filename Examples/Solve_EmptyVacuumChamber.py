#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:00:59 2021

@author: ppycb3

Environment - fenics2019

Example solving the chamelon field for an empty vacuum chamber.
"""
import numpy as np
import matplotlib.pyplot as plt
import dolfin as d

import sys
sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools
from Main.Solver_Chameleon import Field_Solver
from Main.Density_Profiles import vacuum_chamber_density_profile

# Import mesh and convert from .msh to .xdmf.
MT = Meshing_Tools()
filename = "../Saved Meshes/Circle_Empty_Vacuum_chamber"
mesh, subdomains, boundaries = MT.msh_2_xdmf(filename, dim=2)


# Set model parameters.
n = 1
alpha = [1e6, 1e12, 1e18]
pw = 1e17


# Define the density profile of the mesh using its subdomains.
p = vacuum_chamber_density_profile(mesh = mesh, subdomain_markers = subdomains, 
                                   source_density = 1,
                                   vacuum_density = 1, 
                                   wall_density = pw, 
                                   mesh_symmetry = 'vertical axis-symmetry', 
                                   degree = 0)


# Plot rescaled field for range of alpha values.
dx = 0.01
R_max = 1.1 - 1e-4

phi_0 = []
lbs = ['r-', 'k--', 'c.']

plt.figure()
plt.ylabel("$\hat{\phi}$")
plt.xlabel("$\hat{r}$")

for i, a in enumerate(alpha):
    # Setup problem.
    s = Field_Solver("name", alpha = a, n = n, density_profile = p)
    
    
    # Set tolerance on field solutions and solve for above problems.
    s.tol_du = 1e-10
    s.picard()
    
    
    for t in np.linspace(0, 2*np.pi, 100, endpoint = False):
        dr = dx*np.array([np.cos(t),np.sin(t)])
        
        calculated_field = s.probe_function(function = s.field, gradient_vector = dr, 
                                            radial_limit = R_max)
        
        X = np.array([i*dx for i, _ in enumerate(calculated_field)])
        rescaled_field = pow(s.alpha, 1/(s.n+2))*calculated_field
        
        plt.plot(X, rescaled_field, lbs[i], label = r"$\alpha $ = {:e}".format(a),
                 linewidth = 1, markersize = 3)
    
    phi_0.append(pow(s.alpha, 1/(s.n+2))*s.field(0.0, 0.0))


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

print(phi_0)


"Make image of subdomains."
G = 1.15

fig = plt.figure(figsize=(6, 4))

ax_pos_0 = [0.10, 0.10, 0.85, 0.85]
ax_pos_1 = [0.55, 0.65, 0.25, 0.25]

ax0 = fig.add_axes(ax_pos_0)
plt.ylim([-G, G])
plt.xlim([-G, G])
plt.ylabel('Y')
plt.xlabel('X')
d.plot(subdomains)

plt.show()