#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:41:48 2024

@author: chad-briddon

Compare 1D and 2D solutions with translational symmetry.
"""

import numpy as np
import matplotlib.pyplot as plt

from os.path import isdir
from SELCIE import MeshingTools, DensityProfile, FieldSolver


def create_mesh_parallel_plates_1D():

    filename = "parallel_plates_1D"
    file_path = "Saved Meshes/" + filename

    # Skip function if mesh was previously made.
    if isdir(file_path):
        return filename

    # Make mesh.
    MT = MeshingTools(dimension=1, display_messages=False)

    vac = MT.create_1D_line(x_start=-0.5, x_end=+0.5)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.01, DistMax=0.1)

    MT.create_1D_line(x_start=-0.55, x_end=+0.55, embed=vac)
    MT.create_subdomain(CellSizeMin=1e-3, CellSizeMax=0.01, DistMax=0.1)

    MT.generate_mesh(filename, show_mesh=True)
    MT.msh_2_xdmf(filename, auto_override=True)

    return filename


def create_mesh_parallel_plates_2D():

    filename = "parallel_plates_2D"
    file_path = "Saved Meshes/" + filename

    # Skip function if mesh was previously made.
    if isdir(file_path):
        return filename

    # Make mesh.
    MT = MeshingTools(dimension=2, display_messages=False)

    vac = MT.points_to_surface([(-0.5, -0.5, 0), (+0.5, -0.5, 0),
                                (+0.5, +0.5, 0), (-0.5, +0.5, 0)])

    MT.create_subdomain(CellSizeMin=1e-4, CellSizeMax=0.01, DistMax=0.1)

    MT.points_to_surface([(-0.55, -0.5, 0), (+0.55, -0.5, 0),
                          (+0.55, +0.5, 0), (-0.55, +0.5, 0)], embed=vac)
    MT.create_subdomain(CellSizeMin=1e-3, CellSizeMax=0.01, DistMax=0.1)

    MT.generate_mesh(filename, show_mesh=True)
    MT.msh_2_xdmf(filename, auto_override=True)

    return filename


def vacuum(x):
    return 1


def wall(x):
    return 1e17


# Calculate 1D solution.
filename_1D = create_mesh_parallel_plates_1D()

p_1D = DensityProfile(filename=filename_1D, dimension=1,
                      symmetry="translation symmetry", profiles=[vacuum, wall])

s_1D = FieldSolver(alpha=1, n=1, density_profile=p_1D)

s_1D.calc_density_field()
s_1D.picard(display_progress=True)
s_1D.plot_results(field_scale='linear', density_scale='linear')


# Calculate solution using 2D mesh to compare to 1D solution above.
filename_2D = create_mesh_parallel_plates_2D()

p_2D = DensityProfile(filename=filename_2D, dimension=2,
                      symmetry="translation symmetry", profiles=[vacuum, wall])

s_2D = FieldSolver(alpha=1, n=1, density_profile=p_2D)

s_2D.calc_density_field()
s_2D.tol_du = 1e-11
s_2D.picard(display_progress=True)
s_2D.plot_results(field_scale='linear', density_scale='linear')

# Make plot comparing fields.
x = np.linspace(-0.55, 0.55, 1000)
field_1D = np.array([s_1D.field(xi, 0) for xi in x])
field_2D = np.array([s_2D.field(xi, 0) for xi in x])
df = abs(field_1D - field_2D)

plt.figure()
plt.ylabel(r"$\hat{\phi}$")
plt.xlabel("x")
plt.plot(x, field_1D, label="1D")
plt.plot(x, field_2D, label="2D")

plt.figure()
plt.ylabel(r"$\delta \hat{\phi}$")
plt.xlabel("x")
plt.yscale('log')
plt.plot(x, df)
