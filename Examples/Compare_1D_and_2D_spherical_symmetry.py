#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:48:17 2024

@author: chad-briddon

Compare 1D and 2D solutions with spherical symmetry.
"""

import numpy as np
import matplotlib.pyplot as plt

from os.path import isdir
from SELCIE import MeshingTools, DensityProfile, FieldSolver


def create_mesh_hollow_chamber_1D():

    filename = "hollow_chamber_1D"
    file_path = "Saved Meshes/" + filename

    # Skip function if mesh was previously made.
    if isdir(file_path):
        return filename

    # Make mesh.
    MT = MeshingTools(dimension=1, display_messages=False)

    vac = MT.create_1D_line(x_start=0, x_end=0.5)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.01, DistMax=0.1)

    MT.create_1D_line(x_start=0, x_end=0.55, embed=vac)
    MT.create_subdomain(CellSizeMin=1e-3, CellSizeMax=0.01, DistMax=0.1)

    MT.generate_mesh(filename, show_mesh=True)
    MT.msh_2_xdmf(filename, auto_override=True)

    return filename


def create_mesh_hollow_chamber_2D():

    filename = "hollow_chamber_2D"
    file_path = "Saved Meshes/" + filename

    # Skip function if mesh was previously made.
    if isdir(file_path):
        return filename

    # Make mesh.
    MT = MeshingTools(dimension=2, display_messages=False)

    MT.create_background_mesh(CellSizeMin=1e-4, CellSizeMax=0.01, DistMax=0.1,
                              background_radius=0.5, wall_thickness=0.05,
                              symmetry='vertical')

    MT.generate_mesh(filename, show_mesh=True)
    MT.msh_2_xdmf(filename, auto_override=True)

    return filename


def vacuum(x):
    return 1


def wall(x):
    return 1e17


# Choose symmetry.
print("Choose : spherical vacuum chamber    (1)")
print("       : cylindrical vacuum chamber  (2)")
print()
sym_choice = input("1 or 2 : ")
print()

if sym_choice == '1':
    sym_1D, sym_2D = "spherical symmetry", "vertical axis-symmetry"
    print("'spherical vacuum chamber' chosen.")
elif sym_choice == '2':
    sym_1D, sym_2D = "cylindrical symmetry", "translation symmetry"
    print("'cylindrical vacuum chamber' chosen.")
else:
    print("Note a valid choice. Defaulted to 'spherical vacuum chamber (1)'.")
print()


# Calculate 1D solution.
filename_1D = create_mesh_hollow_chamber_1D()

p_1D = DensityProfile(filename=filename_1D, dimension=1,
                      symmetry=sym_1D, profiles=[vacuum, wall])

s_1D = FieldSolver(alpha=1, n=1, density_profile=p_1D)

s_1D.calc_density_field()
s_1D.picard(display_progress=True)
s_1D.plot_results(field_scale='linear', density_scale='linear')


# Calculate solution using 2D mesh to compare to 1D solution above.
filename_2D = create_mesh_hollow_chamber_2D()

p_2D = DensityProfile(filename=filename_2D, dimension=2,
                      symmetry=sym_2D, profiles=[vacuum, wall])

s_2D = FieldSolver(alpha=1, n=1, density_profile=p_2D)

s_2D.calc_density_field()
s_2D.picard(display_progress=True, tol_du=1e-11)
s_2D.plot_results(field_scale='linear', density_scale='linear')

# Make plot comparing fields.
x = np.linspace(0, 0.55, 1000)
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
