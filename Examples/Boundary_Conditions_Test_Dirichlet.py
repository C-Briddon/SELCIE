#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:14:46 2024

@author: Chad Briddon

Show that Dirichlet boundary conditions work.

"""
import numpy as np
import matplotlib.pyplot as plt

from SELCIE import MeshingTools, DensityProfile, FieldSolver


def create_mesh():

    filename = "BC_Test_Dirichlet"

    MT = MeshingTools(dimension=2)
    MT.create_ellipse(rx=0.1, ry=0.1)
    MT.create_subdomain(CellSizeMin=1e-3, CellSizeMax=0.05, DistMax=0.4)

    MT.create_background_mesh(CellSizeMin=1e-3, CellSizeMax=0.05, DistMax=0.4,
                              background_radius=1.0, wall_thickness=0.0,
                              symmetry=None)

    MT.generate_mesh(filename, show_mesh=True)

    MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)

    return None


if __name__ == "__main__":
    # Generate mesh.
    create_mesh()

    # Create density profile class.
    def source(x):
        return 1e17

    def background(x):
        return 1.0

    p = DensityProfile(filename="BC_Test_Dirichlet", dimension=2,
                       symmetry='vertical axis-symmetry',
                       profiles=[source, background])

    p.assign_boundary_labels([])

    # Solve field equations with and without Dirichlet boundary conditions.
    Dirichlet_BC = [None, ('Dirichlet', '1e-8')]
    labels = ['BC=None', 'BC=$10^{-8}$']

    fig, ax = plt.subplots()
    ax.set_ylabel(r'$\hat{\phi}$')
    ax.set_xlabel('x')
    ax.axhline(y=1e-8, color='k', linestyle='--',
               label=r'$\hat{\phi}=10^{-8}$')

    for g, label in zip(Dirichlet_BC, labels):

        # Solve for field.
        solver = FieldSolver(alpha=1, n=1, density_profile=p)
        solver.picard(BCs=[g], tol_du=1e-12)

        # Plot field and gradient throughout doamin and on the boundary.
        solver.plot_results(field_scale='linear')

        # Plot field.
        x = np.linspace(0, 1, 1000)
        y = np.array([solver.field(xi, 0) for xi in x])
        ax.plot(x, y, label=label)

    fig.legend(bbox_to_anchor=(0.9, 0.75))
