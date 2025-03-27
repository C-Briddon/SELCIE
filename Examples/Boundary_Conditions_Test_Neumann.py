#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:52:30 2024

@author: Chad Briddon

Varifying the code gives the correct solution by comparing two solutions for
different sized meshes but the same system. The system is a dense spherical
source in a spherical background.

"""
import numpy as np
import matplotlib.pyplot as plt

from dolfin import BoundaryMesh
from SELCIE import MeshingTools, DensityProfile, FieldSolver


def create_mesh(rv=1.0):

    filename = "BC_Test_Neumann(rv=%f)" % rv

    r = 0.1

    MT = MeshingTools(dimension=2)
    MT.points_to_surface([(r*np.sin(t), r*np.cos(t), 0.0)
                          for t in np.linspace(0, np.pi, 1000)])

    MT.create_subdomain(CellSizeMin=1e-3, CellSizeMax=0.05, DistMax=0.4)

    MT.create_background_mesh(CellSizeMin=1e-3, CellSizeMax=0.05, DistMax=0.4,
                              background_radius=rv, wall_thickness=0.0,
                              symmetry='vertical')

    MT.generate_mesh(filename, show_mesh=True)

    MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)

    return None


if __name__ == "__main__":
    '''To show that the Neumann boundary conditions provide the correct
    solution, first solve the field for the full domain then compare to
    the solution for part of the domain.'''

    # Generate mesh.
    create_mesh(rv=1.0)
    create_mesh(rv=0.5)

    mesh_full = "Neumann_BC_example_mesh(rv=1.000000)"
    mesh_semi = "Neumann_BC_example_mesh(rv=0.500000)"

    # Create density profile class.
    def source(x):
        return 1e17

    def background(x):
        return 1.0

    # Solve field in full domain.
    ###########################################################################
    p = DensityProfile(filename=mesh_full, dimension=2,
                       symmetry='vertical axis-symmetry',
                       profiles=[source, background])

    solver = FieldSolver(alpha=1, n=1, density_profile=p)
    solver.picard(tol_du=1e-12)
    solver.calc_field_grad_vector()

    # Plot field and gradient throughout doamin and on the boundary.
    solver.plot_results(field_scale='linear')

    x1 = np.linspace(0, 1, 1000, endpoint=False)
    y1 = np.array([solver.field(xi, 0) for xi in x1])

    fig, ax = plt.subplots()
    ax.set_ylabel(r'$\hat{\phi}$')
    ax.set_xlabel('x')
    ax.plot(x1, y1, '--', label='Full Domain')

    bmesh = BoundaryMesh(solver.mesh, "exterior")
    theta, grad_norm = [], []
    for x in bmesh.coordinates():
        if x[0] > 1e-10:
            theta.append(np.arccos(x[1]/1)/np.pi)
            grad_norm.append(np.dot(solver.field_grad(x), x)/np.linalg.norm(x))

    plt.figure()
    plt.ylabel(r'$\vec{n} \cdot \vec{\nabla} \phi$')
    plt.xlabel(r'$\theta/\pi$')
    plt.plot(theta, grad_norm, 'b.')

    ###########################################################################

    # Measure gradient at r = 0.5.
    grad = []
    for t in np.linspace(0, np.pi, 1000):
        xi, yi = 0.5*np.sin(t), 0.5*np.cos(t)
        if xi > 1e-10:
            x = np.array([xi, yi])
            grad.append(
                np.dot(solver.field_grad(x[0], x[1]), x)/np.linalg.norm(x))

    grad_new_BC = np.mean(grad)

    # Solve field in semi domain.
    ###########################################################################
    p = DensityProfile(filename=mesh_semi, dimension=2,
                       symmetry='vertical axis-symmetry',
                       profiles=[source, background])

    p.assign_boundary_labels([])

    # Get field profile for reduced domain (with and without Neumann bc).
    Neumann_BC = [None, ('Neumann', str(grad_new_BC))]
    labels = ['Reduced Domain \n(without Neumann)',
              'Reduced Domain \n(with Neumann)']

    x2 = np.linspace(0, 0.5, 1000, endpoint=False)
    for g, label in zip(Neumann_BC, labels):

        solver = FieldSolver(alpha=1, n=1, density_profile=p)

        solver.picard(BCs=[g], tol_du=1e-12)
        solver.calc_field_grad_vector()

        # Plot field and gradient throughout doamin and on the boundary.
        solver.plot_results(field_scale='linear')

        y2 = np.array([solver.field(xi, 0) for xi in x2])
        ax.plot(x2, y2, '-', label=label)

        bmesh = BoundaryMesh(solver.mesh, "exterior")
        theta, grad_norm = [], []
        for x in bmesh.coordinates():
            if x[0] > 1e-10:
                theta.append(np.arccos(x[1]/0.5)/np.pi)
                grad_norm.append(
                    np.dot(solver.field_grad(x), x)/np.linalg.norm(x))

        plt.figure()
        plt.ylabel(r'$\vec{n} \cdot \vec{\nabla} \hat{\phi}$')
        plt.xlabel(r'$\theta/\pi$')
        plt.plot(theta, grad_norm, 'b.')

    ax.axvline(x=0.5, linestyle='--', color='k')
    fig.legend(bbox_to_anchor=(0.9, 0.7))
