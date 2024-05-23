#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:55:00 2024

@author: Chad Briddon

Demonstrate how to define different boundary regions and assign boundary
conditions to each.

"""
import matplotlib.pyplot as plt

from dolfin import BoundaryMesh
from SELCIE import MeshingTools, DensityProfile, FieldSolver


def create_mesh():

    filename = "BC_example_Neumann"
    MT = MeshingTools(dimension=2)

    MT.create_rectangle(dx=1.0, dy=1.0)
    MT.create_subdomain(CellSizeMin=1e-3, CellSizeMax=0.05, DistMax=0.4)

    MT.generate_mesh(filename, show_mesh=True)
    MT.msh_2_xdmf(filename, delete_old_file=True, auto_override=True)

    return filename


def background(x):
    return 1.0


def plot_boundary(solver):
    "Plot field and components of the field's gradient at the boundary."

    # Make plot for field at boundaries.
    bmesh = BoundaryMesh(solver.mesh, "exterior")
    vertex_coordinates = bmesh.coordinates()

    x_left, x_right, x_top, x_bottom = [], [], [], []
    b_left, b_right, b_top, b_bottom = [], [], [], []
    gx_left, gx_right, gy_top, gy_bottom = [], [], [], []

    for vc in vertex_coordinates:
        if vc[0] < -0.4999:
            x_left.append(vc[1])
            b_left.append(solver.field(vc))
            gx_left.append(-solver.field_grad(vc)[0])

        elif vc[0] > 0.4999:
            x_right.append(vc[1])
            b_right.append(solver.field(vc))
            gx_right.append(+solver.field_grad(vc)[0])

        elif vc[1] > 0.4999:
            x_top.append(vc[0])
            b_top.append(solver.field(vc))
            gy_top.append(+solver.field_grad(vc)[1])

        elif vc[1] < -0.4999:
            x_bottom.append(vc[0])
            b_bottom.append(solver.field(vc))
            gy_bottom.append(-solver.field_grad(vc)[1])

    # Order plots.
    x_left, b_left, gx_left = zip(*sorted(zip(x_left, b_left, gx_left)))
    x_right, b_right, gx_right = zip(*sorted(zip(x_right, b_right, gx_right)))
    x_top, b_top, gy_top = zip(*sorted(zip(x_top, b_top, gy_top)))
    x_bottom, b_bottom, gy_bottom = zip(*sorted(zip(x_bottom, b_bottom,
                                                    gy_bottom)))

    # Make plot for values at the boundary.

    # Field.
    plt.figure()
    plt.title(r"$\hat{\phi}$", fontsize=14)
    plt.xlabel("x")
    plt.plot(x_left, b_left, label='left')
    plt.plot(x_right, b_right, label='right')
    plt.plot(x_top, b_top, label='top')
    plt.plot(x_bottom, b_bottom, label='bottom')
    plt.legend(fontsize=13, frameon=False)

    # Gradient in the x-direction.
    plt.figure()
    plt.title(r"$[\vec{\nabla}\hat{\phi}]_x$", fontsize=14)
    plt.xlabel("y")
    plt.plot(x_left, gx_left, label='left')
    plt.plot(x_right, gx_right, label='right')
    plt.legend(fontsize=13, frameon=False)

    # Gradient in the y-direction.
    plt.figure()
    plt.title(r"$[\vec{\nabla}\hat{\phi}]_y$", fontsize=14)
    plt.xlabel("x")
    plt.plot(x_top, gy_top, label='top')
    plt.plot(x_bottom, gy_bottom, label='bottom')
    plt.legend(fontsize=13, frameon=False)

    return None


if __name__ == "__main__":
    # Generate mesh.
    create_mesh()

    # Solve field.
    p = DensityProfile(filename="BC_example_Neumann", dimension=2,
                       symmetry='cylinder slice', profiles=[background])

    # Define regions on the boundary for Neumann boundary conditions.
    def left(x):
        return x[0] < -0.4999

    def right(x):
        return x[0] > +0.4999

    def top(x):
        return x[1] > +0.4999 and abs(x[0]) < 0.1

    def bottom(x):
        return x[1] < -0.4999

    p.assign_boundary_labels([left, right, top, bottom])

    # Solve field with boundary conditions.
    solver = FieldSolver(alpha=1, n=1, density_profile=p)
    solver.tol_du = 1e-12
    solver.picard(BCs=[('Neumann', 'x[0]'), ('Neumann', 'sin(x[1]*2*pi)'),
                       ('Dirichlet', '1e-8'), ('Neumann', '0.4'),
                       ('Neumann', 'pow(x[0], 2)')])
    solver.calc_field_grad_vector()

    # Plot field and gradient throughout doamin and on the boundary.
    solver.plot_results(field_scale='linear')
    plot_boundary(solver=solver)
