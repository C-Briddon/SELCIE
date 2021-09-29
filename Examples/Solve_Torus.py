#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:47:14 2021

@author: ppycb3

Environment - fenics2019

Solve the chameleon field around a torus shaped source inside a vacuum chamber.
"""
import numpy as np
import dolfin as d
import matplotlib.pyplot as plt
from timeit import default_timer
from Density_profiles import source_wall, vacuum

import sys
sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools
from Main.Solver_Chameleon import Field_Solver
from Main.Density_Profiles import Density_Profile


def solve_torus_2D():
    # Import mesh and convert from .msh to .xdmf.
    MT_2D = Meshing_Tools(Dimension=2)
    filename = "../Saved Meshes/Torus_in_Vacuum_2D"
    mesh, subdomains, boundaries = MT_2D.msh_2_xdmf(filename)

    # Define the density profile of the mesh using its subdomains.
    p_2D = Density_Profile(mesh=mesh, subdomain_markers=subdomains,
                           mesh_symmetry='vertical axis-symmetry',
                           profiles=[source_wall, vacuum, source_wall],
                           degree=0)

    # Setup problem and solve.
    s2D = Field_Solver(alpha, n, density_profile=p_2D)
    s2D.picard()

    return s2D


def solve_torus_3D():
    # Import mesh and convert from .msh to .xdmf.
    MT_3D = Meshing_Tools(Dimension=3)
    filename = "../Saved Meshes/Torus_in_Vacuum_3D"
    mesh, subdomains, boundaries = MT_3D.msh_2_xdmf(filename)

    # Define the density profile of the mesh using its subdomains.
    p_3D = Density_Profile(mesh=mesh, subdomain_markers=subdomains,
                           mesh_symmetry='',
                           profiles=[source_wall, vacuum, source_wall],
                           degree=0)

    # Setup problem and solve.
    s3D = Field_Solver(alpha, n, density_profile=p_3D)
    # s3D.picard()
    s3D.picard('cg', 'jacobi')

    return s3D


def check_axis_sym(s3D, r, y):
    # Check if 3D solution is vertically axis symmetric.
    angle = np.linspace(0, 2*np.pi, endpoint=False)

    x = r*np.cos(angle)
    z = r*np.sin(angle)

    phi_rot = [s3D.field(x_i, y, z_i) for x_i, z_i in zip(x, z)]

    m = np.mean(phi_rot)
    s = np.std(phi_rot)

    return m, s


def compare_path(s2D, s3D, angle1, angle2):
    # Plot 1D radial line to verify solutions match.
    dr = 0.01
    r = np.linspace(0, 1+dr, 100)

    x_2D = r*np.sin(angle2)
    y_2D = r*np.cos(angle2)

    x_3D = r*np.cos(angle1)*np.sin(angle2)
    y_3D = r*np.cos(angle2)
    z_3D = r*np.sin(angle1)*np.sin(angle2)

    phi_2D = [s2D.field(x_i, y_i) for x_i, y_i in zip(x_2D, y_2D)]
    phi_3D = [s3D.field(x_i, y_i, z_i) for x_i, y_i, z_i in zip(x_3D,
                                                                y_3D, z_3D)]

    plt.figure()
    plt.ylabel('phi')
    plt.xlabel('r')

    plt.plot(r, phi_2D, 'r-', label='2D')
    plt.plot(r, phi_3D, 'b--', label='3D')
    plt.legend()

    return None


def plot_error_slice(s2D, s3D, angle=0.0):
    # Plot 3D slice as 2D plot.
    field3D_slice2D = d.Function(s2D.V)
    v2d = d.vertex_to_dof_map(s2D.V)
    P = s_2D.mesh.coordinates()

    dr = 0.04
    for i, p in enumerate(P):
        if np.linalg.norm(p) < 1 + dr:
            diff = s3D.field(p[0]*np.cos(angle), p[1], p[0]*np.sin(angle)) \
                - s2D.field(p)
            field3D_slice2D.vector()[v2d[i]] = abs(diff)/s2D.field(p)

    fig = plt.figure()
    plt.title(r'$\theta$ = %f' % angle)
    img = d.plot(field3D_slice2D)
    fig.colorbar(img)

    return None


def plot_max_error(s2D, s3D, N=1, zoom=False, show_field=True, log_scale=False,
                   show_mesh=False, show_subdomains=False):
    # Zoom point and size.
    zoom_p = [0.05, 0]
    dx = 0.05

    # Plot maximum error for all slices.
    field3D_slice2D = d.Function(s2D.V)
    v2d = d.vertex_to_dof_map(s2D.V)
    P = s2D.mesh.coordinates()

    dr = 0.04
    for i, p in enumerate(P):
        diff = np.zeros(N)
        if np.linalg.norm(p) < 1 + dr:

            for j, angle in enumerate(np.linspace(0, 2*np.pi, N,
                                                  endpoint=False)):
                diff[j] = s3D.field(p[0]*np.cos(angle), p[1],
                                    p[0]*np.sin(angle)) - s2D.field(p)

            if log_scale:
                field3D_slice2D.vector()[v2d[i]] = np.log10(
                    max(abs(diff))/s2D.field(p))
            else:
                field3D_slice2D.vector()[v2d[i]] = max(abs(diff))/s2D.field(p)

    fig = plt.figure()
    plt.ylabel('y')
    plt.xlabel('x')

    if zoom:
        plt.ylim([zoom_p[1]-dx, zoom_p[1]+dx])
        plt.xlim([zoom_p[0]-dx, zoom_p[0]+dx])

    if show_field:
        img = d.plot(field3D_slice2D)
        fig.colorbar(img)

    if show_mesh:
        d.plot(s2D.mesh)

    if show_subdomains:
        d.plot(s2D.subdomains)

    return None


def plot_Res_slice(s2D, s3D, angle=0.0):
    # Plot 3D slice as 2D plot.
    Res_slice = d.Function(s_2D.V)
    v2d = d.vertex_to_dof_map(s_2D.V)
    P = s_2D.mesh.coordinates()

    dr = 0.04
    for i, p in enumerate(P):
        if np.linalg.norm(p) < 1 + dr:
            Res_slice.vector()[v2d[i]] = np.log10(
                abs(s3D.residual(p[0]*np.cos(angle), p[1],
                                 p[0]*np.sin(angle))))
        else:
            Res_slice.vector()[v2d[i]] = 5

    fig = plt.figure()
    plt.title(r'$\theta$ = %f' % angle)
    img = d.plot(Res_slice)
    fig.colorbar(img)

    return None


def plot_line(s2D, s3D):
    Y2D_x = []
    Y3D_x = []

    Y2D_y = []
    Y3D_y = []

    dr = 0.04
    X = np.linspace(0, 1 + dr, 1000)

    for x in X:
        Y2D_x.append(s2D.field(x, 0))
        Y3D_x.append(s3D.field(x, 0, 0))

        Y2D_y.append(s2D.field(0, x))
        Y3D_y.append(s3D.field(0, x, 0))

    plt.figure()
    plt.ylabel(r'$\hat{\phi}$')
    plt.xlabel('x')
    plt.plot(X, Y2D_x, 'b-', label='2D')
    plt.plot(X, Y3D_x, 'r--', label='3D')
    plt.legend()

    plt.figure()
    plt.ylabel(r'$\hat{\phi}$')
    plt.xlabel('y')
    plt.plot(X, Y2D_y, 'b-', label='2D')
    plt.plot(X, Y3D_y, 'r--', label='3D')
    plt.legend()

    # plot errors.
    er_x = [abs(x3-x2)/x2 for x2, x3 in zip(Y2D_x, Y3D_x)]
    er_y = [abs(y3-y2)/y2 for y2, y3 in zip(Y2D_y, Y3D_y)]

    plt.figure()
    plt.ylabel(r'$\delta\hat{\phi}/\hat{\phi}$')
    plt.xlabel('x')
    # plt.ylim([0,0.05])
    plt.plot(X, er_x)

    plt.figure()
    plt.ylabel(r'$\delta\hat{\phi}/\hat{\phi}$')
    plt.xlabel('y')
    # plt.ylim([0,0.05])
    plt.plot(X, er_y)

    return None


def error(s3D):
    # dr = 0.04
    R = np.linspace(0.2, 0.3, 100)

    mean = []
    std = []

    for r in R:
        m, s = check_axis_sym(s3D, r, y=0)
        mean.append(m)
        std.append(s)

    plt.figure()
    plt.ylabel(r'$\hat{\phi}$')
    plt.xlabel('x')
    plt.errorbar(R, mean, std)

    return None


# Set model parameters.
n = 3
alpha = 1e12

s_2D = solve_torus_2D()
s_3D = solve_torus_3D()


# Check 3D solution is axis-symmetric.
error(s_3D)


# Compare 2D and 3D solutions to check consistency.
compare_path(s_2D, s_3D, angle1=np.pi/2, angle2=0.5)
plot_error_slice(s_2D, s_3D, angle=0.1)


Ns = 6
plot_max_error(s_2D, s_3D, N=Ns, zoom=False, log_scale=False, show_mesh=False)
plot_max_error(s_2D, s_3D, N=Ns, zoom=False, log_scale=True, show_mesh=False)

plot_max_error(s_2D, s_3D, N=Ns, zoom=True, log_scale=False, show_mesh=False)
plot_max_error(s_2D, s_3D, N=Ns, zoom=True, log_scale=True, show_mesh=False)

plot_max_error(s_2D, s_3D, N=Ns, zoom=False, log_scale=False, show_mesh=True)
plot_max_error(s_2D, s_3D, N=Ns, zoom=True, log_scale=False, show_mesh=True)
plot_max_error(s_2D, s_3D, N=Ns, zoom=True, log_scale=True, show_mesh=True)

plot_max_error(s_2D, s_3D, N=Ns, zoom=True, show_mesh=True,
               show_subdomains=True, show_field=False)


plot_line(s_2D, s_3D)
