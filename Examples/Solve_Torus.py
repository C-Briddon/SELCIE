#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:47:14 2021

@author: Chad Briddon

Solve the chameleon field around a torus shaped source inside a vacuum chamber
in both 2D and 3D, and compares the results.

To generate meshes run 'CreateMesh_Torus2D.py' and 'CreateMesh_Torus3D.py'.
"""
import numpy as np
import dolfin as d
import matplotlib.pyplot as plt

from SELCIE import FieldSolver
from SELCIE import DensityProfile


# Define density profile functions.
def source_wall(x):
    return 1.0e17


def vacuum(x):
    return 1.0


# Define functions used to compare 2D and 3D solutions.
def solve_torus_2D():
    '''
    Generate 2D solution for field sourced by torus in vacuum chamber.

    Returns
    -------
    s2D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.

    '''

    # Define the density profile of the mesh using its subdomains.
    p_2D = DensityProfile(filename="Torus_in_Vacuum_2D",
                          dimension=2, symmetry='vertical axis-symmetry',
                          profiles=[source_wall, vacuum, source_wall])

    # Setup problem and solve.
    s2D = FieldSolver(alpha, n, density_profile=p_2D)
    s2D.picard()

    return s2D


def solve_torus_3D():
    '''
    Generate 3D solution for field sourced by torus in vacuum chamber.

    Returns
    -------
    s3D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.

    '''

    # Define the density profile of the mesh using its subdomains.
    p_3D = DensityProfile(filename="Torus_in_Vacuum_3D",
                          dimension=3, symmetry='',
                          profiles=[source_wall, vacuum, source_wall])

    # Setup problem and solve.
    s3D = FieldSolver(alpha, n, density_profile=p_3D)
    s3D.picard()

    return s3D


def check_axis_sym(s3D, r, y):
    '''
    Check if 3D solution is vertically axis-symmetric.

    Parameters
    ----------
    s3D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    r : float
        Radial coordinate.
    y : float
        y-coordinate.

    Returns
    -------
    m : float
        Mean of values taken from various azimuthal angles.
    s : float
        Standard deviations of values taken from various azimuthal angles.

    '''

    angle = np.linspace(0, 2*np.pi, endpoint=False)

    x = r*np.cos(angle)
    z = r*np.sin(angle)

    phi_rot = [s3D.field(x_i, y, z_i) for x_i, z_i in zip(x, z)]

    m = np.mean(phi_rot)
    s = np.std(phi_rot)

    return m, s


def compare_path(s2D, s3D, angle1, angle2):
    '''
    Plot 1D radial line to verify solutions match. Also plots relative error
    between these lines.

    Parameters
    ----------
    s2D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    s3D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    angle1 : float
        Polar angular coordinate.
    angle2 : float
        Azimuthal angular coordinate.

    Returns
    -------
    None.

    '''

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

    plt.figure(figsize=[5.8, 4.0], dpi=150)
    plt.ylabel(r'$\hat{\phi}$')
    plt.xlabel(r'$\hat{r}$')

    plt.plot(r, phi_2D, 'r-', label='2D')
    plt.plot(r, phi_3D, 'b--', label='3D')
    plt.legend()

    # plot errors.
    er = [abs(x3-x2)/x2 for x2, x3 in zip(phi_2D, phi_3D)]

    plt.figure()
    plt.ylabel(r'$\delta\hat{\phi}/\hat{\phi}$')
    plt.xlabel('x')
    plt.plot(r, er)

    return None


def plot_error_slice(s2D, s3D, angle=0.0):
    '''
    Plot 3D slice as 2D plot.

    Parameters
    ----------
    s2D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    s3D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    angle : float, optional
        Azimuthal angle of the plane. The default is 0.0.

    Returns
    -------
    None.

    '''

    field3D_slice2D = d.Function(s2D.V)
    v2d = d.vertex_to_dof_map(s2D.V)
    P = s2D.mesh.coordinates()

    dr = 0.04
    for i, p in enumerate(P):
        if np.linalg.norm(p) < 1 + dr:
            diff = s3D.field(p[0]*np.cos(angle), p[1], p[0]*np.sin(angle)) \
                - s2D.field(p)
            field3D_slice2D.vector()[v2d[i]] = abs(diff)/s2D.field(p)

    fig = plt.figure(figsize=[5.8, 4.0], dpi=150)
    plt.title(r'$\theta$ = %f' % angle)
    img = d.plot(field3D_slice2D)
    fig.colorbar(img)

    return None


def plot_max_error(s2D, s3D, N=1, zoom=False, show_field=True, log_scale=False,
                   show_mesh=False, show_subdomains=False):
    '''
    Measures and plots the largest relative error between the 2D and 3D
    solutions (assuming the system is axis-symmetric in the y-axis).

    Parameters
    ----------
    s2D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    s3D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    N : int, optional
        Number of slices of the 3D solution to be compared to 2D.
        The default is 1.
    zoom : bool, optional
        If true will zoom in on region {x between 0 and 0.1} and
        {y between -0.05 and +0.05}. The default is False.
    show_field : bool, optional
        If True will plot maximum relative between 2D and 3D solutions.
        The default is True.
    log_scale : bool, optional
        If True then when plotting the maximum relative error will do so with
        a log scale. The default is False.
    show_mesh : bool, optional
        If True will show the mesh. The default is False.
    show_subdomains : bool, optional
        If True will plot the subdomains of the system. Note will overwrite
        the plotting of the maximum relative error. The default is False.

    Returns
    -------
    None.

    '''

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

    fig = plt.figure(figsize=[5.8, 4.0], dpi=150)
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
    '''
    Plot 2D slice of strong residual of the 3D solution using the function
    space of the 2D solution.

    Parameters
    ----------
    s2D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    s3D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.
    angle : float, optional
        Azimuthal angle of the displayed plane. The default is 0.0.

    Returns
    -------
    None.

    '''

    Res_slice = d.Function(s2D.V)
    v2d = d.vertex_to_dof_map(s2D.V)
    P = s2D.mesh.coordinates()

    dr = 0.04
    for i, p in enumerate(P):
        if np.linalg.norm(p) < 1 + dr:
            Res_slice.vector()[v2d[i]] = np.log10(
                abs(s3D.residual(p[0]*np.cos(angle), p[1],
                                 p[0]*np.sin(angle))))
        else:
            Res_slice.vector()[v2d[i]] = 5

    fig = plt.figure(figsize=[5.8, 4.0], dpi=150)
    plt.title(r'$\theta$ = %f' % angle)
    img = d.plot(Res_slice)
    fig.colorbar(img)

    return None


def error(s3D):
    '''
    Compares field on y=0 plane for range of azimuthal angles and plots the
    result.

    Parameters
    ----------
    s3D : Main.SolverChameleon.FieldSolver
        FieldSolver class with solved field.

    Returns
    -------
    None.

    '''

    R = np.linspace(0, 1, 100)

    mean = []
    std = []

    for r in R:
        m, s = check_axis_sym(s3D, r, y=0)
        mean.append(m)
        std.append(s)

    plt.figure(figsize=[5.8, 4.0], dpi=150)
    plt.ylabel(r'$\hat{\phi}$')
    plt.xlabel('r')
    plt.errorbar(R, mean, std)

    return None


# Set model parameters.
n = 3
alpha = 1e12

s_2D = solve_torus_2D()
s_3D = solve_torus_3D()


# Set figure fonts.
plt.rc('axes', titlesize=10)                # fontsize of the axes title
plt.rc('axes', labelsize=14)                # fontsize of the x and y labels
plt.rc('legend', fontsize=12.3)             # legend fontsize


# Plot 2D solution.
s_2D.plot_residual_slice(np.array([0.01, 0]), radial_limit=1.05)
s_2D.plot_residual_slice(np.array([0, 0.01]), radial_limit=1.05)

s_2D.plot_results(field_scale='log', density_scale='log', grad_scale='log',
                  res_scale='log')


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
