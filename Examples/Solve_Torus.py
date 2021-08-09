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

import sys
sys.path.append("..")
from Main.Meshing_Tools import Meshing_Tools
from Main.Solver_Chameleon import Field_Solver
from Main.Density_Profiles import vacuum_chamber_density_profile


def solve_torus_2D(n, alpha, ps):
    # Import mesh and convert from .msh to .xdmf.
    MT_2D = Meshing_Tools()
    filename = "../Saved Meshes/Torus_in_Vacuum_2D"
    mesh, subdomains, boundaries = MT_2D.msh_2_xdmf(filename, dim=2)
    
    
    # Define the density profile of the mesh using its subdomains.
    p = vacuum_chamber_density_profile(mesh = mesh, 
                                       subdomain_markers = subdomains, 
                                       source_density = ps, 
                                       vacuum_density = 1, 
                                       wall_density = ps, 
                                       mesh_symmetry = 'vertical axis-symmetry', 
                                       degree = 0)
    
    
    # Setup problem and solve.
    s = Field_Solver("Torus_2D", alpha = alpha, n = n, density_profile = p)
    s.picard()
    s.calc_field_grad_mag()
    s.calc_field_residual()
    
    
    # Plot results.
    plt.figure()
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.ylim([-1.1,1.1])
    plt.xlim([-1.1,1.1])
    d.plot(subdomains)
    
    s.plot_results(field_scale = 'log', grad_scale = 'log', res_scale = 'log')
    return s


def solve_torus_3D(n, alpha, ps):
    # Import mesh and convert from .msh to .xdmf.
    MT_3D = Meshing_Tools()
    filename = "../Saved Meshes/Torus_in_Vacuum_3D"
    mesh, subdomains, boundaries = MT_3D.msh_2_xdmf(filename, dim=3)
    
    
    # Define the density profile of the mesh using its subdomains.
    p = vacuum_chamber_density_profile(mesh = mesh, 
                                       subdomain_markers = subdomains, 
                                       source_density = ps, 
                                       vacuum_density = 1, 
                                       wall_density = ps, 
                                       mesh_symmetry = '', 
                                       degree = 0)
    
    
    # Setup problem and solve.
    s = Field_Solver("Torus_3D", alpha = alpha, n = n, density_profile = p)
    #s.picard('gmres', 'jacobi')
    t0 = default_timer()
    s.picard('cg', 'jacobi')
    #s.newton('bicgstab', 'ilu')
    t1 = default_timer()
    print(t1-t0)
    
    '''
    picard_M:
        'bicgstab', 'default' : 12.83       485.4965758137405
        'bicgstab', 'icc' :     12.96       475.0414276942611
        'bicgstab', 'ilu' :     12.84       478.09547620266676
        'cg', 'default' :       12.42       476.4821430183947
        'cg', 'icc' :           12.49       465.13315154239535
        'cg', 'ilu' :           12.43       465.27152643352747
        'cg', 'jacobi' :        12.60       464.6584523767233
        'cg', 'sor' :           13.03       
        'default', 'default' :  13.04       
        'default', 'icc' :      13.17       
        'default', 'ilu' :      13.03       
        'gmres', 'default' :    13.05       
        'gmres', 'icc' :        13.15       
        'gmres', 'ilu' :        13.03       
        'minres', 'default' :   12.73       
        'minres', 'icc' :       12.87       
        'minres', 'ilu' :       12.73       
        'tfqmr', 'default' :    12.87       
        'tfqmr', 'icc' :        13.00       
        'tfqmr', 'ilu' :        12.91       
        'tfqmr', 'jacobi' :     13.42       
        'tfqmr', 'sor' :        13.34       
    
    Newton_M:
        'bicgstab', 'icc' :     56.89       x
        'bicgstab', 'ilu' :     56.85       x
        'bicgstab', 'sor' :     57.35       
        'cg', 'jacobi' :        56.42       
        'default', 'sor' :      57.28       
        'gmres', 'jacobi' :     57.95       
        'gmres', 'sor' :        57.27       
        'minres', 'default' :   56.77       
        'minres', 'icc' :       56.87       
        'minres', 'ilu' :       56.75       
        'minres', 'jacobi' :    57.44       
        'minres', 'sor' :       57.23       
        'tfqmr', 'default' :    56.75       
        'tfqmr', 'icc' :        56.88       
        'tfqmr', 'ilu' :        56.79       
        'tfqmr', 'jacobi' :     57.50       
        'tfqmr', 'sor' :        57.09       
    '''
    
    return s


def check_axis_sym(s3D, r, y):
    # Check if 3D solution is vertically axis symmetric.
    angle = np.linspace(0, 2*np.pi, endpoint = False)
    
    x = r*np.cos(angle)
    z = r*np.sin(angle)
    
    phi_rot = [s3D.field(x_i, y, z_i) for x_i, z_i in zip(x, z)]
    
    phi_rot /= phi_rot[0]
    
    print(np.mean(phi_rot))
    print(np.std(phi_rot))
    
    return None


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
    phi_3D = [s3D.field(x_i, y_i, z_i) for x_i, y_i, z_i in zip(x_3D, y_3D, z_3D)]
    
    
    plt.figure()
    plt.ylabel('phi')
    plt.xlabel('r')
    
    plt.plot(r, phi_2D, 'r-', label = '2D')
    plt.plot(r, phi_3D, 'b--', label = '3D')
    plt.legend()
    
    return None


def plot_error_slice(s2D, s3D, angle = 0.0):
    # Plot 3D slice as 2D plot.
    field3D_slice2D = d.Function(s2D.V)
    v2d = d.vertex_to_dof_map(s2D.V)
    P = s_2D.mesh.coordinates()
    
    dr = 0.04
    for i, p in enumerate(P):
        if np.linalg.norm(p) < 1 + dr:
            diff = s3D.field(p[0]*np.cos(angle), p[1], p[0]*np.sin(angle)) - s2D.field(p)
            field3D_slice2D.vector()[v2d[i]] = abs(diff)#/s2D.field(p)
    
    fig = plt.figure()
    plt.title(r'$\theta$ = %f' %angle)
    img = d.plot(field3D_slice2D)
    fig.colorbar(img)
    
    return None


def plot_max_error(s2D, s3D, N = 1):
    # Plot maximum error for all slices.
    field3D_slice2D = d.Function(s2D.V)
    v2d = d.vertex_to_dof_map(s2D.V)
    P = s_2D.mesh.coordinates()
    
    dr = 0.04
    for i, p in enumerate(P):
        diff = np.zeros(N)
        if np.linalg.norm(p) < 1 + dr:
            
            for j, angle in enumerate(np.linspace(0, 2*np.pi, N, endpoint = False)):
                diff[j] = s3D.field(p[0]*np.cos(angle), p[1], p[0]*np.sin(angle)) - s2D.field(p)
            
            field3D_slice2D.vector()[v2d[i]] = max(abs(diff))#/s2D.field(p)
    
    fig = plt.figure()
    img = d.plot(field3D_slice2D)
    fig.colorbar(img)
    
    return None


def plot_Res_slice(s2D, s3D, angle = 0.0):
    # Plot 3D slice as 2D plot.
    Res_slice = d.Function(s_2D.V)
    v2d = d.vertex_to_dof_map(s_2D.V)
    P = s_2D.mesh.coordinates()
    
    dr = 0.04
    for i, p in enumerate(P):
        if np.linalg.norm(p) < 1 + dr:
            Res_slice.vector()[v2d[i]] = np.log10(abs(s3D.residual(p[0]*np.cos(angle), 
                                                                   p[1], 
                                                                   p[0]*np.sin(angle))))
        else:
            Res_slice.vector()[v2d[i]] = 5
    
    fig = plt.figure()
    plt.title(r'$\theta$ = %f' %angle)
    img = d.plot(Res_slice)
    fig.colorbar(img)
    
    return None


# Set model parameters.
n = 3
alpha = 1e18
ps = 1e17

s_2D = solve_torus_2D(n, alpha, ps)
s_3D = solve_torus_3D(n, alpha, ps)


check_axis_sym(s_3D, r = 0.1, y = 0.4)
compare_path(s_2D, s_3D, angle1 = np.pi/2, angle2 = 0.5)
plot_error_slice(s_2D, s_3D, angle = 0.1)
plot_max_error(s_2D, s_3D, N = 4)



#s_3D.calc_field_residual()
#plot_Res_slice(s_2D, s_3D, angle = 0.1)
