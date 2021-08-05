#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:06:05 2021

@author: ppycb3

Environment - fenics2019

Solving symmetron model using finite element method and FEniCS.
(Attempted.)
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units, constants
import dolfin as d
import sys


class Field_Solver(object):
    def __init__(self, name, alpha, nu, density_profile, deg_V = 1):
        
        'Manatory inputs.'
        self.name = name
        self.alpha = alpha
        self.nu = nu
        self.p = density_profile
        
        
        'Get information from density_profile.'
        self.mesh = density_profile.mesh
        self.subdomains = density_profile.subdomain_markers
        self.wall_density = density_profile.wall_density
        self.source_density = density_profile.source_density
        self.mesh_dimension = density_profile.mesh.topology().dim()
        self.mesh_symmetry = density_profile.mesh_symmetry
        
        
        if self.mesh_dimension == 2:
            if self.mesh_symmetry == 'vertical axis-symmetry':
                self.sym_factor = d.Expression('abs(x[0])', degree = 0)
            elif self.mesh_symmetry == 'horizontal axis-symmetry':
                self.sym_factor = d.Expression('abs(x[1])', degree = 0)
            elif self.mesh_symmetry == 'cylinder slice':
                self.sym_factor = d.Constant(1)
            else:
                print('Inputted mesh symmetry not recognised.')
                print('Terminated code prematurely.')
                sys.exit()
        
        elif self.mesh_dimension == 3:
            self.sym_factor = d.Constant(1)
        
        
        'Calculate field min and corresponding Compton wavelength.'
        self.wall_field_min = d.sqrt((nu**2 - self.wall_density)/(nu**2 - 1))
        self.source_field_min = d.sqrt((nu**2 - self.source_density)/(nu**2 - 1))
        self.vacuum_compton_wavelength = d.sqrt(0.5*self.alpha/(nu**2 - 1))
        self.source_compton_wavelength = d.sqrt(0.5*self.alpha/(nu**2 - self.source_density))
        self.boundary_compton_wavelength = d.sqrt(0.5*self.alpha/(nu**2 - self.wall_density))
        
        
        'Define various function and vector function spaces.'
        self.deg_V = deg_V
        self.V = d.FunctionSpace(self.mesh,'CG',self.deg_V)
        self.v = d.TestFunction(self.V)
        self.u = d.TrialFunction(self.V)
        self.field = d.interpolate(d.Constant(min(self.wall_field_min, self.source_field_min)), self.V)
        self.field_grad_mag = d.Function(self.V)
        self.residual = d.Function(self.V)
        self.laplacian = d.Function(self.V)
        self.d_potential = d.Function(self.V)
        
        self.V_vector = d.VectorFunctionSpace(self.mesh,'CG',self.deg_V)
        self.v_vector = d.TestFunction(self.V_vector)
        self.u_vector = d.TrialFunction(self.V_vector)
        self.field_grad = d.Function(self.V_vector)
        
        
        'Define general solver parameters.'
        self.w = 1.0
        self.tol_residual = 1.0e5
        self.tol_rel_residual = 1.0e-10
        self.tol_du = 1.0e-14
        self.tol_rel_du = 1.0e-10
        self.maxiter = 1000
        
        return None
    
    
    def numerical_intergration(self, Del):
        C0 = -self.alpha*Del**2
        C1 = 1 + (Del*self.nu)**2
        C2 = -Del**2
        C3 = -(Del**2)*(self.nu**2 - 1)
        
        field_old = self.field.copy()
        du = d.Function(self.V)
        
        i = 0
        du_norm = 1
        while du_norm > self.tol_du and i < self.maxiter:
            i += 1
            
            A_k = d.assemble(d.dot(d.grad(self.field),d.grad(self.v))*self.sym_factor*d.dx)
            P_k = d.assemble(self.p*self.field*self.v*self.sym_factor*d.dx)
            K_k = d.assemble(pow(self.field, 3)*self.v*self.sym_factor*d.dx)
            
            du.vector()[:] = C0*A_k + C1*self.field.vector() + C2*P_k \
                + C3*K_k - field_old.vector()
            
            self.field.vector()[:] += du.vector()
            
            du_norm = d.norm(du.vector(),'linf')
            print('iter=%d: du_norm=%g' % (i, du_norm))
        
        return None




