#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:02:00 2021

@author: ppycb3

Environment - fenics2019

Solving for the gravitational potencial using finite element method and FEniCS.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units, constants
import dolfin as d
#import pickle
import sys
#import os


class Field_Solver(object):
    def __init__(self, name, alpha, density_profile, deg_V = 1):
        
        'Manatory inputs.'
        self.name = name
        self.alpha = alpha
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
        
        
        'Define various function and vector function spaces.'
        self.deg_V = deg_V
        self.V = d.FunctionSpace(self.mesh,'CG',self.deg_V)
        self.v = d.TestFunction(self.V)
        self.u = d.TrialFunction(self.V)
        self.field = d.Function(self.V)
        self.field_grad_mag = d.Function(self.V)
        self.residual = d.Function(self.V)
        self.laplacian = d.Function(self.V)
        
        self.V_vector = d.VectorFunctionSpace(self.mesh,'CG',self.deg_V)
        self.v_vector = d.TestFunction(self.V_vector)
        self.u_vector = d.TrialFunction(self.V_vector)
        self.field_grad = d.Function(self.V_vector)
        
        
        'Define general solver parameters.'
        self.w = 1.0
        self.tol_residual = 1.0e5
        self.tol_rel_residual = 1.0e-10
        self.tol_du = 1.0E-14
        self.tol_rel_du = 1.0e-10
        self.maxiter = 1000
        
        'Experimental.'
        self.A_grad = d.assemble(d.inner(self.u_vector, self.v_vector)*d.dx)
        self.b_grad = d.assemble(d.inner(d.grad(self.field), self.v_vector)*d.dx)
        
        self.A = d.assemble(self.u*self.v*d.dx)
        
        self.field_vacuum_value = None
        
        return None
    
    
    def solve(self, solver_method = "default", preconditioner = "default"):
        '''
        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter
        
        u = d.Function(self.V)
        
        A = d.assemble(self.alpha*d.dot(d.grad(self.u),d.grad(self.v))*self.sym_factor*d.dx)
        P = d.assemble(self.p*self.v*self.sym_factor*d.dx)
        
        solver.solve(A, u.vector(), P)
        '''
        
        a = self.alpha*d.dot(d.grad(self.u),d.grad(self.v))*self.sym_factor*d.dx
        L = self.p*self.v*self.sym_factor*d.dx
        
        
        d.solve(a == L, self.field)
        
        return None
    
    
    def calc_field_grad_vector(self, solver_method = "cg", preconditioner = "jacobi"):
        '''
        Calculate the vector gradient of self.field.

        Parameters
        ----------
        solver_method : TYPE str, optional
            DESCRIPTION. The default is "cg".
        preconditioner : TYPE str, optional
            DESCRIPTION. The default is "jacobi".
            
        Returns
        -------
        None.
        
        '''
        
        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter
        
        A = d.assemble(d.inner(self.u_vector, self.v_vector)*d.dx)
        b = d.assemble(d.inner(d.grad(self.field), self.v_vector)*d.dx)
        
        solver.solve(A, self.field_grad.vector(), b)
        
        return None
    
    
    def calc_field_grad_mag(self, solver_method = "cg", preconditioner = "jacobi"):
        '''
        Calculate the magnitude of the field gradient of self.field. 
        Is equivalent to |self.field_grad_vector|.
        
        Parameters
        ----------
        solver_method : TYPE str, optional
            DESCRIPTION. The default is "cg".
        preconditioner : TYPE str, optional
            DESCRIPTION. The default is "jacobi".
            
        Returns
        -------
        None.
        
        '''
        
        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter
        
        
        b = d.assemble(d.sqrt(d.inner(d.grad(self.field), d.grad(self.field)))*self.v*d.dx)
        
        solver.solve(self.A, self.field_grad_mag.vector(), b)
        return None
    
    
    def calc_field_residual(self, solver_method = "richardson", preconditioner = "icc"):
        '''
        Calculate the residual of the solution self.field from the field's
        equation of motion.
        
        Parameters
        ----------
        solver_method : TYPE str, optional
            DESCRIPTION. The default is "richardson".
        preconditioner : TYPE str, optional
            DESCRIPTION. The default is "icc".
            
        Returns
        -------
        None.
        
        '''
        
        self.calc_field_grad_vector()
        
        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter
        
        
        b = d.assemble(self.alpha*d.div(self.field_grad)*self.v*d.dx + \
                       pow(self.field,-self.n-1)*self.v*d.dx - self.p*self.v*d.dx)
        
        solver.solve(self.A, self.residual.vector(), b)
        return None
    
    
    def measure_fifth_force(self, boundary_distance, tol):
        '''
        Measure the fifth force induced by the field by locating where 
        self.field_grad_mag has a maximum in the region 'boundary_distance'
        away from the exterior of the 'vacuum' region.
        
        Parameters
        ----------
        boundary_distance : TYPE float
            DESCRIPTION. 
            The distance away from boundaries of the measurable 
            for which you want to probe the fifth force.
        tol : TYPE float
            DESCRIPTION. 
            The tolerance of 'boundary_distance'. 
            E.g. measurements will be taken at mesh vertices in the region
            'boundary_distance' +/- 'tol'/2 away from the boundaries of the 
            measurable region.
            
        Returns
        -------
        fifth_force_max : TYPE float
            The largest value of self.field_grad_mag found in the probed region.
        probe_point : TYPE dolfin.cpp.geometry.Point
            The mesh point which corresponds to 'fifth_force_max'.
            
        '''
        
        measuring_mesh = d.SubMesh(self.mesh, self.subdomains, 2)
        bmesh = d.BoundaryMesh(measuring_mesh, "exterior")
        bbtree = d.BoundingBoxTree()
        bbtree.build(bmesh)
        
        fifth_force_max = 0.0
        
        for v in d.vertices(measuring_mesh):
            _, distance = bbtree.compute_closest_entity(v.point())
            
            if distance > boundary_distance - tol/2 and distance < boundary_distance + tol/2:
                ff = self.field_grad_mag(v.point())
                
                if ff > fifth_force_max:
                    fifth_force_max = ff
                    probe_point = v.point()
        
        return fifth_force_max, probe_point
    
    
    def plot_results(self, field_scale = None, grad_scale = None, res_scale = None):
        '''
        Plots self.field, self.field_grad_mag and or self.field_residual.
        Options for plot scales are:
            - None (default)
            - 'linear'
            - 'log'
        
        Returns
        -------
        None.
        '''
        
        if field_scale is not None:
            fig_field = plt.figure()
            plt.title("Field Profile")
            
            if field_scale.lower() == "linear":
                img_field = d.plot(self.field)
                fig_field.colorbar(img_field)
            
            elif field_scale.lower() == "log":
                log_field = d.Function(self.V)
                log_field.vector()[:] = np.log10(self.field.vector()[:])
                img_field = d.plot(log_field)
                fig_field.colorbar(img_field)
                
            else:
                print("")
                print('"' + field_scale + '"', "is not a valid argument for field_scale.")
        
        
        if grad_scale is not None:
            fig_grad = plt.figure()
            plt.title("Magnitude of Field Gradient")
            
            if grad_scale.lower() == "linear":
                img_grad = d.plot(self.field_grad_mag)
                fig_grad.colorbar(img_grad)
            
            elif grad_scale.lower() == "log":
                log_grad = d.Function(self.V)
                log_grad.vector()[:] = np.log10(abs(self.field_grad_mag.vector()[:]) + 1e-14)
                img_grad = d.plot(log_grad)
                fig_grad.colorbar(img_grad)
                
            else:
                print("")
                print('"' + grad_scale + '"', "is not a valid argument for grad_scale.")
        
        
        if res_scale is not None:
            fig_res = plt.figure()
            plt.title("Field Residual")
            
            if res_scale.lower() == "linear":
                img_res = d.plot(self.residual)
                fig_res.colorbar(img_res)
            
            elif res_scale.lower() == "log":
                log_res = d.Function(self.V)
                log_res.vector()[:] = np.log10(abs(self.residual.vector()[:]) + 1e-14)
                img_res = d.plot(log_res)
                fig_res.colorbar(img_res)
                
            else:
                print("")
                print('"' + res_scale + '"', "is not a valid argument for res_scale.")
        
        return None
    
    
    def probe_function(self, function, gradient_vector, origin = np.array([0, 0]), radial_limit = 1.0):
        '''
        Evaluate the inputted 'function' by measuring its values along the line 
        defined by the argument vectors according to 
        'Y = gradient_vector*X + origin', where X takes intager values starting 
        at zero.
        
        Parameters
        ----------
        function : TYPE dolfin Function.
            The function to be evaluated.
        gradient_vector : TYPE numpy array.
            Gradient of the vector along which 'function' is measured.
        origin : TYPE numpy array, optional
            Origin of the vector along which 'function' is measured. 
            The default is np.array([0, 0]).
        radial_limit : TYPE float, optional
            The maximum radial value that 'function' is to be measured at. 
            The default is 1.0.
        
        Returns
        -------
        TYPE numpy array.
            An array containing the values of 'function' along 
            'Y = gradient_vector*X + origin'.
        
        '''
        
        if len(gradient_vector) != self.mesh_dimension or \
            len(origin) != self.mesh_dimension:
                print("Vectors given have the wrong dimesion.")
                print("Mesh is %iD while 'gradient_vector' and 'origin' are dimension %i and %i, respectively." 
                      %(self.mesh_dimension, len(gradient_vector), len(origin)))
                
                return None
        
        radius_distence = 0
        displacement = 0
        
        values = []
        
        v = gradient_vector*displacement + origin
        
        while radius_distence < radial_limit:
            values.append(function(v))
            
            displacement += 1
            v = gradient_vector*displacement + origin
            radius_distence = np.linalg.norm(v)
        
        return np.array(values)
    
    


