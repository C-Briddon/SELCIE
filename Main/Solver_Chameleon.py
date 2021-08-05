#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:54:00 2021

@author: ppycb3

Environment - fenics2019

Solving screened scalar field models using finite element method and FEniCS.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units, constants
import dolfin as d
#import pickle
import sys
#import os


class Field_Solver(object):
    def __init__(self, name, alpha, n, density_profile, deg_V = 1):
        
        'Manatory inputs.'
        self.name = name
        self.alpha = alpha
        self.n = n
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
        self.wall_field_min = pow(self.wall_density,-1/(self.n+1))
        self.source_field_min = pow(self.source_density,-1/(self.n+1))
        self.vacuum_compton_wavelength = d.sqrt(self.alpha/(self.n+1))
        self.source_compton_wavelength = self.vacuum_compton_wavelength*pow(self.source_density,-0.5*(self.n+2)/(self.n+1))
        self.boundary_compton_wavelength = self.vacuum_compton_wavelength*pow(self.wall_density,-0.5*(self.n+2)/(self.n+1))
        
        
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
        
        'Experimental.'
        self.A_grad = d.assemble(d.inner(self.u_vector, self.v_vector)*d.dx)
        self.b_grad = d.assemble(d.inner(d.grad(self.field), self.v_vector)*d.dx)
        
        self.A = d.assemble(self.u*self.v*d.dx)
        
        self.field_vacuum_value = None
        
        return None
    
    
    def picard(self, solver_method = "cg", preconditioner = "default"):
        '''
        Use picard method to solve for the chameloen field throughout 
        self.mesh according to the parameters, self.n, self.alpha and self.p.
        
        Parameters
        ----------
        solver_method : TYPE str, optional
            DESCRIPTION. The default is "cg".
        preconditioner : TYPE str, optional
            DESCRIPTION. The default is "default".
            
        Returns
        -------
        None.
        
        '''
        
        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter
        
        du = d.Function(self.V)
        u = d.Function(self.V)
        
        A0 = d.assemble(d.dot(d.grad(self.u),d.grad(self.v))*self.sym_factor*d.dx)
        P = d.assemble(self.p*self.v*self.sym_factor*d.dx)
        
        i = 0
        du_norm = 1
        while du_norm > self.tol_du and i < self.maxiter:
            i += 1
            
            A1 = d.assemble((self.n + 1)*pow(self.field, -self.n - 2)*self.u*self.v*self.sym_factor*d.dx)
            B = d.assemble((self.n + 2)*pow(self.field, -self.n - 1)*self.v*self.sym_factor*d.dx)
            
            solver.solve(self.alpha*A0 + A1, u.vector(), B - P)
            du.vector()[:] = u.vector() - self.field.vector()
            #self.field.assign(u)
            self.field.assign(self.w*u + (1 - self.w)*self.field)
            
            du_norm = d.norm(du.vector(),'linf')
            print('iter=%d: du_norm=%g' % (i, du_norm))
        
        return None
    
    
    def newton(self, relaxation_parameter = 1.0, solver_method = "cg", 
               preconditioner = "jacobi"):
        '''
        Use newton method to solve for the chameloen field throughout 
        self.mesh according to the parameters, self.n, self.alpha and self.p.
        
        Parameters
        ----------
        relaxation_parameter : TYPE float, optional
            DESCRIPTION. The default is 1.0.
            The relaxation parameter controls the iterative time step. 
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
        
        
        A0 = d.assemble(d.dot(d.grad(self.u),d.grad(self.v))*self.sym_factor*d.dx)
        P = d.assemble(self.p*self.v*self.sym_factor*d.dx)
        
        du = d.Function(self.V)
        
        i = 0
        du_norm = 1
        while du_norm > self.tol_du and i < self.maxiter:
            i += 1
            
            A1 = d.assemble((self.n + 1)*pow(self.field,-self.n - 2)*self.u*self.v*self.sym_factor*d.dx)
            B = d.assemble(-self.alpha*d.dot(d.grad(self.field),d.grad(self.v))*self.sym_factor*d.dx + pow(self.field,-self.n - 1)*self.v*self.sym_factor*d.dx)
            
            solver.solve(self.alpha*A0 + A1, du.vector(), B - P)
            self.field.vector()[:] += self.w*du.vector()
            
            du_norm = d.norm(du.vector(),'linf')
            print('iter=%d: du_norm=%g' % (i, du_norm))
        
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
    
    
    def calc_laplacian(self, solver_method = "richardson", preconditioner = "icc"):
        '''
        Calculates the laplacian of the self.field.
        
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
        
        
        b = d.assemble(d.div(self.field_grad)*self.v*d.dx)
        
        solver.solve(self.A, self.laplacian.vector(), b)
        return None
    
    
    def calc_d_potential(self, solver_method = "richardson", preconditioner = "icc"):
        '''
        Calculate the derivative of the field potencial of self.field.
        
        
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
        
        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter
        
        
        b = d.assemble(pow(self.field,-self.n-1)*self.v*d.dx)
        
        solver.solve(self.A, self.d_potential.vector(), b)
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
    
    
    def plot_results(self, field_scale = None, grad_scale = None, res_scale = None,
                     lapl_scale = None, pot_scale = None):
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
            plt.ylabel('Y')
            plt.xlabel('X')
            
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
            plt.ylabel('Y')
            plt.xlabel('X')
            
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
            plt.ylabel('Y')
            plt.xlabel('X')
            
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
        
        
        if lapl_scale is not None:
            fig_lapl = plt.figure()
            plt.title("Laplacian of Field")
            plt.ylabel('Y')
            plt.xlabel('X')
            
            if lapl_scale.lower() == "linear":
                img_lapl = d.plot(self.laplacian)
                fig_lapl.colorbar(img_lapl)
                
            elif lapl_scale.lower() == "log":
                log_lapl = d.Function(self.V)
                log_lapl.vector()[:] = np.log10(abs(self.laplacian.vector()[:]) + 1e-14)
                img_lapl = d.plot(log_lapl)
                fig_lapl.colorbar(img_lapl)
            
            else:
                print("")
                print('"' + lapl_scale + '"', "is not a valid argument for lapl_scale.")
        
        
        if pot_scale is not None:
            fig_pot = plt.figure()
            plt.title("Field Potential")
            plt.ylabel('Y')
            plt.xlabel('X')
            
            if pot_scale.lower() == "linear":
                img_pot = d.plot(self.d_potential)
                fig_pot.colorbar(img_pot)
                
            elif pot_scale.lower() == "log":
                log_pot = d.Function(self.V)
                log_pot.vector()[:] = np.log10(abs(self.d_potential.vector()[:]) + 1e-14)
                img_pot = d.plot(log_pot)
                fig_pot.colorbar(img_pot)
                
            else:
                print("")
                print('"' + pot_scale + '"', "is not a valid argument for pot_scale.")
        
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
    
    
    def plot_residual_slice(self, gradient_vector, origin = np.array([0, 0]), radial_limit = 1.0):
        '''
        Plots the residual and its components along the line defined by the 
        argument vectors according to 'Y = gradient_vector*X + origin'.
        
        Parameters
        ----------
        gradient_vector : TYPE numpy array.
            DESCRIPTION.
            Gradient of the vector along which the residual and its components 
            are measured.
        origin : TYPE numpy array, optional
            DESCRIPTION. The default is np.array([0, 0]).
            Origin of the vector along which the residual and its components 
            are measured.
        radial_limit : TYPE float, optional
            The maximum radial value that each function is to be measured at. 
            The default is 1.0.
        
        Returns
        -------
        None.
        
        '''
        
        self.calc_field_residual()
        self.calc_laplacian()
        self.calc_d_potential()
        p_func = d.interpolate(self.p, self.V)
        
        res_value = self.probe_function(self.residual, gradient_vector, 
                                        origin = np.array([0, 0]), radial_limit = 1)
        lapl_value = self.probe_function(self.laplacian, gradient_vector, 
                                         origin = np.array([0, 0]), radial_limit = 1)
        dpot_value = self.probe_function(self.d_potential, gradient_vector, 
                                         origin = np.array([0, 0]), radial_limit = 1)
        p_value = self.probe_function(p_func, gradient_vector, 
                                      origin = np.array([0, 0]), radial_limit = 1)
        
        plt.figure()
        plt.title("Residual Components Against Displacement Along Given Vector")
        plt.yscale("log")
        plt.xlabel("X")
        plt.plot(abs(res_value), label = "Residual")
        plt.plot(self.alpha*abs(lapl_value), label = r"$\alpha \nabla^2 \phi$")
        plt.plot(abs(dpot_value), label = "$|V'(\phi)|$")
        plt.plot(p_value, label = r"$|-\rho|$")
        plt.legend()
        
        return None
    
    
    def calc_field_vacuum_value(self, M, Lam, p_vac, Field_NonEVUnits = None, 
                                M_NonEVUnits = None, Lam_NonEVUnits = None, 
                                p_vac_NonEVUnits = None):
        '''
        Calculates the vacuum value for the field with units eV, unless 
        specifed to the contrary with 'Field_NonEVUnits'. Then the field will 
        be defined in the units provided as 'Field_NonEVUnits'.
        
        Parameters
        ----------
        M : TYPE float
            The coupling constant of the field to matter.
        Lam : TYPE float
            The energy scale of the field potencial.
        p_vac : TYPE float
            Vacuum density.
        Field_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of the returned field. If set to None then the units are 
            taken to be eV. The default is None.
        M_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of M. If set to None then the units are taken to be eV. 
            The default is None.
        Lam_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of Lam. If set to None then the units are taken to be eV. 
            The default is None.
        p_vac_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of p_vac. If set to None then the units are taken to be eV. 
            The default is None.
        
        Returns
        -------
        field_vacuum_value : TYPE float
            Value of the vacuum value of the field in units eV unless 
            'Field_NonEVUnits' is not None. Then return value will be in units
            'Field_NonEVUnits'.
        
        '''
        
        kg_2_eV = (constants.c.value**2)/constants.e.value
        _per_m_2_eV = constants.c.value*constants.hbar.value/constants.e.value
        kg_per_m3_2_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(constants.e.value**4)
        
        
        if M_NonEVUnits:
            M *= M_NonEVUnits.to(units.kg)*kg_2_eV
        
        if Lam_NonEVUnits:
            Lam *= Lam_NonEVUnits.to(units.m**(-1))*_per_m_2_eV
        
        if p_vac_NonEVUnits:
            p_vac *= p_vac_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_2_eV4
        
        
        field_vacuum_value = Lam*pow(self.n*M*(Lam**3)/p_vac, 1/(self.n+1))
        
        if Field_NonEVUnits:
            field_vacuum_value *= (units.m**(-1)).to(Field_NonEVUnits)/_per_m_2_eV
        
        return field_vacuum_value
    
    
    def conv_fifth_force(self, M, Lam, p_vac, L, g = 9.80665, 
                         g_NonEVUnits = units.m/units.s**2, 
                         M_NonEVUnits = None, Lam_NonEVUnits = None, 
                         p_vac_NonEVUnits = None, L_NonEVUnits = None):
        '''
        Calculates the constant of proportionality between the dimensionless 
        field gradient and the fifth force induced.
        
        Parameters
        ----------
        M : TYPE float
            The coupling constant of the field to matter.
        Lam : TYPE float
            The energy scale of the field potencial.
        p_vac : TYPE float
            Vacuum density.
        L : TYPE float
            Length scale of the system.
        g : TYPE, optional
            Rescaling factor of the fifth force. E.g. fifith force will be in 
            units of g. The default is 9.80665.
        g_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of g. If set to None then the units are taken to be eV. 
            The default is units.m/units.s**2.
        M_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of M. If set to None then the units are taken to be eV. 
            The default is None.
        Lam_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of Lam. If set to None then the units are taken to be eV. 
            The default is None.
        p_vac_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of p_vac. If set to None then the units are taken to be eV. 
            The default is None.
        L_NonEVUnits : TYPE astropy.units.core.Unit, optional
            Units of L. If set to None then the units are taken to be eV. 
            The default is None.
        
        Returns
        -------
        Conv_Xi_ff : TYPE
            The constant of proportionality between the dimensionless field 
            gradient and the fifth force induced.
        
        '''
        
        kg_2_eV = (constants.c.value**2)/constants.e.value
        _per_m_2_eV = constants.c.value*constants.hbar.value/constants.e.value
        kg_per_m3_2_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(constants.e.value**4)
        m_per_s2_2_eV = constants.hbar.value/(constants.c.value*constants.e.value)
        
        
        if M_NonEVUnits:
            M *= M_NonEVUnits.to(units.kg)*kg_2_eV
        
        if Lam_NonEVUnits:
            Lam *= Lam_NonEVUnits.to(units.m**(-1))*_per_m_2_eV
        
        if p_vac_NonEVUnits:
            p_vac *= p_vac_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_2_eV4
        
        if L_NonEVUnits:
            L *= L_NonEVUnits.to(units.m)/_per_m_2_eV
        
        if g_NonEVUnits:
            g *= g_NonEVUnits.to(units.m/units.s**2)*m_per_s2_2_eV
        
        
        Conv_Xi_ff = Lam*pow(self.n*M*(Lam**3)/p_vac, 1/(self.n+1))/(M*L*g)
        
        return Conv_Xi_ff
