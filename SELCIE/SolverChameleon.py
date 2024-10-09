#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:54:00 2021

@author: Chad Briddon

Solving the chameleon screened scalar field model using the finite element
method and FEniCS.
"""
import os
import sys
import numpy as np
import dolfin as d
import matplotlib.pyplot as plt
from SELCIE.InitialiseField import InitialiseField


class FieldSolver(object):
    def __init__(self, alpha, n, density_profile, initial_field_profiles=None,
                 deg_V=1):
        '''
        Class used to calculate the chameleon scalar field profiles for given
        parameters and density field profile. The equation of motion of the
        field must be of the form, 'alpha*nabla^2 phi + phi^{-(n+1)} = p',
        where nabla, phi, and p are all dimensionless through the rescaling
        described in https://arxiv.org/abs/2110.11917.

        This class also contains tools to diagnose and enterpolate the field
        solution, such as calculating its field gradient, plotting the results,
        and locating where the force produced by this field is maximum.

        Parameters
        ----------
        alpha : float
            Spatial coupling constant of the rescaled chameleon field.
        n : int
            Integer value which defines the form of the field potential.
        density_profile : SECLIE.DensityProfiles.DensityProfile
            A class used to define the piecewise density field attributed to a
            given mesh consisting of some number of subdomains.
        initial_field_profiles : SECLIE.InitialiseField.InitialiseField
            A class used to define the piecewise field profile attributed to a
            given mesh consisting of some number of subdomains.
        deg_V : int, optional
            Degree of finite-element space. The default is 1.

        '''

        # Manatory inputs.
        self.alpha = alpha
        self.n = n
        self.p = density_profile
        self.deg_V = deg_V

        # Get information from density_profile.
        self.mesh = density_profile.mesh
        self.subdomains = density_profile.subdomains
        self.mesh_dimension = density_profile.mesh.topology().dim()
        self.mesh_symmetry = density_profile.symmetry

        if self.mesh_dimension == 3:
            self.sym_factor = d.Constant(1)

        elif self.mesh_dimension == 2:
            if (self.mesh_symmetry == 'translation symmetry' or
                    self.mesh_symmetry == 'cylinder slice'):
                self.sym_factor = d.Constant(1)

            elif self.mesh_symmetry == 'vertical axis-symmetry':
                self.sym_factor = d.Expression('abs(x[0])', degree=0)

            elif self.mesh_symmetry == 'horizontal axis-symmetry':
                self.sym_factor = d.Expression('abs(x[1])', degree=0)

            else:
                print()
                print('------------------------------------------------------')
                print()
                print('Inputted mesh symmetry not recognised.')
                print('Terminated code prematurely.')
                print()
                print('Allowed symmetrise for 2D include:')
                print('    - "translation symmetry" (or "cylinder slice")')
                print('    - "vertical axis-symmetry"')
                print('    - "horizontal axis-symmetry"')
                print()
                print('------------------------------------------------------')
                print()
                sys.exit()

        elif self.mesh_dimension == 1:
            if self.mesh_symmetry == 'translation symmetry':
                self.sym_factor = d.Constant(1)

            elif self.mesh_symmetry == 'spherical symmetry':
                self.sym_factor = d.Expression('pow(x[0], 2)', degree=0)

            elif self.mesh_symmetry == 'cylindrical symmetry':
                self.sym_factor = d.Expression('abs(x[0])', degree=0)

            else:
                print()
                print('------------------------------------------------------')
                print()
                print('Inputted mesh symmetry not recognised.')
                print('Terminated code prematurely.')
                print()
                print('Allowed symmetrise for 1D include:')
                print('    - "translation symmetry"')
                print('    - "spherical symmetry"')
                print('    - "cylindrical symmetry"')
                print()
                print('------------------------------------------------------')
                print()
                sys.exit()

            # Print warnig message.
            print()
            print('------------------------------------------------------')
            print()
            print("Note : Even though system is 1D a y-value is still")
            print("     : needed. This is due to a quirk of meshio. I ")
            print("     : aim to fix this in a future update. In the ")
            print("     : mean time when calling for example the field ")
            print("     : value at x, use field(x, 0).")
            print()
            print('------------------------------------------------------')
            print()

        # Define function space, trial function and test function.
        self.V = d.FunctionSpace(self.mesh, 'CG', self.deg_V)
        self.v = d.TestFunction(self.V)
        self.u = d.TrialFunction(self.V)

        self.V_vector = d.VectorFunctionSpace(self.mesh, 'CG', self.deg_V)
        self.v_vector = d.TestFunction(self.V_vector)
        self.u_vector = d.TrialFunction(self.V_vector)

        # Construct initial field profile.
        if initial_field_profiles is None:
            # Get maximum density and set field to minimum value.
            density_projection = d.interpolate(self.p, self.V)
            density_max = density_projection.vector().max()
            self.field_min = pow(density_max, -1/(self.n+1))

            self.field = d.interpolate(d.Constant(self.field_min), self.V)
        else:
            # Use piecewise field profile.
            self.field = d.interpolate(
                InitialiseField(self.p, initial_field_profiles), self.V)

        # Setup scalar and vector fields.
        self.field_grad_mag = None
        self.residual = None
        self.laplacian = None
        self.potential_derivative = None
        self.field_grad = None
        self.p_field = None

        # Assemble matrices.
        self.P = d.assemble(self.p*self.v*self.sym_factor*d.dx)
        self.A = d.assemble(self.u*self.v*self.sym_factor*d.dx)

        # Define general solver parameters.
        self.relaxation_parameter = 1.0
        self.tol_du = 1.0e-14
        self.tol_rel_du = 1.0e-10
        self.maxiter = 1000

        return None

    def picard(self, solver_method="cg", preconditioner="default",
               display_progress=True, BCs=None):
        '''
        Use Picard method to solve for the chameleon field throughout
        self.mesh according to the parameters, self.n, self.alpha and self.p.

        Parameters
        ----------
        solver_method : str, optional
            Method applied by FEniCS to solve the linear approximation in each
            of the iterative steps. For up-to-date list use
            dolfin.list_linear_solver_methods(). The default is "cg".
        preconditioner : str, optional
            Preconditioner applied to the linear calculation. For up-to-date
            list use dolfin.list_krylov_solver_preconditioners().
            The default is "default".
        display_progress : bool, optional
            If true then current progress of solver, including number of
            iterations and larest change in field value, will be printed.
            The default is True.
        BCs : None or list of tuples, optional
            Defines Dirichlet and Neumann boundary condition function along
            the corresponding sections of the boundary. If the argument is a
            list then the position of the element in the list determines which
            boundary section it is applied to (BCs[i] --> boundary labeled i).
            Each element should be a tuple where the first element is a string
            indicating the type of boundary condition ('Dirichlet'/'Neumann'),
            while the second gives the boundary function in the form of C++
            code as a string. If the boundary function is instead None then
            defaults are used which isno boundary condition is applied for
            Dirichlet and zero for Neumann. If BCs is None then the default is
            used for all boundaries. The default is None.

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

        A0 = d.assemble(self.alpha*d.dot(d.grad(self.u), d.grad(self.v)) *
                        self.sym_factor*d.dx)

        # Apply Dirichlet and Neumann boundary conditions.
        Dirichlet_BCs = []
        Neumann_BCs = []

        if BCs is not None:
            for i, bc in enumerate(BCs):
                if bc is not None:
                    bc_type = bc[0]
                    bc_expression = d.Expression(bc[1], degree=self.deg_V)

                    if bc_type == "Dirichlet":
                        Dirichlet_BCs.append(
                            d.DirichletBC(self.V, bc_expression,
                                          self.p.boundary, i))

                    elif bc_type == "Neumann":
                        Neumann_BCs.append((i, bc_expression))

                    else:
                        print("Unknown boundary condition type : %s" % bc_type)

        # If Neumann bc were found construct surface integration element.
        if Neumann_BCs:
            ds = d.Measure('ds', domain=self.mesh,
                           subdomain_data=self.p.boundary)

            F = self.P - self.alpha*d.assemble(
                sum([bg*self.v*self.sym_factor*ds(i)
                     for i, bg in Neumann_BCs]))

        else:
            F = self.P

        i = 0
        du_norm = 1
        while du_norm > self.tol_du and i < self.maxiter:
            i += 1

            A1 = d.assemble((self.n + 1)*pow(self.field, -self.n - 2)*self.u *
                            self.v*self.sym_factor*d.dx)
            B = d.assemble((self.n + 2)*pow(self.field, -self.n - 1)*self.v *
                           self.sym_factor*d.dx)

            A = A0 + A1
            L = B - F

            [bc.apply(A, L) for bc in Dirichlet_BCs]  # Apply Dirichlet bc.

            solver.solve(A, u.vector(), L)
            du.vector()[:] = u.vector() - self.field.vector()
            self.field.assign(self.relaxation_parameter*u +
                              (1 - self.relaxation_parameter)*self.field)

            du_norm = d.norm(du.vector(), 'linf')

            if display_progress:
                print('iter=%d: du_norm=%g' % (i, du_norm))

        if display_progress:
            print()
            print("Solver completed.")
            print()

        return None

    def newton(self, solver_method="cg", preconditioner="default",
               display_progress=True, BCs=None):
        '''
        Use Newton method to solve for the chameloen field throughout
        self.mesh according to the parameters, self.n, self.alpha and self.p.

        Parameters
        ----------
        solver_method : str, optional
            Method applied by FEniCS to solve the linear approximation in each
            of the iterative steps. For up-to-date list use
            dolfin.list_linear_solver_methods(). The default is "cg".
        preconditioner : str, optional
            Preconditioner applied to the linear calculation. For up-to-date
            list use dolfin.list_krylov_solver_preconditioners().
            The default is "default".
        display_progress : bool, optional
            If true then current progress of solver, including number of
            iterations and larest change in field value, will be printed.
            The default is True.
        BCs : None or list of tuples, optional
            Defines Dirichlet and Neumann boundary condition function along
            the corresponding sections of the boundary. If the argument is a
            list then the position of the element in the list determines which
            boundary section it is applied to (BCs[i] --> boundary labeled i).
            Each element should be a tuple where the first element is a string
            indicating the type of boundary condition ('Dirichlet'/'Neumann'),
            while the second gives the boundary function in the form of C++
            code as a string. If the boundary function is instead None then
            defaults are used which isno boundary condition is applied for
            Dirichlet and zero for Neumann. If BCs is None then the default is
            used for all boundaries. The default is None.

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
        A0 = d.assemble(d.dot(d.grad(self.u), d.grad(self.v)) *
                        self.sym_factor*d.dx)

        # Apply Dirichlet and Neumann boundary conditions.
        Dirichlet_BCs = []
        Neumann_BCs = []

        if BCs is not None:
            for i, bc in enumerate(BCs):
                if bc is not None:
                    bc_type = bc[0]
                    bc_expression = d.Expression(bc[1], degree=self.deg_V)

                    if bc_type == "Dirichlet":
                        Dirichlet_BCs.append(
                            d.DirichletBC(self.V, bc_expression,
                                          self.p.boundary, i))

                    elif bc_type == "Neumann":
                        Neumann_BCs.append((i, bc_expression))

                    else:
                        print("Unknown boundary condition type : %s" % bc_type)

        # If Neumann bc were found construct surface integration element.
        if Neumann_BCs:
            ds = d.Measure('ds', domain=self.mesh,
                           subdomain_data=self.p.boundary)

            F = self.P - self.alpha*d.assemble(
                sum([bg*self.v*self.sym_factor*ds(i)
                     for i, bg in Neumann_BCs]))

        else:
            F = self.P

        i = 0
        du_norm = 1
        while du_norm > self.tol_du and i < self.maxiter:
            i += 1

            A1 = d.assemble((self.n + 1)*pow(self.field, -self.n - 2)*self.u *
                            self.v*self.sym_factor*d.dx)
            B = d.assemble(-self.alpha*d.dot(d.grad(self.field),
                                             d.grad(self.v))*self.sym_factor *
                           d.dx + pow(self.field, -self.n - 1)*self.v *
                           self.sym_factor*d.dx)

            A = self.alpha*A0 + A1
            L = B - F

            [bc.apply(A, L) for bc in Dirichlet_BCs]  # Apply Dirichlet bc.

            solver.solve(A, du.vector(), L)
            self.field.vector()[:] += self.relaxation_parameter*du.vector()

            du_norm = d.norm(du.vector(), 'linf')

            if display_progress:
                print('iter=%d: du_norm=%g' % (i, du_norm))

        if display_progress:
            print()
            print("Solver completed.")
            print()

        return None

    def calc_field_grad_vector(self, solver_method="cg",
                               preconditioner="jacobi"):
        '''
        Calculate the field vector gradient of self.field and stores it as
        'self.field_grad'.

        Parameters
        ----------
        solver_method : str, optional
            Method applied by FEniCS to solve the linear approximation in each
            of the iterative steps. For up-to-date list use
            dolfin.list_linear_solver_methods(). The default is "cg".
        preconditioner : str, optional
            Preconditioner applied to the linear calculation. For up-to-date
            list use dolfin.list_krylov_solver_preconditioners().
            The default is "jacobi".

        Returns
        -------
        None.

        '''

        if self.field is None:
            self.picard()

        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter

        A = d.assemble(d.inner(self.u_vector, self.v_vector) *
                       self.sym_factor*d.dx)
        b = d.assemble(d.inner(d.grad(self.field), self.v_vector) *
                       self.sym_factor*d.dx)

        self.field_grad = d.Function(self.V_vector)
        solver.solve(A, self.field_grad.vector(), b)

        return None

    def calc_field_grad_mag(self, solver_method="cg", preconditioner="jacobi"):
        '''
        Calculate the magnitude of the field gradient of self.field and stores
        it as 'self.field_grad_mag'. Is equivalent to |self.field_grad_vector|.

        Parameters
        ----------
        solver_method : str, optional
            Method applied by FEniCS to solve the linear approximation in each
            of the iterative steps. For up-to-date list use
            dolfin.list_linear_solver_methods(). The default is "cg".
        preconditioner : str, optional
            Preconditioner applied to the linear calculation. For up-to-date
            list use dolfin.list_krylov_solver_preconditioners().
            The default is "jacobi".

        Returns
        -------
        None.

        '''

        if self.field is None:
            self.picard()

        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter

        b = d.assemble(d.sqrt(d.inner(d.grad(self.field),
                                      d.grad(self.field)))*self.v *
                       self.sym_factor*d.dx)

        self.field_grad_mag = d.Function(self.V)
        solver.solve(self.A, self.field_grad_mag.vector(), b)

        return None

    def calc_field_residual(self, solver_method="richardson",
                            preconditioner="icc"):
        '''
        Inputs 'self.field' into the equation of motion to get the strong
        residual of the solution and stores it as 'self.residual'.

        Parameters
        ----------
        solver_method : str, optional
            Method applied by FEniCS to solve the linear approximation in each
            of the iterative steps. For up-to-date list use
            dolfin.list_linear_solver_methods(). The default is "richardson".
        preconditioner : str, optional
            Preconditioner applied to the linear calculation. For up-to-date
            list use dolfin.list_krylov_solver_preconditioners().
            The default is "icc".

        Returns
        -------
        None.

        '''

        if self.field is None:
            self.picard()

        if self.field_grad is None:
            self.calc_field_grad_vector()

        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter

        b = d.assemble(self.alpha*d.div(self.field_grad)*self.v *
                       self.sym_factor*d.dx + pow(self.field, -self.n-1) *
                       self.v*self.sym_factor*d.dx - self.p*self.v *
                       self.sym_factor*d.dx)

        self.residual = d.Function(self.V)
        solver.solve(self.A, self.residual.vector(), b)

        return None

    def calc_laplacian(self, solver_method="richardson", preconditioner="icc"):
        '''
        Calculates the Laplacian of 'self.field' and stores it as
        'self.laplacian'.

        Parameters
        ----------
        solver_method : str, optional
            Method applied by FEniCS to solve the linear approximation in each
            of the iterative steps. For up-to-date list use
            dolfin.list_linear_solver_methods(). The default is "richardson".
        preconditioner : str, optional
            Preconditioner applied to the linear calculation. For up-to-date
            list use dolfin.list_krylov_solver_preconditioners().
            The default is "icc".

        Returns
        -------
        None.

        '''

        if self.field_grad is None:
            self.calc_field_grad_vector()

        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter

        b = d.assemble(d.div(self.field_grad)*self.v*self.sym_factor*d.dx)

        self.laplacian = d.Function(self.V)
        solver.solve(self.A, self.laplacian.vector(), b)

        return None

    def calc_density_field(self):
        '''
        Calculates a field profile that corresponds to the density profile
        contained in 'self.p' and stores it as 'self.p_field'.

        Returns
        -------
        None.

        '''

        self.p_field = d.interpolate(self.p, self.V)

        return None

    def calc_potential_derivative(self, solver_method="richardson",
                                  preconditioner="icc"):
        '''
        Calculate the derivative of the field potencial of 'self.field', which
        is equivalent to 'self.field^{-(self.n+1)}', and saves it as
        'self.potential_derivative'.

        Parameters
        ----------
        solver_method : str, optional
            Method applied by FEniCS to solve the linear approximation in each
            of the iterative steps. For up-to-date list use
            dolfin.list_linear_solver_methods(). The default is "richardson".
        preconditioner : str, optional
            Preconditioner applied to the linear calculation. For up-to-date
            list use dolfin.list_krylov_solver_preconditioners().
            The default is "icc".

        Returns
        -------
        None.

        '''

        if self.field is None:
            self.picard()

        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter

        b = d.assemble(pow(self.field, -self.n-1)*self.v*self.sym_factor*d.dx)

        self.potential_derivative = d.Function(self.V)
        solver.solve(self.A, self.potential_derivative.vector(), b)

        return None

    def measure_function(self, function, subdomain, check_boundary_only=False,
                         constraint=None):
        '''
        Locates mesh vertex where 'function' is maximised in specified
        subdomain.

        Parameters
        ----------
        function : function
            Function that takes position arguments (x, y, z) and returns float
            value.
        subdomain : int
            Index of the subdomain that will be searched.
        check_boundary_only : bool, optional
            If True then the search will be restricted to the boundary of the
            specified subdomain. If False the entire subdomain wil be searched.
            The default is False.
        constraint : function, optional
            Takes position argumentsb (x, y, z), return True for points to be
            measure and Flase for points to be ignored. The default is None.

        Returns
        -------
        function_max : float
            Maximum value of function on a vertex in sellected subdomain.
        pos : numpy.ndarray
            Position of function_max.

        '''

        function_max = 0.0
        pos = None

        # Get vertex positions.
        measuring_mesh = d.SubMesh(self.mesh, self.subdomains, subdomain)

        if check_boundary_only:
            bmesh = d.BoundaryMesh(measuring_mesh, "exterior")
            vertex_coordinates = bmesh.coordinates()
        else:
            vertex_coordinates = measuring_mesh.coordinates()

        # Check function at vertices.
        if constraint:
            for p in vertex_coordinates:
                if constraint(p):
                    if function(p) > function_max:
                        function_max = function(p)
                        pos = p
        else:

            for p in vertex_coordinates:
                if function(p) > function_max:
                    function_max = function(p)
                    pos = p

        # Check a point was found.
        if pos is None:
            raise Exception("No vertex found. Check 'constraint'.")

        return function_max, pos

    def plot_results(self, field_scale=None, grad_scale=None, res_scale=None,
                     lapl_scale=None, dpot_scale=None, density_scale=None):
        '''
        Plot calculated field properties such as the field, gradient, strong
        residual, Laplacian, potential derivative, and density profile.
        Note currently only works for 1D & 2D solutions.

        Parameters
        ----------
        field_scale : {None, 'linear', 'log'}, optional
            Plots 'self.field' using specified scale.
            Plots nothing if None. The default is None.
        grad_scale : {None, 'linear', 'log'}, optional
            Plots 'self.field_grad_mag' using specified scale.
            Plots nothing if None. The default is None.
        res_scale : {None, 'linear', 'log'}, optional
            Plots 'self.residual' using specified scale.
            Plots nothing if None. The default is None.
        lapl_scale : {None, 'linear', 'log'}, optional
            Plots 'self.laplacian' using specified scale.
            Plots nothing if None. The default is None.
        dpot_scale : {None, 'linear', 'log'}, optional
            Plots 'self.potential_derivative' using specified scale.
            Plots nothing if None. The default is None.
        density_scale : {None, 'linear', 'log'}, optional
            Plots 'self.p_field' using specified scale.
            Plots nothing if None. The default is None.

        Returns
        -------
        plot_list : list
            List containing the generated figures.

        '''

        plot_list = []

        if self.mesh_dimension == 3:
            print("FieldSover.plot_results() does not currently work in 3D.")
            return None

        elif self.mesh_dimension == 2:

            if field_scale is not None:
                if self.field is None:
                    print("Field must be solved before being plotted.")
                else:
                    fig_field = plt.figure(dpi=150)
                    plt.title("Field")
                    plt.ylabel('y')
                    plt.xlabel('x')

                    if field_scale.lower() == "linear":
                        img_field = d.plot(self.field)
                        fig_field.colorbar(img_field, label=r"$\hat{\phi}$")

                    elif field_scale.lower() == "log":
                        log_field = d.Function(self.V)
                        log_field.vector()[:] = np.log10(
                            abs(self.field.vector()[:]) + 1e-14)
                        img_field = d.plot(log_field)
                        fig_field.colorbar(img_field,
                                           label=r"$\log_{10}(\hat{\phi})$")

                    else:
                        print("")
                        print('"' + field_scale + '"',
                              "is not a valid argument for field_scale.")

                    plot_list.append(fig_field)

            if grad_scale is not None:
                if self.field_grad_mag is None:
                    print(
                        "Field gradient must be solved before being plotted.")
                else:
                    fig_grad = plt.figure(dpi=150)
                    plt.title("Magnitude of Field Gradient")
                    plt.ylabel('y')
                    plt.xlabel('x')

                    if grad_scale.lower() == "linear":
                        img_grad = d.plot(self.field_grad_mag)
                        fig_grad.colorbar(img_grad,
                                          label=r"$|\hat{\nabla} \hat{\phi}|$")

                    elif grad_scale.lower() == "log":
                        log_grad = d.Function(self.V)
                        log_grad.vector()[:] = np.log10(
                            abs(self.field_grad_mag.vector()[:]) + 1e-14)
                        img_grad = d.plot(log_grad)
                        fig_grad.colorbar(
                            img_grad,
                            label=r"$\log_{10}(|\hat{\nabla} \hat{\phi}|)$")

                    else:
                        print("")
                        print('"' + grad_scale + '"',
                              "is not a valid argument for grad_scale.")

                    plot_list.append(fig_grad)

            if res_scale is not None:
                if self.residual is None:
                    print("Residual must be solved before being plotted.")
                else:
                    fig_res = plt.figure(dpi=150)
                    plt.title("Field Residual")
                    plt.ylabel('y')
                    plt.xlabel('x')

                    if res_scale.lower() == "linear":
                        img_res = d.plot(self.residual)
                        fig_res.colorbar(img_res, label=r"$\hat{\epsilon}$")

                    elif res_scale.lower() == "log":
                        log_res = d.Function(self.V)
                        log_res.vector()[:] = np.log10(
                            abs(self.residual.vector()[:]) + 1e-14)
                        img_res = d.plot(log_res)
                        fig_res.colorbar(
                            img_res, label=r"$\log_{10}(|\hat{\epsilon}|)$")

                    else:
                        print("")
                        print('"' + res_scale + '"',
                              "is not a valid argument for res_scale.")

                    plot_list.append(fig_res)

            if lapl_scale is not None:
                if self.laplacian is None:
                    print("Laplacian must be solved before being plotted.")
                else:
                    fig_lapl = plt.figure()
                    plt.title("Laplacian of Field")
                    plt.ylabel('y')
                    plt.xlabel('x')

                    if lapl_scale.lower() == "linear":
                        img_lapl = d.plot(self.laplacian)
                        fig_lapl.colorbar(img_lapl,
                                          label=r"$\hat{\nabla}^2 \hat{\phi}$")

                    elif lapl_scale.lower() == "log":
                        log_lapl = d.Function(self.V)
                        log_lapl.vector()[:] = np.log10(
                            abs(self.laplacian.vector()[:]) + 1e-14)
                        img_lapl = d.plot(log_lapl)
                        fig_lapl.colorbar(
                            img_lapl,
                            label=r"$\log_{10}(|\hat{\nabla}^2\hat{\phi}|)$")

                    else:
                        print("")
                        print('"' + lapl_scale + '"',
                              "is not a valid argument for lapl_scale.")

                    plot_list.append(fig_lapl)

            if dpot_scale is not None:
                if self.potential_derivative is None:
                    print(
                        "Potential derivative must be solved before being"
                        + " plotted.")
                else:
                    fig_pot = plt.figure()
                    plt.title("Field Potential")
                    plt.ylabel('y')
                    plt.xlabel('x')

                    if dpot_scale.lower() == "linear":
                        img_pot = d.plot(self.potential_derivative)
                        fig_pot.colorbar(img_pot,
                                         label=r"|$\hat{V}'(\hat{\phi})|$")

                    elif dpot_scale.lower() == "log":
                        log_pot = d.Function(self.V)
                        log_pot.vector()[:] = np.log10(
                            abs(self.potential_derivative.vector()[:]) + 1e-14)
                        img_pot = d.plot(log_pot)
                        fig_pot.colorbar(
                            img_pot,
                            label=r"$\log_{10}(|\hat{V}'(\hat{\phi})|)$")

                    else:
                        print("")
                        print('"' + dpot_scale + '"',
                              "is not a valid argument for dpot_scale.")

                    plot_list.append(fig_pot)

            if density_scale is not None:
                if self.p_field is None:
                    print("Density field must be solved before being plotted.")
                else:
                    fig_density = plt.figure(dpi=150)
                    plt.title("Density Field")
                    plt.ylabel('y')
                    plt.xlabel('x')

                    if density_scale.lower() == "linear":
                        img_density = d.plot(self.p_field, extend='max')
                        fig_density.colorbar(img_density,
                                             label=r"$\hat{\rho}$")

                    elif density_scale.lower() == "log":
                        log_density = d.Function(self.V)
                        log_density.vector()[:] = np.log10(
                            self.p_field.vector()[:] + 1e-14)
                        img_density = d.plot(log_density, extend='max')
                        fig_density.colorbar(img_density,
                                             label=r'$\log_{10}(\hat{\rho})$')

                    else:
                        print("")
                        print('"' + density_scale + '"',
                              "is not a valid argument for density_scale.")

                    plot_list.append(fig_density)

        elif self.mesh_dimension == 1:
            # Get range of x-values.
            x = [xi[0] for xi in self.mesh.coordinates()]
            x.sort()

            if field_scale is not None:
                if self.field is None:
                    print("Field must be solved before being plotted.")

                else:
                    fig_field = plt.figure(dpi=150)
                    plt.title("Field")
                    plt.ylabel(r"$\hat{\phi}$")
                    plt.xlabel('x')

                    if field_scale.lower() == "linear":
                        plt.plot(x, [self.field(xi, 0) for xi in x])

                    elif field_scale.lower() == "log":
                        plt.yscale('log')
                        plt.plot(x, [abs(self.field(xi, 0)) + 1e-14
                                     for xi in x])

                    else:
                        print("")
                        print('"' + field_scale + '"',
                              "is not a valid argument for field_scale.")

                    plot_list.append(fig_field)

            if grad_scale is not None:
                if self.field_grad_mag is None:
                    print(
                        "Field gradient must be solved before being plotted.")
                else:
                    fig_grad = plt.figure(dpi=150)
                    plt.title("Magnitude of Field Gradient")
                    plt.ylabel(r"$|\hat{\nabla} \hat{\phi}|$")
                    plt.xlabel('x')

                    if grad_scale.lower() == "linear":
                        plt.plot(x, [self.field_grad_mag(xi, 0) for xi in x])

                    elif grad_scale.lower() == "log":
                        plt.yscale('log')
                        plt.plot(x, [abs(self.field_grad_mag(xi, 0)) + 1e-14
                                     for xi in x])

                    else:
                        print("")
                        print('"' + grad_scale + '"',
                              "is not a valid argument for grad_scale.")

                    plot_list.append(fig_grad)

            if res_scale is not None:
                if self.residual is None:
                    print("Residual must be solved before being plotted.")
                else:
                    fig_res = plt.figure(dpi=150)
                    plt.title("Field Residual")
                    plt.ylabel(r"$\hat{\epsilon}$")
                    plt.xlabel('x')

                    if res_scale.lower() == "linear":
                        plt.plot(x, [self.residual(xi, 0) for xi in x])

                    elif res_scale.lower() == "log":
                        plt.yscale('log')
                        plt.plot(x, [abs(self.residual(xi, 0)) + 1e-14
                                     for xi in x])

                    else:
                        print("")
                        print('"' + res_scale + '"',
                              "is not a valid argument for res_scale.")

                    plot_list.append(fig_res)

            if lapl_scale is not None:
                if self.laplacian is None:
                    print("Laplacian must be solved before being plotted.")
                else:
                    fig_lapl = plt.figure()
                    plt.title("Laplacian of Field")
                    plt.ylabel(r"$\hat{\nabla}^2 \hat{\phi}$")
                    plt.xlabel('x')

                    if lapl_scale.lower() == "linear":
                        plt.plot(x, [self.laplacian(xi, 0) for xi in x])

                    elif lapl_scale.lower() == "log":
                        plt.yscale('log')
                        plt.plot(x, [abs(self.laplacian(xi, 0)) + 1e-14
                                     for xi in x])

                    else:
                        print("")
                        print('"' + lapl_scale + '"',
                              "is not a valid argument for lapl_scale.")

                    plot_list.append(fig_lapl)

            if dpot_scale is not None:
                if self.potential_derivative is None:
                    print(
                        "Potential derivative must be solved before being"
                        + " plotted.")
                else:
                    fig_pot = plt.figure()
                    plt.title("Field Potential")
                    plt.ylabel(r"|$\hat{V}'(\hat{\phi})|$")
                    plt.xlabel('x')

                    if dpot_scale.lower() == "linear":
                        plt.plot(x, [self.potential_derivative(xi, 0)
                                     for xi in x])

                    elif dpot_scale.lower() == "log":
                        plt.yscale('log')
                        plt.plot(x, [abs(self.potential_derivative(xi, 0))
                                     + 1e-14 for xi in x])

                    else:
                        print("")
                        print('"' + dpot_scale + '"',
                              "is not a valid argument for dpot_scale.")

                    plot_list.append(fig_pot)

            if density_scale is not None:
                if self.p_field is None:
                    print("Density field must be solved before being plotted.")
                else:
                    fig_density = plt.figure(dpi=150)
                    plt.title("Density Field")
                    plt.ylabel(r"$\hat{\rho}$")
                    plt.xlabel('x')

                    if density_scale.lower() == "linear":
                        plt.plot(x, [self.p_field(xi, 0) for xi in x])

                    elif density_scale.lower() == "log":
                        plt.yscale('log')
                        plt.plot(x, [abs(self.p_field(xi, 0)) + 1e-14
                                     for xi in x])

                    else:
                        print("")
                        print('"' + density_scale + '"',
                              "is not a valid argument for density_scale.")

                    plot_list.append(fig_density)

        return plot_list

    def probe_function(self, function, gradient_vector,
                       origin=np.array([0, 0]), radial_limit=1.0):
        '''
        Evaluate the inputted 'function' by measuring its values along the
        vector path defined by (Y = 'gradient_vector'*X + 'origin'), where X
        takes intager values starting at zero.

        Parameters
        ----------
        function : dolfin.function.function.Function.
            The function to be evaluated.
        gradient_vector : numpy.array.
            Gradient of the vector along which 'function' is measured.
        origin : numpy.array, optional
            Origin of the vector along which 'function' is measured.
            The default is np.array([0, 0]).
        radial_limit : float, optional
            The maximum radial distance from the origin that 'function' is to
            be measured at. The default is 1.0.

        Returns
        -------
        numpy.array.
            An array containing the values of 'function' along the path Y.

        '''

        if len(gradient_vector) != self.mesh_dimension or \
                len(origin) != self.mesh_dimension:
            print("Vectors given have the wrong dimesion.")
            print("Mesh is %iD while 'gradient_vector'" % self.mesh_dimension,
                  "and 'origin' are dimension %i" % len(gradient_vector),
                  "and %i, respectively." % len(origin))

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

    def plot_residual_slice(self, gradient_vector, origin=np.array([0, 0]),
                            radial_limit=1.0):
        '''
        Plots the strong residual and its components along the vector path
        defined by (Y = 'gradient_vector'*X + 'origin').

        Parameters
        ----------
        gradient_vector : numpy.array.
            Gradient of the vector along which the strong residual and its
            components are measured.
        origin : numpy.array, optional
            Origin of the vector along which the strong residual and its
            components are measured. The default is np.array([0, 0]).
        radial_limit : float, optional
            The maximum radial distance from the origin that the strong
            residual and its components are to be measured at.
            The default is 1.0.

        Returns
        -------
        None.

        '''

        # Get field values for each part of the equation of motion.
        if self.residual is None:
            self.calc_field_residual()

        if self.laplacian is None:
            self.calc_laplacian()

        if self.potential_derivative is None:
            self.calc_potential_derivative()

        if self.p_field is None:
            self.calc_density_field()

        res_value = self.probe_function(self.residual, gradient_vector,
                                        origin, radial_limit)
        lapl_value = self.probe_function(self.laplacian, gradient_vector,
                                         origin, radial_limit)
        dpot_value = self.probe_function(self.potential_derivative,
                                         gradient_vector,
                                         origin, radial_limit)
        p_value = self.probe_function(self.p_field, gradient_vector,
                                      origin, radial_limit)

        # Get distence from origin of where each measurment was taken.
        ds = np.linalg.norm(gradient_vector)
        N = len(res_value)
        s = np.linspace(0, N-1, N)
        s *= ds

        plt.figure(figsize=[5.8, 4.0], dpi=150)
        plt.title("Residual Components Vs Displacement Along Given Vector")
        plt.xlim([0, max(s)])
        plt.yscale("log")
        plt.xlabel(r"$\hat{x}$")
        plt.plot(s, abs(res_value), label=r"$\left|\hat{\varepsilon}\right|$")
        plt.plot(s, self.alpha*abs(lapl_value),
                 label=r"$\alpha \left|\nabla^2 \hat{\phi}\right|$")
        plt.plot(s, abs(dpot_value), label=r"$\hat{\phi}^{-(n+1)}$")
        plt.plot(s, p_value, label=r"$\left|-\hat{\rho}\right|$")
        plt.legend()

        return None

    def save(self, filename, path=None, auto_override=False):
        '''
        Save calculated field properties as .h5 files in a directory named
        'Saved Solutions/filename'. If directory 'Saved Solutions' does not
        exist then it will be created in working directory.

        Parameters
        ----------
        filename : str
            Name of directory that contains the saved files.
        path : None or string, optional
            If saving to a different directory than the current one then
            specify it using path. The default is None.
        auto_override : bool, optional
            If True, will automatically overwrite any existing directory with
            the same name as 'filename'. If False and directory 'filename'
            already exists then the user will be asked if they want to
            overwrite the old directory.

        Returns
        -------
        None.

        '''

        if path is None:
            file_path = 'Saved Solutions/' + filename
        else:
            file_path = path + '/Saved Solutions/' + filename

        try:
            os.makedirs(file_path)
        except FileExistsError:
            if auto_override is False:
                print('Directory %s already exists.' % file_path)
                ans = input('Overwrite files inside this directory? (y/n) ')

                if ans.lower() == 'y':
                    pass
                elif ans.lower() == 'n':
                    sys.exit()
                else:
                    print('Invalid input.')
                    sys.exit()

        # Save solution
        if self.field is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/field.h5", "w") as f:
                f.write(self.field, "field")

        if self.field_grad_mag is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/field_grad_mag.h5", "w") as f:
                f.write(self.field_grad_mag, "field_grad_mag")

        if self.residual is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/residual.h5", "w") as f:
                f.write(self.residual, "residual")

        if self.laplacian is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/laplacian.h5", "w") as f:
                f.write(self.laplacian, "laplacian")

        if self.potential_derivative is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/potential_derivative.h5", "w") as f:
                f.write(self.potential_derivative, "potential_derivative")

        if self.field_grad is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/field_grad.h5", "w") as f:
                f.write(self.field_grad, "field_grad")

        if self.p_field is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/p_field.h5", "w") as f:
                f.write(self.p_field, "p_field")

        return None

    def load(self, filename, path=None):
        '''
        Loads field properties from directory named 'Saved Solutions/filename'.

        Parameters
        ----------
        filename : str
            Name of directory from which field properties will be loaded from.
        path : None or string, optional
            If saving to a different directory than the current one then
            specify it using path. The default is None.

        Returns
        -------
        None.

        '''

        if path is None:
            file_path = 'Saved Solutions/' + filename
        else:
            file_path = path + '/Saved Solutions/' + filename

        # Load solution
        if os.path.exists(file_path + "/field.h5"):
            self.field = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/field.h5", "r") as f:
                f.read(self.field, "field")

        if os.path.exists(file_path + "/field_grad_mag.h5"):
            self.field_grad_mag = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/field_grad_mag.h5", "r") as f:
                f.read(self.field_grad_mag, "field_grad_mag")

        if os.path.exists(file_path + "/residual.h5"):
            self.residual = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/residual.h5", "r") as f:
                f.read(self.residual, "residual")

        if os.path.exists(file_path + "/laplacian.h5"):
            self.laplacian = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/laplacian.h5", "r") as f:
                f.read(self.laplacian, "laplacian")

        if os.path.exists(file_path + "/potential_derivative.h5"):
            self.potential_derivative = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/potential_derivative.h5", "r") as f:
                f.read(self.potential_derivative, "potential_derivative")

        if os.path.exists(file_path + "/field_grad.h5"):
            self.field_grad = d.Function(self.V_vector)
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/field_grad.h5", "r") as f:
                f.read(self.field_grad, "field_grad")

        if os.path.exists(file_path + "/p_field.h5"):
            self.p_field = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            file_path + "/p_field.h5", "r") as f:
                f.read(self.p_field, "p_field")

        return None
