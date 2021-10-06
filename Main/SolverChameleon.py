#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:54:00 2021

@author: ppycb3

Solving screened scalar field models using finite element method and FEniCS.
"""
import os
import sys
import numpy as np
import dolfin as d
import matplotlib.pyplot as plt


class FieldSolver(object):
    def __init__(self, alpha, n, density_profile, deg_V=1):
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

        if self.mesh_dimension == 1:
            if self.mesh_symmetry == 'spherical':
                self.sym_factor = d.Expression('abs(x)', degree=0)
            elif self.mesh_symmetry == 'cylindrical':
                self.sym_factor = d.Expression('pow(x, 2)', degree=0)
            else:
                print('Inputted mesh symmetry not recognised.')
                print('Terminated code prematurely.')
                sys.exit()

        if self.mesh_dimension == 2:
            if self.mesh_symmetry == 'vertical axis-symmetry':
                self.sym_factor = d.Expression('abs(x[0])', degree=0)
            elif self.mesh_symmetry == 'horizontal axis-symmetry':
                self.sym_factor = d.Expression('abs(x[1])', degree=0)
            elif self.mesh_symmetry == 'cylinder slice':
                self.sym_factor = d.Constant(1)
            else:
                print('Inputted mesh symmetry not recognised.')
                print('Terminated code prematurely.')
                sys.exit()

        elif self.mesh_dimension == 3:
            self.sym_factor = d.Constant(1)

        # Define function space, trial function and test function.
        self.V = d.FunctionSpace(self.mesh, 'CG', self.deg_V)
        self.v = d.TestFunction(self.V)
        self.u = d.TrialFunction(self.V)

        self.V_vector = d.VectorFunctionSpace(self.mesh, 'CG', self.deg_V)
        self.v_vector = d.TestFunction(self.V_vector)
        self.u_vector = d.TrialFunction(self.V_vector)

        # Get maximum density and minimum field values.
        self.density_projection = d.interpolate(self.p, self.V)
        self.density_max = self.density_projection.vector().max()
        self.field_min = pow(self.density_max, -1/(self.n+1))

        # Setup scalar and vector fields.
        self.field = None
        self.field_grad_mag = None
        self.residual = None
        self.laplacian = None
        self.d_potential = None
        self.field_grad = None

        # Assemble matrices.
        self.P = d.assemble(self.p*self.v*self.sym_factor*d.dx)
        self.A = d.assemble(self.u*self.v*self.sym_factor*d.dx)

        # Define general solver parameters.
        self.w = 1.0
        self.tol_residual = 1.0e5
        self.tol_rel_residual = 1.0e-10
        self.tol_du = 1.0e-14
        self.tol_rel_du = 1.0e-10
        self.maxiter = 1000

        return None

    def picard(self, solver_method="cg", preconditioner="default"):
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

        A0 = d.assemble(d.dot(d.grad(self.u), d.grad(self.v)) *
                        self.sym_factor*d.dx)

        self.field = d.interpolate(d.Constant(self.field_min), self.V)

        i = 0
        du_norm = 1
        while du_norm > self.tol_du and i < self.maxiter:
            i += 1

            A1 = d.assemble((self.n + 1)*pow(self.field, -self.n - 2)*self.u *
                            self.v*self.sym_factor*d.dx)
            B = d.assemble((self.n + 2)*pow(self.field, -self.n - 1)*self.v *
                           self.sym_factor*d.dx)

            solver.solve(self.alpha*A0 + A1, u.vector(), B - self.P)
            du.vector()[:] = u.vector() - self.field.vector()
            self.field.assign(self.w*u + (1 - self.w)*self.field)

            du_norm = d.norm(du.vector(), 'linf')
            print('iter=%d: du_norm=%g' % (i, du_norm))

        return None

    def newton(self, relaxation_parameter=1.0, solver_method="cg",
               preconditioner="jacobi"):
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

        A0 = d.assemble(d.dot(d.grad(self.u), d.grad(self.v)) *
                        self.sym_factor*d.dx)

        du = d.Function(self.V)

        self.field = d.interpolate(d.Constant(self.field_min), self.V)

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

            solver.solve(self.alpha*A0 + A1, du.vector(), B - self.P)
            self.field.vector()[:] += self.w*du.vector()

            du_norm = d.norm(du.vector(), 'linf')
            print('iter=%d: du_norm=%g' % (i, du_norm))

        return None

    def calc_field_grad_vector(self, solver_method="cg",
                               preconditioner="jacobi"):
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

    def calc_d_potential(self, solver_method="richardson",
                         preconditioner="icc"):
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
        if self.field is None:
            self.picard()

        solver = d.KrylovSolver(solver_method, preconditioner)
        prm = solver.parameters
        prm['absolute_tolerance'] = self.tol_du
        prm['relative_tolerance'] = self.tol_rel_du
        prm['maximum_iterations'] = self.maxiter

        b = d.assemble(pow(self.field, -self.n-1)*self.v*self.sym_factor*d.dx)

        self.d_potential = d.Function(self.V)
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
            The largest value of self.field_grad_mag found in the
            probed region.
        probe_point : TYPE dolfin.cpp.geometry.Point
            The mesh point which corresponds to 'fifth_force_max'.

        '''
        if self.field_grad_mag is None:
            self.calc_field_grad_mag()

        measuring_mesh = d.SubMesh(self.mesh, self.subdomains, 2)
        bmesh = d.BoundaryMesh(measuring_mesh, "exterior")
        bbtree = d.BoundingBoxTree()
        bbtree.build(bmesh)

        fifth_force_max = 0.0

        for v in d.vertices(measuring_mesh):
            _, distance = bbtree.compute_closest_entity(v.point())

            if distance > boundary_distance - tol/2 and \
                    distance < boundary_distance + tol/2:
                ff = self.field_grad_mag(v.point())

                if ff > fifth_force_max:
                    fifth_force_max = ff
                    probe_point = v.point()

        return fifth_force_max, probe_point

    def plot_results(self, field_scale=None, grad_scale=None, res_scale=None,
                     lapl_scale=None, pot_scale=None):
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
            if self.field is None:
                self.picard()

            fig_field = plt.figure()
            plt.title("Field Profile")
            plt.ylabel('y')
            plt.xlabel('x')

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
                print('"' + field_scale + '"',
                      "is not a valid argument for field_scale.")

        if grad_scale is not None:
            if self.field_grad_mag is None:
                self.calc_field_grad_mag()

            fig_grad = plt.figure()
            plt.title("Magnitude of Field Gradient")
            plt.ylabel('y')
            plt.xlabel('x')

            if grad_scale.lower() == "linear":
                img_grad = d.plot(self.field_grad_mag)
                fig_grad.colorbar(img_grad)

            elif grad_scale.lower() == "log":
                log_grad = d.Function(self.V)
                log_grad.vector()[:] = np.log10(
                    abs(self.field_grad_mag.vector()[:]) + 1e-14)
                img_grad = d.plot(log_grad)
                fig_grad.colorbar(img_grad)

            else:
                print("")
                print('"' + grad_scale + '"',
                      "is not a valid argument for grad_scale.")

        if res_scale is not None:
            if self.residual is None:
                self.calc_field_residual()

            fig_res = plt.figure()
            plt.title("Field Residual")
            plt.ylabel('y')
            plt.xlabel('x')

            if res_scale.lower() == "linear":
                img_res = d.plot(self.residual)
                fig_res.colorbar(img_res)

            elif res_scale.lower() == "log":
                log_res = d.Function(self.V)
                log_res.vector()[:] = np.log10(
                    abs(self.residual.vector()[:]) + 1e-14)
                img_res = d.plot(log_res)
                fig_res.colorbar(img_res)

            else:
                print("")
                print('"' + res_scale + '"',
                      "is not a valid argument for res_scale.")

        if lapl_scale is not None:
            if self.laplacian is None:
                self.calc_laplacian()

            fig_lapl = plt.figure()
            plt.title("Laplacian of Field")
            plt.ylabel('y')
            plt.xlabel('x')

            if lapl_scale.lower() == "linear":
                img_lapl = d.plot(self.laplacian)
                fig_lapl.colorbar(img_lapl)

            elif lapl_scale.lower() == "log":
                log_lapl = d.Function(self.V)
                log_lapl.vector()[:] = np.log10(
                    abs(self.laplacian.vector()[:]) + 1e-14)
                img_lapl = d.plot(log_lapl)
                fig_lapl.colorbar(img_lapl)

            else:
                print("")
                print('"' + lapl_scale + '"',
                      "is not a valid argument for lapl_scale.")

        if pot_scale is not None:
            if self.d_potential is None:
                self.calc_d_potential()

            fig_pot = plt.figure()
            plt.title("Field Potential")
            plt.ylabel('y')
            plt.xlabel('x')

            if pot_scale.lower() == "linear":
                img_pot = d.plot(self.d_potential)
                fig_pot.colorbar(img_pot)

            elif pot_scale.lower() == "log":
                log_pot = d.Function(self.V)
                log_pot.vector()[:] = np.log10(
                    abs(self.d_potential.vector()[:]) + 1e-14)
                img_pot = d.plot(log_pot)
                fig_pot.colorbar(img_pot)

            else:
                print("")
                print('"' + pot_scale + '"',
                      "is not a valid argument for pot_scale.")

        return None

    def probe_function(self, function, gradient_vector,
                       origin=np.array([0, 0]), radial_limit=1.0):
        '''
        Evaluate the inputted 'function' by measuring its values along
        the line defined by the argument vectors according to
        'Y = gradient_vector*X + origin', where X takes intager
        values starting at zero.

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
        # Get field values for each part of the equation of motion.
        if self.residual is None:
            self.calc_field_residual()

        if self.laplacian is None:
            self.calc_laplacian()

        if self.d_potential is None:
            self.calc_d_potential()

        p_func = d.interpolate(self.p, self.V)

        res_value = self.probe_function(self.residual, gradient_vector,
                                        origin, radial_limit)
        lapl_value = self.probe_function(self.laplacian, gradient_vector,
                                         origin, radial_limit)
        dpot_value = self.probe_function(self.d_potential, gradient_vector,
                                         origin, radial_limit)
        p_value = self.probe_function(p_func, gradient_vector,
                                      origin, radial_limit)

        # Get distence from origin of where each measurment was taken.
        ds = np.linalg.norm(gradient_vector)
        N = len(res_value)
        s = np.linspace(0, N-1, N)
        s *= ds

        plt.figure()
        plt.title("Residual Components Vs Displacement Along Given Vector")
        plt.xlim([0, max(s)])
        plt.yscale("log")
        plt.xlabel("x")
        plt.plot(s, abs(res_value), label=r"$\left|\varepsilon\right|$")
        plt.plot(s, self.alpha*abs(lapl_value),
                 label=r"$\alpha \left|\nabla^2 \hat{\phi}\right|$")
        plt.plot(s, abs(dpot_value), label=r"$\hat{\phi}^{-(n+1)}$")
        plt.plot(s, p_value, label=r"$\left|-\hat{\rho}\right|$")
        plt.legend()

        return None

    def save(self, filename, auto_override=False):
        path = "Saved Solutions/" + filename
        try:
            os.makedirs(path)
        except FileExistsError:
            if auto_override is False:
                print('Directory %s already exists.' % path)
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
                            path + "/field.h5", "w") as f:
                f.write(self.field, "field")

        if self.field_grad_mag is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/field_grad_mag.h5", "w") as f:
                f.write(self.field_grad_mag, "field_grad_mag")

        if self.residual is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/residual.h5", "w") as f:
                f.write(self.residual, "residual")

        if self.laplacian is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/laplacian.h5", "w") as f:
                f.write(self.laplacian, "laplacian")

        if self.d_potential is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/d_potential.h5", "w") as f:
                f.write(self.d_potential, "d_potential")

        if self.field_grad is not None:
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/field_grad.h5", "w") as f:
                f.write(self.field_grad, "field_grad")

        return None

    def load(self, filename):
        path = "Saved Solutions/" + filename

        # Load solution
        if os.path.exists(path + "/field.h5"):
            self.field = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/field.h5", "r") as f:
                f.read(self.field, "field")

        if os.path.exists(path + "/field_grad_mag.h5"):
            self.field_grad_mag = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/field_grad_mag.h5", "r") as f:
                f.read(self.field_grad_mag, "field_grad_mag")

        if os.path.exists(path + "/residual.h5"):
            self.residual = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/residual.h5", "r") as f:
                f.read(self.residual, "residual")

        if os.path.exists(path + "/laplacian.h5"):
            self.laplacian = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/laplacian.h5", "r") as f:
                f.read(self.laplacian, "laplacian")

        if os.path.exists(path + "/d_potential.h5"):
            self.d_potential = d.Function(self.V)
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/d_potential.h5", "r") as f:
                f.read(self.d_potential, "d_potential")

        if os.path.exists(path + "/field_grad.h5"):
            self.field_grad = d.Function(self.V_vector)
            with d.HDF5File(self.mesh.mpi_comm(),
                            path + "/field_grad.h5", "r") as f:
                f.read(self.field_grad, "field_grad")

        return None
