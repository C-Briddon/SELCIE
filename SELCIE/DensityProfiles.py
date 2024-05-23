#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:47:03 2021

@author: Chad Briddon

Code to construct system using saved mesh and user defined density profiles.
"""
import os
import dolfin as d


def create_boundary_class(func):
    class C(d.SubDomain):
        def inside(self, x, on_boundary):
            return func(x) and on_boundary
    return C


class DensityProfile(d.UserExpression):
    def __init__(self, filename, dimension, symmetry, profiles, degree=0,
                 path=None):
        '''
        A class used to define the piecewise density field attributed to a
        given mesh consisting of some number of subdomains.

        Parameters
        ----------
        filename : str
            The name of the directory containing the saved mesh files.
            From current directory the path to files must be
            'Saved Meshes'/'filename'. Files must be in .xdmf format which can
            be converted from .msh using 'SELCIE.MeshingTools.msh_2_xdmf()'.
        dimension : int
            Number of spacial dimensions of the inputted mesh.
        symmetry : str
            Specify a symmetry to be imposed on the density field.
            For 3D meshes symmetry is used. Options for 2D meshes are:
                'vertical axis-symmetry', 'horizontal axis-symmetry',
                'cylinder slice'.
        profiles : list of function
            List containing the functions that define the density field inside
            each subdomain of the given mesh. The functions are assigned to
            the subdomains in numerical order. E.g. first function in list
            assigned to subdomain with index 0. If 'profiles' is too long the
            extra functions will be left unsed.
        degree : int, optional
            Degree of finite-element space. The default is 0.
        path : None or string, optional
            If saving to a different directory than the current one then
            specify it using path. The default is None.

        '''

        super().__init__(degree)
        
        # Set path to location of directory.
        if path is None:
            file_path = 'Saved Meshes/' + filename
        else:
            file_path = path + '/Saved Meshes/' + filename

        if os.path.isdir(file_path) is False:
            raise Exception("Directory '%s' does not exist." % file_path)

        # Import Mesh, subdomain and boundary information.
        self.mesh = d.Mesh()
        with d.XDMFFile(file_path + "/mesh.xdmf") as meshfile:
            meshfile.read(self.mesh)
            self.subdomains = d.MeshFunction('size_t', self.mesh, dimension)
            meshfile.read(self.subdomains, "Subdomain")

        with d.XDMFFile(file_path + "/boundaries.xdmf") as boundaryfile:
            mvc = d.MeshValueCollection("size_t", self.mesh, dimension)
            boundaryfile.read(mvc, "Boundary")
            self.boundary = d.MeshFunction("size_t", self.mesh, mvc)

        # Assign remaining inputs.
        self.symmetry = symmetry
        self.profiles = profiles

        return None

    def eval_cell(self, values, x, cell):
        i = 0
        while i < len(self.profiles):
            if self.subdomains[cell.index] == i:
                values[0] = self.profiles[i](x)
                break
            else:
                i += 1

        if i == len(self.profiles):
            raise Exception("More subdomains then functions in 'profiles'.")

        return None

    def value_shape(self):
        return ()

    def assign_boundary_labels(self, boundary_definitions):
        '''
        Relabels boundary of mesh according to functions contained within
        'boundary_definitions'. The first element will correspond to boundary
        label 1, the second 2 and so on. Any remaining regions of the boundary
        not covered by 'boundary_definitions' will have index 0.

        Parameters
        ----------
        boundary_definitions : list of function
            List containg functions that defines how to seperate the boundary
            of the mesh into different regions. Each function must take input
            argument x, where x is a coordinate, and return True if x is on
            the region of the boundary you wish to label and False if not.

        Returns
        -------
        None.

        '''

        # Start by setting all boundaries to the default value.
        class Default_Boundary(d.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary
        Default_Boundary().mark(self.boundary, 0)
        #self.boundary.set_all(0)

        # Assign boundary regions.
        for i, func in enumerate(boundary_definitions):
            create_boundary_class(func)().mark(self.boundary, i+1)

        return None
