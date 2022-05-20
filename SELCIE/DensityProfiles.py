#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:47:03 2021

@author: Chad Briddon

Code to construct system using saved mesh and user defined density profiles.
"""
import os
import dolfin as d


class DensityProfile(d.UserExpression):
    def __init__(self, filename, dimension, symmetry, profiles, degree=0):
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

        '''

        super().__init__(degree)

        file_path = "Saved Meshes/" + filename
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
