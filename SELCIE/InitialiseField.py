#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:05:21 2022

@author: Chad Briddon

Code to define initial field to be used by field solver.
"""
import dolfin as d


class InitialiseField(d.UserExpression):
    def __init__(self, p, profiles, degree=0):
        '''
        A class used to define the initial field. The field is a piecewise
        solution of field profiles in the various subdomains defining the mesh.

        Parameters
        ----------
        p : SECLIE.DensityProfiles.DensityProfile
            Density profile class containing mesh information.
        profiles : list of function
            List containing functions that define the field inside each
            subdomain of the mesh contained in 'p'. The functions are assigned
            to the subdomains in numerical order. E.g. first function in list
            assigned to subdomain with index 0. If 'profiles' is too long the
            extra functions will be left unsed.
        degree : int, optional
            Degree of finite-element space. The default is 0.

        Returns
        -------
        None.

        '''

        super().__init__(degree)

        # Assign inputs.
        self.profiles = profiles
        self.subdomains = p.subdomains

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
