#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:47:03 2021

@author: ppycb3

Library of density profile function.
"""
import os
import sys
import dolfin as d


class DensityProfile(d.UserExpression):
    def __init__(self, filename, dimension, symmetry, profiles, **kwargs):
        super().__init__(**kwargs)

        if os.path.isdir(filename):
            pass
        else:
            print('Directory %s does not exist.' % filename)
            sys.exit()

        # Import Mesh, subdomain and boundary information.
        self.mesh = d.Mesh()
        with d.XDMFFile(filename + "/mesh.xdmf") as meshfile:
            meshfile.read(self.mesh)
            self.subdomains = d.MeshFunction('size_t', self.mesh, dimension)
            meshfile.read(self.subdomains, "Subdomain")

        with d.XDMFFile(filename + "/boundaries.xdmf") as boundaryfile:
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

        return None

    def value_shape(self):
        return ()
