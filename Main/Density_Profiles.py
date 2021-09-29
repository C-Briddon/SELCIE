#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:47:03 2021

@author: ppycb3

Environment - fenics2019

Library of density profile functions.
"""
import dolfin as d


class vacuum_chamber_density_profile(d.UserExpression):
    def __init__(self, mesh, subdomain_markers, source_density, vacuum_density,
                 wall_density, mesh_symmetry, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.subdomain_markers = subdomain_markers
        self.source_density = source_density
        self.vacuum_density = vacuum_density
        self.wall_density = wall_density
        self.mesh_symmetry = mesh_symmetry

    def eval_cell(self, values, x, cell):
        if self.subdomain_markers[cell.index] == 1:
            values[0] = self.source_density
        elif self.subdomain_markers[cell.index] == 3:
            values[0] = self.wall_density
        else:
            values[0] = self.vacuum_density

    def value_shape(self):
        return ()


class Density_Profile(d.UserExpression):
    def __init__(self, mesh, subdomain_markers, mesh_symmetry, profiles,
                 **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.subdomain_markers = subdomain_markers
        self.mesh_symmetry = mesh_symmetry
        self.profiles = profiles

        return None

    def eval_cell(self, values, x, cell):
        i = 0
        while i < len(self.profiles):
            if self.subdomain_markers[cell.index] == i:
                values[0] = self.profiles[i](x)
                break
            else:
                i += 1

        return None

    def value_shape(self):
        return ()
