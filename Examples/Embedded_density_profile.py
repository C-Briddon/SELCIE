#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:11:03 2022

@author: Chad Briddon

Example of constructing an embedded density distribution from multiple
functions.
"""
import SELCIE

MT = SELCIE.MeshingTools(dimension=2)

# For each 'independent' density profile in your system create a subdoamin.
region_0 = MT.create_ellipse(rx=0.1, ry=0.1)
MT.create_subdomain(CellSizeMin=1e-3, CellSizeMax=1e-1,
                    DistMin=0.0, DistMax=0.1)

hole_1 = MT.create_ellipse(rx=0.1, ry=0.1)
region_1 = MT.create_ellipse(rx=0.5, ry=0.8)
region_1 = MT.subtract_shapes(shapes_1=region_1, shapes_2=hole_1)
MT.create_subdomain(CellSizeMin=1e-2, CellSizeMax=1e-1,
                    DistMin=0.0, DistMax=0.1)

hole_2 = MT.create_ellipse(rx=0.5, ry=0.8)
region_2 = MT.create_ellipse(rx=1.2, ry=1.0)
gas_2 = MT.subtract_shapes(shapes_1=region_2, shapes_2=hole_2)
MT.create_subdomain(CellSizeMin=1e-2, CellSizeMax=1e-1,
                    DistMin=0.0, DistMax=0.1)

# Generate mesh and convert to correct format.
MT.generate_mesh(filename='Embedded_Densities', show_mesh=False)
MT.msh_2_xdmf(filename='Embedded_Densities', delete_old_file=True,
              auto_override=True)

# Define density distributions as functions.


def f0(x):
    return 2


def f1(x):
    return x[1]**2 + 1


def f2(x):
    return x[0]**2 + 1


# Create density profile by assigning functions to each of the subdomains.
p = SELCIE.DensityProfile('Embedded_Densities', dimension=2,
                          symmetry='vertical axis-symmetry',
                          profiles=[f0, f1, f2])

# Show results.
s = SELCIE.FieldSolver(alpha=1, n=1, density_profile=p)
s.calc_density_field()
s.plot_results(density_scale='linear')
