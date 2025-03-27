#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:08:30 2025

@author: chad-briddon

Reproduce the exact analytic solution for a pair of parellel plates found in
"https://arxiv.org/abs/1606.06867".

*** Note there is a mistake in the paper. There is a missing factor of two
    equations 5 and 6. This means in equations 13 and 14 the L_s should be
    replaned by 2*L_s. Below we use the corrected result.

"""

import numpy as np
import dolfin as d
import matplotlib.pyplot as plt

from scipy.special import ellipeinc
from SELCIE import MeshingTools, DensityProfile, FieldSolver


# Set parameters.
L = 1
plate_thickness = 0.1
Ps, Pv = 1e10, 1

n = 1
compton_wl_s = 0.01
alpha = (n+1)*(compton_wl_s**2)*(Ps**((n+2)/(n+1)))


' ---------------------------- Create functions ---------------------------- '


def create_mesh_parellel_plates(L, plate_thickness, show_mesh=True):

    filename = "parellel_plates"

    # Make mesh.
    MT = MeshingTools(dimension=1, display_messages=False)

    plate_L = MT.create_1D_line(-plate_thickness, 0)
    plate_R = MT.create_1D_line(L, L+plate_thickness)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.010, DistMin=0.0,
                        DistMax=0.1)

    MT.create_1D_line(-plate_thickness, L+plate_thickness,
                      embed=plate_L+plate_R)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.001, DistMin=0.05,
                        DistMax=0.1)

    MT.generate_mesh(filename, show_mesh)
    MT.msh_2_xdmf(filename, auto_override=True)

    return filename


def plate(x):
    return Ps


def vacuum(x):
    return Pv


def Sn(u, k):

    E = ellipeinc(u, k**2)
    Sn = (u - E)/k**2

    return Sn


def inverse_sn(y, k, range_):
    'Since sn is monotoic function, use bisection method to find its inverse.'
    'Find f(x) = sn(x, k) - y = 0.'

    sn_inv = []
    for yi in y:

        a, b = range_
        while b-a > 1e-10:

            c = (b+a)/2

            if Sn(c, k) - yi > 0:
                b = c
            else:
                a = c

        sn_inv.append(c)

    return np.array(sn_inv)


' ------------------------ Compute numeric solution ------------------------ '

mesh_name = create_mesh_parellel_plates(L, plate_thickness, show_mesh=False)

p = DensityProfile(filename=mesh_name, dimension=1,
                   symmetry="translation symmetry", profiles=[plate, vacuum])

s = FieldSolver(alpha, n, density_profile=p)
s.picard()

s.plot_results(field_scale='linear')


' ---------------------- Compare to analytic solution ---------------------- '

z = np.linspace(0, L, 1000)
k = d.norm(s.field.vector(), 'linf')
Ls = np.sqrt(2*alpha*(k**3))

y = Sn(np.pi/2, k) - abs(1 - 2*z)/(4*Ls)
theta = inverse_sn(y, k, range_=(0, np.pi/2))

phi = k*np.sin(theta)**2

plt.plot(z, phi)

# Plot relative error.
rel_err = np.array([abs((phi_i - s.field(z_i, 0))/s.field(z_i, 0))
                    for z_i, phi_i in zip(z, phi)])

plt.figure()
plt.title("Relative error (numeric vs analytic)")
plt.yscale("log")
plt.ylabel(r"$\Delta$")
plt.xlabel("z")
plt.plot(z, rel_err)

'Note the relative error scales with mesh refinement.'


' ----------------------- Check Laplacian & residual ----------------------- '
s.calc_laplacian()
s.calc_field_residual()
s.plot_results(lapl_scale='log', res_scale='log')
