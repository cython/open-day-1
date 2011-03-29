from __future__ import division

cdef double Omega_b, Omega_m, Omega_r, Omega_lambda
cdef double H_0, c

import numpy as np
from numpy import log, inf
from scipy.integrate import odeint
from gsl.spline cimport Spline

cimport libc.math as cm# cimport exp, sqrt

from params import Omega_b, Omega_m, Omega_r, Omega_lambda, H_0, c

def debug(x):
    print x
    import sys
    sys.stdout.flush()

#
# Define the grids
#

# Define two epochs, 1) during and 2) after recombination.

n1          = 200                       # Number of grid points during recombination
n2          = 300                       # Number of grid points after recombination
z_start_rec = 1630.4                    # Redshift of start of recombination
z_end_rec   = 614.2                     # Redshift of end of recombination
z_0         = 0                         # Redshift today
x_start_rec = -log(1 + z_start_rec)     # x of start of recombination
x_end_rec   = -log(1 + z_end_rec)       # x of end of recombination
x_0         = 0                         # x today

n_eta       = 1000                      # Number of eta grid points (for spline)
a_init      = 1e-10                     # Start value of a for eta evaluation
x_eta1      = log(a_init)               # Start value of x for eta evaluation
x_eta2      = 0                         # End value of x for eta evaluation

x_grid = np.concatenate((
    np.linspace(np.log(1e-8), x_start_rec, 20, endpoint=False),
    np.linspace(x_start_rec, x_end_rec, n1, endpoint=False),
    np.linspace(x_end_rec, x_0, n2, endpoint=True)))

x_eta_grid = np.linspace(x_eta1, x_eta2, n_eta)

#
# Utility
#

def redshift_to_x(rs):
    return -log(1 + rs)

def x_to_redshift(x):
    return np.exp(-x) - 1


#
# Computation of H. Note: These also work elementwise on whole
# arrays of "x" passed as argument.
#

def get_H(x):
    return H_0 * np.sqrt(
        (Omega_b + Omega_m) * np.exp(-3 * x)
        + Omega_r * np.exp(-4 * x)
        + Omega_lambda)

def get_H_p(x):
    return np.exp(x) * get_H(x)

def get_dH_p(x):
    dHdx = (-H_0 / 2
            * (3 * (Omega_b + Omega_m) * np.exp(-3 * x)
               + 4 * Omega_r * np.exp(-4 * x))
            / np.sqrt((Omega_b + Omega_m) * np.exp(-3 * x) +
                     Omega_r * np.exp(-4*x) + Omega_lambda))
    return np.exp(x) * (get_H(x) + dHdx)    

# Fast, compiled versions which don't work with arrays
cdef double get_H_fast(double x):
    return H_0 * cm.sqrt(
        (Omega_b + Omega_m) * cm.exp(-3 * x)
        + Omega_r * cm.exp(-4 * x)
        + Omega_lambda)

cdef double get_H_p_fast(double x):
    return cm.exp(x) * get_H_fast(x)

#
# Computation and retrieval of eta. get_eta is the front-end
# for user code.
#

def eta_derivative(eta, a):
    # Simplified the expression up-front in order to avoid
    # problems when a is close to 0
    # (since exp(log(0)) != 0...)
    return c / H_0 / np.sqrt(
        (Omega_b + Omega_m) * a
        + Omega_r
        + Omega_lambda * a**4)

def compute_eta(x_grid):
    # Compute it on a grid on a = exp(x).
    # Add an initial 0 to the grid for the boundary condition
    a_grid = np.concatenate(([0], np.exp(x_grid)))
    eta_at_0 = 0 # Light has not travelled before any time has passed

    # Solve ODE
    eta, info = odeint(func=eta_derivative,
                       y0=eta_at_0,
                       t=a_grid,
                       rtol=1e-12,
                       atol=1e-20,
                       full_output=True)
    # trim off first point (t=0) on return (1:)
    # also, odeint returns an n-by-1 array, so index to return a 1-dim array
    return eta[1:,0]

def compute_eta_spline(x_grid):
    eta_vals = compute_eta(x_grid)
    result = Spline(x_grid, eta_vals, algorithm='cspline')
    return result

cdef Spline eta_spline = compute_eta_spline(x_eta_grid)
def get_eta(x):
    r = eta_spline.evaluate(x)
    return r

cdef double get_eta_fast(double x):
    return eta_spline.evaluate_fast(x)
