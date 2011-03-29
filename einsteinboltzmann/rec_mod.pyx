from __future__ import division
import numpy as np
from numpy import log, exp, sqrt, inf, pi
from gsl.spline cimport Spline
from scipy.integrate import ode
import pprint

from params import *
from time_mod import get_H, get_H_p

cimport libc.math as cm

pp = pprint.PrettyPrinter(indent=4)
np.seterr(all='raise') # fail hard on NaN and over/underflow

#
# Code for finding n_e
#
def solve_X_e_saha(x_grid, out, treshold=.99):
    """
    Solves for X_e using the Saha equation. Solving stops
    when the value falls below the given treshold.

    "out" is filled with the resulting values for each
    grid point.

    Returns the last grid index for which a value was computed.
    """
    a_grid = exp(x_grid)
    n_b = Omega_b * rho_c / m_H / a_grid**3
    T_b = T_0 / a_grid
    
    assert a_grid.shape == out.shape
    for i in range(a_grid.shape[0]):
        a = a_grid[i]
        # rhs of Saha equation
        rhs = (sqrt(m_e * T_b[i] * k_b / hbar**2 / 2 / pi)**3
               / n_b[i]
               * exp(-epsilon_0 / k_b / T_b[i]))
        # solve 2nd degree equation
        out[i] = (-rhs + sqrt(rhs**2 + 4*rhs)) / 2
        if out[i] < treshold:
            return i
    raise Exception("Never reached the %f treshold" % treshold)


def peebles_X_e_derivative(x, y):
    """
    Right-hand side of the Peebles' ODE.
    """
    X_e = y[0]

    # Time-dependant "input" quantities
    a = exp(x)
    T_b = T_0 / a
    n_H = Omega_b * rho_c / m_H / a**3
    H = get_H(x)

    # Compute quantities in reverse order of listed in assignment
    phi_2 = 0.448 * log(epsilon_0 / T_b / k_b) # dimensionless
    alpha_2 = (64 * pi / sqrt(27 * pi)
               * (alpha * hbar / m_e)**2 / c
               * sqrt(epsilon_0 / T_b / k_b)
               * phi_2) # units of m^3/s

    # The following beta will become 0 numerically when T_b becomes
    # too small, and is only used in the final expression for the
    # derivative.
    beta = (alpha_2
            * (m_e * T_b * k_b / 2 / pi / hbar**2)**(3/2)
            * exp(-epsilon_0 / T_b / k_b)) # units of 1/s
    
    # In beta_2 we instead insert beta analytically,
    # to get a valid result even when beta underflows above
    beta_2 = (alpha_2
            * (m_e * T_b * k_b / 2 / pi / hbar**2)**(3/2)
            * exp(-(1/4) * epsilon_0 / T_b / k_b)) # units of 1/s
    n_1s = (1 - X_e) * n_H # units of 1/m^3
    Lambda_alpha = (H
                    * (3 * epsilon_0 / c / hbar)**3
                    / (8*pi)**2
                    / n_1s) # units of 1/s
    Lambda_2s_to_1s = 8.227 # units of 1/s

    C_r = ((Lambda_2s_to_1s + Lambda_alpha)
           / (Lambda_2s_to_1s + Lambda_alpha + beta_2)) # dim.less

    # Finally, the main differential equation.
    diff = (beta * (1 - X_e) - n_H * alpha_2 * X_e**2)
    deriv = C_r / H * diff
    return [deriv]

def solve_X_e_peebles(x_grid, X_e_init, out):
    """
    Solves for X_e using Peebles' equation. X_e_init is taken
    to be the initial condition for X_e at the point x_grid[0].
    """
    r = ode(peebles_X_e_derivative)
    r.set_integrator('vode', method='adams', with_jacobian=False,
                     max_step=x_grid[1] - x_grid[0])
    r.set_initial_value(X_e_init, x_grid[0])
    out[0] = X_e_init
    for i in range(1, x_grid.shape[0]):
        r.integrate(x_grid[i])
        if not r.successful():
            raise Exception()
        out[i] = r.y

def compute_X_e(x_grid):
    X_e = np.zeros_like(x_grid)

    # Solve using Saha to the treshold (idx gets the grid point this
    # happens at), then solve using Peebles' on the remainder of the
    # grid/values.
    switch_idx = solve_X_e_saha(x_grid, X_e)
    solve_X_e_peebles(x_grid[switch_idx:], X_e[switch_idx], X_e[switch_idx:])

    # Return gridded X_e
    return X_e

def compute_n_e(x_grid, X_e_values):
    a_grid = exp(x_grid)
    n_H_values = Omega_b * rho_c / m_H / a_grid ** 3
    return X_e_values * n_H_values

xstart = log(1e-10) # Start grid at a = 10^-10
xstop = 0 # Stop grid at a = 1
n = 1000 # Number of grid points between xstart and xstop
x_grid = np.linspace(xstart, xstop, n)

X_e_grid = x_grid
X_e_values = compute_X_e(x_grid)

n_e_values = compute_n_e(x_grid, X_e_values)
log_n_e_spline = Spline(x_grid, log(compute_n_e(x_grid, X_e_values)))

def get_n_e(x):
    return exp(log_n_e_spline.evaluate(x))

#
# Code for finding tau.
#
# tau is truncated to 0 between the second-to-last and last grid point.
#
def tau_derivative(x, y):
    tau = y[0]
    n_e = get_n_e(x)
    H_p = get_H_p(x)
    return -n_e * sigma_T * exp(x) * c / H_p

def solve_tau(x_grid):
    # Integrate from the end of x_grid, i.e. reverse it
    reverse_x_grid = x_grid[::-1]
    out = np.zeros_like(x_grid)
    r = ode(tau_derivative)
    r.set_integrator('vode', method='adams', with_jacobian=False,
                     max_step=x_grid[1] - x_grid[0])
    # There's no probability of scattering during a duration of zero length:
    r.set_initial_value(0, 0)
    out[0] = 0
    for i in range(1, reverse_x_grid.shape[0]):
        r.integrate(reverse_x_grid[i])
        if not r.successful():
            raise Exception()
        out[i] = r.y
    return out[::-1] # remeber to reverse values

tau_values = solve_tau(x_grid)
# When splining it, don't include the last (zero) point
cdef double tau_spline_xstop = x_grid[-2]
cdef Spline log_tau_spline = Spline(x_grid[:-1], log(tau_values[:-1]))

# Python comment: get_tau is supposed to take both array and scalar
# arguments (for easy plotting etc.), which means we must do without
# if-tests (unless we want to code a loop). Just multiply instead...
def get_tau(x):
    return exp(log_tau_spline.evaluate(x)) * (x <= tau_spline_xstop)

def get_dtau(x):
    log_tau = log_tau_spline.evaluate(x)
    d_log_tau = log_tau_spline.derivative(x)
    return exp(log_tau) * d_log_tau * (x <= tau_spline_xstop)

def get_ddtau(x):
    log_tau = log_tau_spline.evaluate(x)
    d_log_tau = log_tau_spline.derivative(x)
    dd_log_tau = log_tau_spline.second_derivative(x)
    return exp(log_tau) * (d_log_tau**2 + dd_log_tau) * (x <= tau_spline_xstop)

# Fast, non-array version
cdef double get_tau_fast(double x):
    return cm.exp(log_tau_spline.evaluate_single(x)) * (x <= tau_spline_xstop)
    
cdef double get_dtau_fast(double x):
    cdef double log_tau, d_log_tau
    log_tau = log_tau_spline.evaluate_fast(x)
    d_log_tau = log_tau_spline.derivative_fast(x)
    return cm.exp(log_tau) * d_log_tau * (x <= tau_spline_xstop)
    
cdef double get_ddtau_fast(double x):
    cdef double log_tau, d_log_tau, dd_log_tau
    log_tau = log_tau_spline.evaluate_fast(x)
    d_log_tau = log_tau_spline.derivative_fast(x)
    dd_log_tau = log_tau_spline.second_derivative_fast(x)
    return cm.exp(log_tau) * (d_log_tau**2 + dd_log_tau) * (x <= tau_spline_xstop)


#
# Code for finding g = g_tilde.
#
# g is set to 0 between the second-to-last and last grid point.
#
log_g_values = log(-get_dtau(x_grid[:-1])) - get_tau(x_grid[:-1])
log_g_spline = Spline(x_grid[:-1], log_g_values)

def get_g(x):
    return exp(log_g_spline.evaluate(x)) * (x <= tau_spline_xstop)

def get_dg(x):
    log_g = log_g_spline.evaluate(x)
    d_log_g = log_g_spline.derivative(x)
    return exp(log_g) * d_log_g * (x <= tau_spline_xstop)

def get_ddg(x):
    log_g = log_g_spline.evaluate(x)
    d_log_g = log_g_spline.derivative(x)
    dd_log_g = log_g_spline.second_derivative(x)
    return exp(log_g) * (d_log_g**2 + dd_log_g) * (x <= tau_spline_xstop)


