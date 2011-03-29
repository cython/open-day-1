from __future__ import division
from numpy import pi

# Units
eV  = 1.60217646e-19
Mpc = 3.08568025e22

# Cosmological parameters
Omega_b      = 0.046
Omega_m      = 0.224
Omega_r      = 8.3e-5 # 5.042e-5
Omega_lambda = 1. - Omega_m - Omega_b - Omega_r
T_0          = 2.725
n_s          = 1.
A_s          = 1.
h0           = 0.7
H_0          = h0 * 100. * 1.e3 / Mpc

# General constants
c            = 2.99792458e8
epsilon_0    = 13.605698 * eV
m_e          = 9.10938188e-31
m_H          = 1.673534e-27
sigma_T      = 6.652462e-29
G_grav       = 6.67258e-11
rho_c        = 3.*H_0**2 / (8.*pi*G_grav)
alpha        = 7.29735308e-3
hbar         = 1.05457148e-34
k_b          = 1.3806503e-23
