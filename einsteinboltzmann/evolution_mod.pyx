from __future__ import division

#
# evolution_mod -- AST5220 Milestone 3
#

from params import c, H_0, Omega_b, Omega_r, Omega_m
from time_mod import x_start_rec, get_eta, get_H, get_H_p, get_dH_p
from rec_mod import get_dtau, get_ddtau
import numpy as np
from scipy.integrate import ode

from numpy import inf, arange, array, abs, nan, pi
from numpy import exp, log, sqrt
np.seterr(all='raise') # fail hard on NaN and over/underflow

#
# Accuracy parameters
#
a_init = 1e-8
k_min = 0.1 * H_0 / c
k_max = 1e3 * H_0 / c
n_k = 100
lmax = 6
# Our n_par here includes Psi, for program simplicity. This may
# be changed depending on requirements of milestone 4. (The ODE
# solving doesn't use n_par, only the final parameter computation).
n_par = 6 + lmax + 1

x_init = log(a_init)

def get_tight_coupling_switch(x_grid, k):
    """
    Find the index in x_grid at which one should switch from tight
    coupling to full equations. Simply do linear searches.
    """
    dtau_values = get_dtau(x_grid)
    idx_cond1 = np.nonzero(-dtau_values < 10)[0][0]
    idx_cond2 = np.nonzero(x_grid >= x_start_rec)[0][0]
    idx_cond3 = np.nonzero(get_H_p(x_grid) * dtau_values > -10 * c * k)[0][0]
    return min([idx_cond1, idx_cond2, idx_cond3])

#
# ODE system
#

i_delta = 0
i_delta_b = 1
i_v = 2
i_v_b = 3
i_Phi = 4
i_Theta = 5
i_Theta0 = 5
i_Theta1 = 6
i_Theta2 = 7

class EinsteinBoltzmannEquations:
    """
    Evaluates the Einstein-Boltzmann equations for given k, lmax,
    both in tight coupling regime and using full equation set.

    The Jacobian is only defined for the full equation set, since
    there are so few oscillations during tight coupling.

    Statistics about how often the function and the Jacobian is
    evaluated is available through f_count and jacobian_count
    (and is the reason for making this a class).
    """    

    def __init__(self, tight_coupling, k, lmax):
        if tight_coupling:
            n = 7 # Only include up to Theta_1
        else:
            n = 6 + lmax
        self.dy = np.zeros(n)
        self.jacobian_array = np.zeros((n,n))
        self.tight_coupling = tight_coupling
        self.k = k
        self._lmax = lmax
        self.f_count = self.jacobian_count = 0

    def f(self, x, y):
        """
        Evaluate the derivative vector/right hand side of the E-B ODE.
        """
        dy = self.dy
        k = self.k
        tight_coupling = self.tight_coupling
        lmax = self._lmax

        self.f_count += 1
        
        # Fetch time-dependant variables
        a = exp(x)
        H = get_H(x)
        H_p = a * H
        dtau = get_dtau(x)
        ddtau = get_ddtau(x)
        R = 4 / 3 * Omega_r / Omega_b / a
        if not tight_coupling:
            eta = get_eta(x)

        # Parse the y array
        delta = y[i_delta]
        delta_b = y[i_delta_b]
        v = y[i_v]
        v_b = y[i_v_b]
        Phi = y[i_Phi]
        Theta0 = y[i_Theta0]
        Theta1 = y[i_Theta1]
        if tight_coupling:
            Theta2 = - 20 / 45 * c * k / H_p / dtau * Theta1
        else:
            Theta2 = y[i_Theta2]
        
        # Evaluate components        
        Psi = -Phi - 12 * H_0**2 / c**2 / k**2 / a**2 * Omega_r * Theta2
        dy[i_Phi] = dPhi = (Psi
                              - (c * k / H_p)**2 * Phi / 3
                              + H_0**2 / H_p**2 / 2
                              * (Omega_m / a * delta
                                 + Omega_b / a * delta_b
                                 + 4 * Omega_r / a**2 * Theta0))
        dy[i_Theta0] = dTheta0 = -(c * k / H_p) * y[i_Theta + 1] - dPhi

        if tight_coupling:
            q = ( (-((1 - 2*R) * dtau + (1 + R) * ddtau) * (3*Theta1 + v_b)
                   - (c * k / H_p) * Psi
                   + (1 - (H_p / H)) * (c * k / H_p) * (-Theta0 + 2*Theta2)
                   - (c * k / H_p) * dTheta0)
                 / ((1 + R) * dtau + (H_p / H) - 1))
            dy[i_v_b] = dv_b = ((-v_b - (c * k / H_p) * Psi
                                   + R * (q + (c * k / H_p) * (-Theta0 + 2*Theta2)
                                          - (c * k / H_p) * Psi)) / (1 + R))
        else:
            dy[i_v_b] = dv_b = -v_b - (c * k / H_p) * Psi + dtau * R * (3*Theta1 + v_b)
        
        dy[i_delta_b] = (c * k / H_p) * v_b - 3 * dPhi
        dy[i_v] = -v - (c * k / H_p) * Psi
        dy[i_delta] = (c * k / H_p) * v - 3 * dPhi

        if tight_coupling:
            dy[i_Theta + 1] = 1/3 * (q - dv_b)
        else:
            dy[i_Theta + 1] = ((c * k / H_p) / 3 * (Theta0 - 2*Theta2 + Psi)
                                 + dtau * (Theta1 + v_b / 3))
            for l in range(2, lmax): # Python: Endpoint is exclusive!
                dy[i_Theta + l] = (
                    l / (2*l + 1) * (c * k / H_p) * y[i_Theta + l - 1]
                    - (l+1)/(2*l + 1) * (c * k / H_p) * y[i_Theta + l + 1]
                    + dtau * (y[i_Theta + l] - 0.1 * y[i_Theta + l] * (l == 2)))
            dy[i_Theta + lmax] = ((c * k / H_p) * y[i_Theta + lmax - 1]
                                    - (lmax+1) * c / H_p / eta * y[i_Theta + lmax]
                                    + dtau * y[i_Theta + lmax])
        return dy

    def jacobian(self, x, y):
        """
        Evaluate the Jacobian of the E-B ODE.
        """
        jac = self.jacobian_array
        k = self.k
        lmax = self._lmax

        self.jacobian_count += 1

        n = jac.shape[0]
        if self.tight_coupling:
            raise NotImplementedError("Only bothered to define Jacobian after tight coupling")
        
        # Fetch time-dependant variables
        a = exp(x)
        H = get_H(x)
        H_p = a * H
        dtau = get_dtau(x)
        ddtau = get_ddtau(x)
        R = 4 / 3 * Omega_r / Omega_b / a
        eta = get_eta(x)

        # Evaluate Jacobian and store it in
        # jac[derivative_of_param, with_respect_to_param]
        # The zero elements will always stay zero from the first initialization in
        # __init__.

        partial_dPsi_Phi = -1
        partial_dPsi_Theta2 = -12*(H_0 / c / k / a)**2 * Omega_r

        jac[i_Phi, i_Phi] = -1 - (c*k/H_p)**2 / 3
        tmp = (H_0/H_p)**2 / 2
        jac[i_Phi, i_delta] = tmp * Omega_m / a
        jac[i_Phi, i_delta_b] = tmp * Omega_b / a
        jac[i_Phi, i_Theta0] = tmp * 4 * Omega_r / a**2
        jac[i_Phi, i_Theta0] = (H_0/H_p)**2 * 2 * Omega_r / a**2
        jac[i_Phi, i_Theta2] = partial_dPsi_Theta2

        # delta, delta_b
        for i in range(n):
            jac[i_delta, i] = -3 * jac[i_Phi, i]
            jac[i_delta_b, i] = -3 * jac[i_Phi, i]
        jac[i_delta, i_v] = jac[i_delta_b, i_v_b] = c * k / H_p

        # v
        jac[i_v, i_v] = -1
        jac[i_v, i_Phi] = - c * k / H_p * partial_dPsi_Phi
        jac[i_v, i_Theta2] = - c * k / H_p * partial_dPsi_Theta2

        # v_b
        jac[i_v_b, i_v_b] = -1 + dtau * R
        jac[i_v_b, i_Theta1] = dtau * R * 3
        jac[i_v, i_Theta2] = - c * k / H_p * partial_dPsi_Theta2
        jac[i_v, i_Phi] = - c * k / H_p * partial_dPsi_Phi

        # Thetas
        for i in range(n):
            jac[i_Theta0, i] = -jac[i_Phi, i]
        jac[i_Theta0, i_Theta1] = -c * k / H_p

        jac[i_Theta1, i_Theta0] = c * k / H_p / 3
        jac[i_Theta1, i_Theta1] = dtau
        jac[i_Theta1, i_Theta2] = c * k / H_p * (- (2/3) + partial_dPsi_Theta2)
        jac[i_Theta1, i_v_b] = dtau / 3

        for l in range(2, lmax):
            jac[i_Theta + l, i_Theta + l - 1] = l / (2*l + 1) * c * k / H_p
            jac[i_Theta + l, i_Theta + l + 0] = - (l + 1) / (2*l + 1) * c * k / H_p
            jac[i_Theta + l, i_Theta + l + 1] = dtau * (1 - (l == 2) / 10)
            
        jac[i_Theta + lmax, i_Theta + lmax - 1] = c * k / H_p
        jac[i_Theta + lmax, i_Theta + lmax] = (l + 1) / H_p / eta + dtau        

        return jac
        
def solve_einstein_boltzmann(x_grid,
                             k, rtol=1e-15,
                             nsteps=10**10, max_step=None,
                             min_step=None,
                             use_jacobian=True,
                             out_y=None, out_dy=None):
    """
    Solve the Einstein-Boltzmann equations for a given k.

    On return, out_y and out_dy contains the quantities and their
    derivatives. The first axis is the parameters and the second the
    x grid. The order of the parameters is:

    delta, delta_b, v, v_b, Phi, Theta_0, Theta_1, ..., Theta_lmax, Psi

    Return value: (out_y, out_dy, nf, nj), where nf and nj are the
    number of times the rhs and the Jacobian were evaluated.

    """
    i_Psi = i_Theta + lmax + 1
    
    assert np.isscalar(k)
    if out_y is None:
        out_y = np.zeros((n_par, x_grid.shape[0]))
    if out_dy is None:
        out_dy = np.zeros_like(out_y)

    # Options for the ODE solver
    ode_opts = dict(name='vode',
                    method='bdf',
                    rtol=rtol,
                    max_step=max_step,
                    # The default maximum steps is 500 per grid point, which is
                    # way too low. It doesn't hurt to let this be really big
                    # (set to small to get partial results when debugging)
                    nsteps=nsteps)

    # Find tight coupling regime
    idx_switch = get_tight_coupling_switch(x_grid, k)

    # Initial conditions for tight coupling
    x_init_tight = x_grid[0]
    H_p = get_H_p(x_init_tight)
    Phi = 1
    delta = delta_b = Phi * 3 / 2
    v = v_b = 1/2 * Phi * (c * k / H_p)
    Theta0 = Phi / 2
    Theta1 = - 1/6 * Phi * (c * k / H_p)
    y_init_tight = np.array([delta, delta_b, v, v_b, Phi, Theta0, Theta1])

    # Integrate in tight coupling regime -- Theta[2:] is not included here
    rhs_tight = EinsteinBoltzmannEquations(tight_coupling=True, k=k, lmax=lmax)
    integrator = ode(rhs_tight.f)
    integrator.set_integrator(**ode_opts)
    integrator.set_initial_value(y_init_tight, x_init_tight)
    out_y[:i_Theta1 + 1, 0] = y_init_tight # Python: Slice endpoint not included
    out_dy[:i_Theta1 + 1, 0] = rhs_tight.f(x_init_tight, y_init_tight)
    for i in range(1, idx_switch): # idx_switch is not included!
        integrator.integrate(x_grid[i])
        if not integrator.successful():
            raise Exception()
        out_y[:i_Theta1 + 1, i] = integrator.y
        out_dy[:i_Theta1 + 1, i] = rhs_tight.f(x_grid[i], integrator.y)

    # Assign values to Theta[2:] and dTheta[2:] in the tight coupling
    # regime. This also gives initial conditions for the full equation
    # set. This is all array arithmetic, working on all values until
    # the switchpoint.
    x_grid_tight = x_grid[:idx_switch]
    H_p = get_H_p(x_grid_tight)
    dH_p = get_dH_p(x_grid_tight)
    dtau = get_dtau(x_grid_tight)
    ddtau = get_ddtau(x_grid_tight)
    # Python comment: Changing Theta below will also change out_y
    # accordingly, because Theta is a slice into out_y (NOT
    # a copy). Just like pointers in Fortran.
    Theta = out_y[i_Theta:, :idx_switch]
    dTheta = out_dy[i_Theta:, :idx_switch]

    Theta[2,:] = -20/45 * (c * k / H_p) / dtau * Theta[1,:]
    dTheta[2,:] = -20/45 * c * k * (
        dTheta[1,:] / (H_p * dtau)
        - Theta[1,:] * (dH_p * dtau + H_p * ddtau) / (H_p * dtau)**2)
    for l in range(2, lmax + 1):
        Theta[l,:] = - l/(2*l + 1) * (c * k / H_p) / dtau * Theta[l - 1,:]
        dTheta[l,:] = - l/(2*l + 1) * c * k * (
            dTheta[l-1,:] / (H_p * dtau)
            - Theta[l-1,:] * (dH_p * dtau + H_p * ddtau) / (H_p * dtau)**2)

    # Integrate the full equation set to the end. Remember to not include Psi
    # in the ODE.
    rhs_full = EinsteinBoltzmannEquations(tight_coupling=False, k=k, lmax=lmax)    
    integrator = ode(rhs_full.f, None if not use_jacobian else rhs_full.jacobian)
    integrator.set_integrator(**ode_opts)
    integrator.set_initial_value(out_y[:i_Psi, idx_switch - 1], x_grid[idx_switch - 1])
    for i in range(idx_switch, x_grid.shape[0]):
        x = x_grid[i]
        integrator.integrate(x)
        if not integrator.successful():
            raise Exception()
        out_y[:i_Psi, i] = integrator.y
        out_dy[:i_Psi, i] = rhs_full.f(x_grid[i], integrator.y)

    # Finally, (re)compute Psi and its derivative. This may be removed
    # depending on needs of milestone 4.
    a = np.exp(x_grid)

    # Set up references ("pointers")
    Theta2 = out_y[i_Theta2, :]
    Phi = out_y[i_Phi, :]
    Psi = out_y[i_Psi, :]
    dTheta2 = out_dy[i_Theta2, :]
    dPhi = out_dy[i_Phi, :]
    dPsi = out_dy[i_Psi, :]

    # Then, compute Psi. Psi[:] assigns values rather than reassigning
    # reference.
    Psi[:] = -Phi - 12 * H_0**2 / c**2 / k**2 / a**2 * Omega_r * Theta2
    dPsi[:] = (-dPhi - 12 * H_0**2 / c**2 / k**2 * Omega_r / a**2
               * (dTheta2 - 2*Theta2))

    f_count = rhs_tight.f_count + rhs_full.f_count
    jacobian_count = rhs_tight.jacobian_count + rhs_full.jacobian_count
    return out_y, out_dy, f_count, jacobian_count

#
# Make a class for the solution, so that we can easily pickle it to file
# (see part3.py)
#
class EinsteinBoltzmannSolution:
    def __init__(self, x_grid, k_grid, y, dy):
        self.x_grid = x_grid
        self.k_grid = k_grid
        
        # y has indices (parameter, k, x); unpack into seperate
        # variables for each parameter
        self.delta, self.delta_b, self.v, self.v_b, self.Phi = y[:i_Theta, :, :]
        self.Theta = y[i_Theta:i_Theta + lmax + 1, :, :]
        self.Psi = y[i_Theta + lmax + 1, :, :]
        
        self.ddelta, self.ddelta_b, self.dv, self.dv_b, self.dPhi = dy[:i_Theta, :, :]
        self.dTheta = dy[i_Theta:i_Theta + lmax + 1, :, :]
        self.dPsi = dy[i_Theta + lmax + 1, :, :]

#
# Set up grids in k and x and solve ODE for each k. This is a callable
# function which is *not* called automatically due to the computing
# time, so that plotting scripts etc. can use a pickled version.
#
def compute_einstein_boltzmann_grid(x_grid=None, k_grid=None, n_x=2500,
                                    use_jacobian=True):
    """
    Does all the computation for solving the Einstein-Boltzmann
    equations over a default grid over k. The result is an
    EinsteinBoltzmannSolution object.
    """        
    if k_grid is None:
        # Grid in k is quadratic!
        k_grid = k_min + (k_max - k_min) * (np.arange(n_k) / (n_k-1))**2
    if x_grid is None:
        x_grid = np.linspace(x_init, 0, n_x)

    y = np.zeros((n_par, n_k, x_grid.shape[0]), np.double)
    dy = np.zeros_like(y)
    fctot = 0; jctot = 0
    for idx_k, k in enumerate(k_grid):
        a, b, fc, jc = solve_einstein_boltzmann(x_grid, k,
                                                out_y=y[:, idx_k, :],
                                                out_dy=dy[:, idx_k, :],
                                                use_jacobian=use_jacobian)
        fctot += fc
        jctot += jc
    return EinsteinBoltzmannSolution(x_grid, k_grid, y, dy)


