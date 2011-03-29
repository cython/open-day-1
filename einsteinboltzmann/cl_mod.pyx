from __future__ import division

cdef double pi, c, x_start_rec, x_end_rec

import numpy as np
from numpy import pi, exp
from time_mod import get_H_p, get_dH_p, get_eta, get_H, x_start_rec, x_end_rec
from rec_mod import get_g, get_dg, get_ddg, get_tau, get_dtau, get_ddtau
from evolution_mod import compute_einstein_boltzmann_grid, k_min, k_max, x_init
from time_mod cimport get_H_p_fast
import os
from time import clock
import cPickle as pickle
from gsl.spline cimport Spline, MultiSpline
cimport gsl.spline
import gsl.spline
cimport numpy as np
from params import Omega_b, Omega_m, Omega_r, H_0, c
from libc.math cimport fabs


ctypedef int (*spline_eval_func_type)(void *, double* xa, double* ya, size_t size, double x,
                                      gsl.spline.gsl_interp_accel *, double * y)    

cimport cython

#
# More zeroth order stuff related to HH'. Logically belongs to
# time_mod perhaps.
#

def get_H_dH(x):
    return - H_0**2 / 2 * (3 * (Omega_m + Omega_b) * exp(-3*x)
                           + 4 * Omega_r * exp(-4*x))

def get_H_dH_derivative(x):
    return H_0**2 / 2 * (9 * (Omega_m + Omega_b) * exp(-3*x)
                         + 16 * Omega_r * exp(-4*x))

def get_H_p_dH_p(x):
    return exp(2*x) * (get_H(x)**2 + get_H_dH(x))

def get_H_p_dH_p_derivative(x):
    return exp(2*x) * (2 * get_H(x)**2
                       + 4 * get_H_dH(x)
                       + get_H_dH_derivative(x))

#
# Grid evaluation and resampling
#

def resample(values, oldgrid, newgrid, out=None):
    """
    Resamples a 1D function evaluated in oldgrid onto newgrid, by
    creating a spline and evaluating it.
    """
    spline = Spline(oldgrid, values)
    return spline.evaluate(newgrid, out=out)

def resample_S(S_values, old_k_grid, new_k_grid, old_x_grid, x_nums, new_x_grids):
    """
    Resamples any function in (k,x), such as S, to a new (presumably
    higher) resolution in k, and then resamples along the x axis
    according to provided grids.  x_grid[ik, :x_nums[ik]] is the
    x-grid for k value ik.
    """
    assert S_values.shape == (old_k_grid.shape[0], old_x_grid.shape[0])

    cdef int ik, ix
    # Resample k
    tmp = np.zeros(shape=(new_k_grid.shape[0], old_x_grid.shape[0]))
    for ix in range(old_x_grid.shape[0]):
        tmp[:, ix] = resample(S_values[:, ix].copy(), old_k_grid, new_k_grid)
    # Resample x
    out = np.zeros(shape=(new_k_grid.shape[0], np.max(x_nums)))
    for ik in range(new_k_grid.shape[0]):
        x_grid = new_x_grids[ik, :x_nums[ik]]
        resample(tmp[ik, :], old_x_grid, x_grid, out=out[ik, :x_nums[ik]])
    return out

def make_x_grids(np.ndarray[double] k_grid, double N_per_wave=10):
    """
    Creates grids in x for each value of k. The start point is
    x_start_rec, and endpoint is today.

    Each k will yield have a different number of grid points.
    The grid is created so that N_per_wave evaluation points
    are made per oscillation of the spherical bessel function.
    Still, there's a minimum of 300 points during recombination
    and 200 points after, which will capture any features of the
    source function.

    Returns (x_nums, x_grids), where x_grid[ik, :x_nums[ik]]
    contains the grid for ik.
    """
    import evolution_mod
    
    cdef double delta_x, x, k, min_delta_x_rec, min_delta_x_after
    cdef int ix, ik, Nx
    cdef np.ndarray[double, ndim=2] x_grid
    cdef np.ndarray[int] x_nums = np.zeros(k_grid.shape[0], dtype=np.intc)

    min_delta_x_rec = (x_end_rec - x_start_rec) / 200
    min_delta_x_after = - x_end_rec / 300

    # Overallocate x_grid
    delta_x = 2 * pi * get_H_p_fast(0) / c / np.max(k_grid) / N_per_wave
    x_grid = np.zeros((k_grid.shape[0], np.max(
        [np.round(-x_start_rec / delta_x), 510])), dtype=np.float64)

    for ik in range(k_grid.shape[0]):
        k = k_grid[ik]
        # Start from end, then dynamically increase delta_x.
        # Finally reverse the resulting grid (we don't know
        # how big it will be up front).
        x_grid[ik, 0] = x = 0
        ix = 1
        while x > x_start_rec:
            delta_x = 2 * pi * get_H_p_fast(x) / c / k / N_per_wave
            if x < x_end_rec:
                delta_x = delta_x if delta_x < min_delta_x_rec else min_delta_x_rec
            else:
                delta_x = delta_x if delta_x < min_delta_x_after else min_delta_x_after
            x -= delta_x
            x_grid[ik, ix] = x
            ix += 1
        # Store grid size
        x_nums[ik] = Nx = ix
        # Reverse grid
        for ix in range(0, Nx // 2):
            x_grid[ik, ix], x_grid[ik, Nx - ix - 1] = x_grid[ik, Nx - ix - 1], x_grid[ik, ix]
        
    return x_nums, x_grid[:, :np.max(x_nums)].copy()

def compute_S(eb):
    """
    Computes the source function S (=Stilde in Callin).
    eb should be an EinsteinBoltzmannSolution object
    from evolution_mod.
    """
    x_grid = eb.x_grid
    k_grid = eb.k_grid

    # All variables here will be arrays with k along the first
    # axis and x along the second. Variables such as g which only
    # depend on one variable is stored in a 1-by-n or n-by-1 array
    # which then automatically "broadcast" (repeats itself to match).
    # arr[None, :] reshapes an n-array to a 1-by-n-array.
    H_p_dH_p = get_H_p_dH_p(x_grid)[None, :]
    H_p_dH_p_derivative = get_H_p_dH_p_derivative(x_grid)[None, :]
    H_p = get_H_p(x_grid)[None, :]
    dH_p = get_dH_p(x_grid)[None, :]
    tau = get_tau(x_grid)[None, :]
    dtau = get_dtau(x_grid)[None, :]
    ddtau = get_ddtau(x_grid)[None, :]
    np.seterr(all='ignore')
    g = get_g(x_grid)[None, :]
    dg = get_dg(x_grid)[None, :]
    ddg = get_ddg(x_grid)[None, :]
    k = k_grid[:, None]
    Theta = eb.Theta
    dTheta = eb.dTheta

    ddTheta2 = (+  (2/5) * c * k / H_p * (-(dH_p / H_p) * Theta[1, :, :] + dTheta[1, :, :])
                -  (3/5) * c * k / H_p * (-(dH_p / H_p) * Theta[3, :, :] + dTheta[3, :, :])
                + (9/10) * (ddtau * Theta[2, :, :] + dtau * dTheta[2, :, :]))

    # First term
    A = g * (eb.Theta[0, :, :] + eb.Psi + (1/4)*eb.Theta[2, :, :])

    # Second term
    B = np.exp(-tau) * (eb.dPsi - eb.dPhi)

    # Third term
    C = -1/(c*k) * (dH_p * g * eb.v_b + H_p * dg * eb.v_b + H_p * g * eb.dv_b) #* -10

    # Fourth term
    D = 3/4/(c*k)**2 * (
           H_p_dH_p_derivative * g * Theta[2, :, :]
           + 3 * H_p_dH_p * (dg * Theta[2, :, :] + g * dTheta[2, :, :])
           + H_p**2 * (ddg * Theta[2, :, :] + 2 * dg * dTheta[2, :, :]
                       + g * ddTheta2))

    S = A + B + C + D
    return S

_spherical_bessel_cache = None # in-session cache
def get_spherical_bessels(zmax, lmax=1200, cache_filename='sphbessel.pickle',
                          double atol=1e-30, N_per_wave=5):
    """
    Function to compute splines for the spherical bessel functions.
    
    Returns (splines, zmins), each of length lmax+1, the first is a list
    of spline objects and the second an array of zmin values (the functions
    are 0 before this point).

    atol is used to determine whether the function is 0, for determining
    zmin, only.

    The resulting splines are cached to the file given by cache_filename.
    zmax should be the largest parameter needed. Changing either zmax
    or lmax to something larger than what is present in cache, the
    cache is recomputed.
    """
    global _spherical_bessel_cache
    import cPickle as pickle
    from sphbess import sphbes_sj

    cdef np.ndarray[double, ndim=2, mode='c'] j_values
    cdef Py_ssize_t l, zi

    zmax += 10 # cut some slack

    cache = _spherical_bessel_cache
    if cache is None and os.path.exists(cache_filename):
        print 'Loading spherical bessel functions from cache %s...' % cache_filename
        with file(cache_filename, 'r') as f:
            cache = pickle.load(f)

    if cache is not None:
        if cache['lmax'] < lmax:
            print 'Cached for lmax=%d, need for lmax=%d, recomputing' % (cache['lmax'], lmax)
        elif cache['zmax'] < zmax:
            print 'Cached for zmax=%e, need for zmax=%e, recomputing' % (cache['zmax'], zmax)
        else:
            # We're done, return the cached splines
            return cache['splines'], cache['zmins']

    print 'Computing spherical bessel functions for lmax=%d, zmax=%e' % (lmax, zmax)
    # Need to (re)compute.
    # Choose our delta_z by the argument from Callin. We select the number of
    # samples dynamically in response to zmax, so only grid step size matter
    delta_z = 2 * pi / N_per_wave

    # Sample function at every combination of l and z
    z_grid = np.arange(0, zmax, delta_z) # from 0 to zmax by delta_z
    ls = np.arange(lmax + 1)
    t0 = clock()
    j_values = sphbes_sj(ls[:, None], z_grid[None, :])
    print 'Done in %.3e secs' % (clock() - t0)

    # Now, find the zmins arrays
    cdef np.ndarray[double, mode='c'] zmins = np.zeros(lmax + 1, np.double)
    zmins[0] = 0 # first bessel starts at 1
    for l in range(1, lmax + 1):
        for zi in range(1, z_grid.shape[0]):
            if fabs(j_values[l, zi]) > atol:
                zmins[l] = z_grid[zi - 1]
                break

    # Make splines
    splines = [Spline(z_grid, j_values[l, :]) for l in range(lmax + 1)]

    # Finally, cache to file and return result
    cache = dict(splines=splines, zmins=zmins, lmax=lmax, zmax=zmax)
    _spherical_bessel_cache = cache
    with file(cache_filename, 'w') as f:
        pickle.dump(cache, f, protocol=2)        
    return splines, zmins

class PowerSpectrumIntegrator:
    """
    Class for computing the power spectrum. We use a class to store
    the various dynamic grids etc. and their derived quantities (which
    are precomputed to speed things up and therefore adds some complexity,
    which we handle through using a class).

    eb - If provided, contains a prevously computed solution to the
         Einstein-Boltzmann equations (otherwise they'll be computed
         when constructing the object)

    n_k - Number of k values to use

    lmax - Maximum l value queried
    """

    def __init__(self, eb=None, n_k=1700, k_grid=None, lmax=1200):
        if eb is None:
            # Must compute E-B solutions. Use coarse grid described in Callin.
            import time_mod
            print 'Computing E-B solutions...'
            t0 = clock()
            eb = compute_einstein_boltzmann_grid(x_grid=time_mod.x_grid)
            print 'done in %.2f secs!' % (clock() - t0)
        if k_grid is None:
            k_grid = np.linspace(eb.k_grid[0], eb.k_grid[-1], n_k)
        self.eb = eb
        self.k_grid = k_grid
        self.lmax = lmax
        
        # Create dynamic grid
        self.x_nums, self.x_grids = make_x_grids(k_grid, N_per_wave=10)

        # Compute source function
        S_lowres = compute_S(eb)
        self.S = resample_S(S_lowres, eb.k_grid, self.k_grid, eb.x_grid,
                            self.x_nums, self.x_grids)

        # Compute dynamic-grid-derived quantities: horizon_delta and dx.
        # These correspond to x_grids w.r.t. indexing
        self.horizon_delta = get_eta(0) - get_eta(self.x_grids)
        self.dxs = self.x_grids[:, 1:] - self.x_grids[:, :-1]

        # Get spherical Bessel functions
        self.j_splines, self.j_zmins = get_spherical_bessels(lmax=lmax,
                                                             zmax=get_eta(0) * k_max)

        # Done, everything set up for computing Theta_k and C_l
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_Theta_k(self, int l, np.ndarray[double, mode='c'] out=None):
        """
        Grid-evaluate Theta_l(k) for all values in our k_grid, for the l given.
        
        This is done through line-of-sight integration.
        """
        cdef Py_ssize_t ik, ix, nx
        cdef double k, I, jval, z, prev_dx, dx, prev_x, x

        cdef Spline j_spline = self.j_splines[l]
        cdef double zmin = self.j_zmins[l]
        cdef np.ndarray[double, mode='c'] k_grid = self.k_grid

        cdef np.ndarray[double, ndim=2, mode='c'] S = self.S
        cdef np.ndarray[double, ndim=2, mode='c'] horizon_delta = self.horizon_delta
        cdef np.ndarray[double, ndim=2, mode='c'] dxs = self.dxs
        cdef np.ndarray[int, ndim=1, mode='c'] x_nums = self.x_nums

        # Here follows some pretty low-level stuff which speeds up
        # spline evaluation in this particular case. j_eval is the
        # spline evaluation routine pointer.
        cdef gsl.spline.gsl_interp_accel* spline_acc = gsl.spline.gsl_interp_accel_alloc()
        cdef spline_eval_func_type j_eval = j_spline.interp.type.eval
        cdef void* j_state = j_spline.interp.state
        cdef size_t j_len = j_spline.interp.size
        cdef double* xa = j_spline.xa
        cdef double* ya = j_spline.ya

        if out is None:
            out = np.zeros_like(k_grid)

        # Evaluate the integral as we go, by using the trapezoidal rule
        # with *non-uniform* grid cell sizes
        for ik in range(k_grid.shape[0]):
            k = k_grid[ik]
            nx = x_nums[ik]
            I = 0

            if k * horizon_delta[ik, 0] < zmin:
                # Wohoo, bessel is all zero
                out[ik] = 0
                continue # In Python/C/Java/everything but Fortran,
                         # continue means "next loop iteration"

            # Evaluate first grid point
            z = k * horizon_delta[ik, 0]
            j_eval(j_state, xa, ya, j_len, z, spline_acc, &jval)
            I += (S[ik, 0] * jval) * dxs[ik, 0]

            # Inner grid points
            for ix in range(1, nx - 1):
                z = k * horizon_delta[ik, ix]
                if z < zmin:
                    continue
                j_eval(j_state, xa, ya, j_len, z, spline_acc, &jval)
                I += (S[ik, ix] * jval) * (dxs[ik, ix - 1] + dxs[ik, ix])

            # Last grid point
            z = k * horizon_delta[ik, nx - 1]
            j_eval(j_state, xa, ya, j_len, z, spline_acc, &jval)
            I += (S[ik, nx - 1] * jval) * dxs[ik, nx - 1 - 1]

            out[ik] = I * 0.5

        gsl.spline.gsl_interp_accel_free(spline_acc)
        return out



    def compute_power_spectrum(self, n_s=1, ls=None):
        """
        Computes the power spectrum. n_s is the tilt parameter from inflation.
        
        High-level driver function; to get and plot intermediate results
        one can call the lower-level functions.

        ls is a grid at which to do the actual computation. This will
        be splined before it is returned.

        The resulting spectrum is unscaled. You may want to multiply it with
        l*(l+1) (and rescale amplitude) prior to plotting.

        Return value: (l, Cl), each going from 2..max(ls)
        """
        from scipy.integrate import simps
        
        if ls is None:
            ls = [ 2, 3, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                   120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 
                   600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]
        lmax = max(ls)
        if lmax > self.lmax:
            raise ValueError('lmax too high (Bessel functions not computed)')
        Theta_k_lores = np.zeros(self.k_grid.shape[0])
        Theta_k_hires = np.zeros(5000)
        k_grid_hires = np.linspace(self.k_grid[0], self.k_grid[-1], 5000)
        integrand_factor = (c * k_grid_hires / H_0)**(n_s - 1) / k_grid_hires
    
        # Sample Cls
        Cl_samples = np.zeros(len(ls))
        t0 = clock()
        print 'Computing power spectrum'
        for idx, l in enumerate(ls):
            self.compute_Theta_k(l, out=Theta_k_lores)
            resample(Theta_k_lores, self.k_grid, k_grid_hires, out=Theta_k_hires)
            integrand = Theta_k_hires **2 * integrand_factor
            
            # Use SciPy's simpson's method
            Cl_samples[idx] = simps(integrand, x=k_grid_hires)
            
        print 'Time taken: %.3f' % (clock() - t0)

        # Resample Cls
        ls_hires = np.arange(2, lmax+1)
        Cl = resample(Cl_samples, np.array(ls, np.double), np.array(ls_hires, np.double))
        return ls_hires, Cl


    

