cimport numpy as np
import numpy as np

cdef extern from "gsl/gsl_sf_bessel.h":
    int gsl_sf_bessel_jl_array (int lmax, double x, double* result_array)
    int gsl_sf_bessel_jl_steed_array (int lmax, double x, double* result_array)

def bessel_jl_array(int lmax, double x, np.ndarray[double, mode='c'] out=None,
                    algorithm='default'):
    if out is None:
        out = np.zeros((lmax + 1), np.double)
    cdef int ret
    if algorithm == 'steed':
        ret = gsl_sf_bessel_jl_steed_array(lmax, x, <double*>out.data)
    elif algorithm == 'default':
        ret = gsl_sf_bessel_jl_array(lmax, x, <double*>out.data)
    else:
        raise ValueError()
    if ret != 0:
        raise ValueError("GSL Error: %d" % ret)
    return out
