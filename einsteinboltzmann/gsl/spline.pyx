
cimport numpy as np
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free

np.import_array()

cdef class Spline:
    
    def __cinit__(self,
                  np.ndarray[double, mode='c'] x,
                  np.ndarray[double, mode='c'] y,
                  algorithm='cspline'):
        self.acc = NULL
        self.interp = NULL
        if y.shape[0] != x.shape[0]:
            raise ValueError('x and y shapes do not match')
        self.npoints = x.shape[0]
        self.x = x
        self.y = y
        self.xa = <double*>x.data
        self.ya = <double*>y.data
        if algorithm == 'cspline':
            self.interp_type = gsl_interp_cspline
        else:
            raise NotImplementedError()
        self.interp = gsl_interp_alloc(self.interp_type, self.npoints)
        gsl_interp_init(self.interp, self.xa, self.ya, self.npoints)
        self.acc = gsl_interp_accel_alloc()

    def __dealloc__(self):
        if self.interp != NULL:
            gsl_interp_free(self.interp)
        if self.acc != NULL:
            gsl_interp_accel_free(self.acc)

    cdef double evaluate_fast(self, double x):
        return gsl_interp_eval(self.interp, self.xa, self.ya, x, self.acc)

    cdef double derivative_fast(self, double x):
        return gsl_interp_eval_deriv(self.interp, self.xa, self.ya, x, self.acc)

    cdef double second_derivative_fast(self, double x):
        return gsl_interp_eval_deriv2(self.interp, self.xa, self.ya, x, self.acc)

    def evaluate(self, x, out=None, hack=False):
        cdef Py_ssize_t i,j
        cdef gsl_interp_accel *acc = self.acc
        cdef gsl_interp *interp = self.interp
        cdef double* xa = self.xa
        cdef double* ya = self.ya
        cdef bint was_scalar = np.isscalar(x)
        cdef double* xbuf, *outbuf
        if was_scalar:
            x = [x]
        x = np.asarray(x, dtype=np.double)
        if out is None:
            out = np.empty_like(x)
        elif out.shape != x.shape:
            raise ValueError()
        cdef np.broadcast mit = np.broadcast(x, out)
        cdef int innerdim = np.PyArray_RemoveSmallest(mit)
        cdef Py_ssize_t xstride = x.strides[innerdim], outstride = out.strides[innerdim]
        cdef Py_ssize_t length = x.shape[innerdim]
        if xstride == sizeof(double) and outstride == sizeof(double):
            # Contiguous case
            while np.PyArray_MultiIter_NOTDONE(mit):
                xbuf = <double*>np.PyArray_MultiIter_DATA(mit, 0)
                outbuf = <double*>np.PyArray_MultiIter_DATA(mit, 1)
                for i in range(length):
                    outbuf[i] = gsl_interp_eval(interp, xa, ya, xbuf[i], acc)
                np.PyArray_MultiIter_NEXT(mit)
        else:
            # Non-contiguous case
            raise NotImplementedError()
        if was_scalar:
            return out[0]
        else:
            return out

    def derivative(self, x, out=None):
        cdef Py_ssize_t i
        was_scalar = np.isscalar(x)
        if was_scalar:
            x = [x]
        cdef np.ndarray[double] xbuf = np.asarray(x, dtype=np.double)
        cdef np.ndarray[double] outbuf = out
        cdef gsl_interp_accel *acc = self.acc
        cdef gsl_interp *interp = self.interp
        cdef double* xa = self.xa
        cdef double* ya = self.ya
        if xbuf is None:
            raise ValueError()
        if out is None:
            outbuf = np.empty_like(xbuf)
        for i in range(xbuf.shape[0]):
            outbuf[i] = gsl_interp_eval_deriv(interp, xa, ya, xbuf[i], acc)
        if was_scalar:
            return outbuf[0]
        else:
            return outbuf

    def second_derivative(self, x, out=None):
        cdef Py_ssize_t i
        was_scalar = np.isscalar(x)
        if was_scalar:
            x = [x]
        cdef np.ndarray[double] xbuf = np.asarray(x, dtype=np.double)
        cdef np.ndarray[double] outbuf = out
        cdef gsl_interp_accel *acc = self.acc
        cdef gsl_interp *interp = self.interp
        cdef double* xa = self.xa
        cdef double* ya = self.ya
        if xbuf is None:
            raise ValueError()
        if out is None:
            outbuf = np.empty_like(xbuf)
        for i in range(xbuf.shape[0]):
            outbuf[i] = gsl_interp_eval_deriv2(interp, xa, ya, xbuf[i], acc)
        if was_scalar:
            return outbuf[0]
        else:
            return outbuf

    def __reduce__(self):
        version = 0
        return (_unpickle, (version, self.x, self.y, 'cspline'))

cdef class MultiSpline:

    def __cinit__(self,
                  np.ndarray[double] x,
                  np.ndarray[double, ndim=2] y,
                  algorithm='cspline'):
        self.acc = NULL
        self.interps = NULL
        cdef Py_ssize_t i, size
        size = x.shape[0]
        if y.shape[1] != x.shape[0]:
            raise ValueError()
        if y.strides[1] != sizeof(double):
            y = y.copy()
        if x.strides[0] != sizeof(double):
            x = x.copy()
        self.nsplines = y.shape[0]
        self.x = x
        self.y = y

        self.interps = <gsl_interp**>malloc(sizeof(gsl_interp*) * y.shape[0])
        self.acc =  gsl_interp_accel_alloc()
        if algorithm == 'cspline':
            for i in range(y.shape[0]):
                self.interps[i] = gsl_interp_alloc(gsl_interp_cspline, size)
                gsl_interp_init(self.interps[i], <double*>x.data,
                                <double*>(y.data + y.strides[0] * i), size)
        else:
            raise NotImplementedError()

    def __dealloc__(self):
        if self.acc == NULL:
            return
        
        cdef Py_ssize_t i
        for i in range(self.y.shape[0]):
            gsl_interp_free(self.interps[i])
        gsl_interp_accel_free(self.acc)
        free(self.interps)

    def evaluate(self, Py_ssize_t splineidx, x, out=None):
        cdef Py_ssize_t i,j
        cdef gsl_interp_accel *acc = self.acc
        if splineidx >= self.nsplines:
            raise ValueError('spline index out of range')
        cdef gsl_interp *interp = self.interps[splineidx]
        cdef bint was_scalar = np.isscalar(x)
        if was_scalar:
            x = [x]
        cdef np.ndarray[double] xbuf = np.asarray(x, dtype=np.double)
        if out is None:
            out = np.empty_like(x)
        elif out.shape != x.shape:
            raise ValueError()
        cdef np.ndarray[double] outbuf = out
        cdef double* xa = <double*>self.x.data
        cdef double* ya = <double*>(self.y.data + self.y.strides[0] * splineidx)
        for i in range(xbuf.shape[0]):
            outbuf[i] = gsl_interp_eval(interp, xa, ya, xbuf[i], acc)
        if was_scalar:
            return out[0]
        else:
            return out


def _unpickle(version, x, y, algorithm):
    if version == 0:
        return Spline(x, y, algorithm)
    else:
        raise ValueError()

def example1():
    """
    The example from the GSL documentation
    """
    cdef int i
    cdef double xi, yi
    cdef np.ndarray[double] x, y
    cdef np.ndarray[double] xhigh, yhigh

    ir = np.arange(10)
    x = (ir + 0.5 * np.sin(ir)).astype(np.double)
    y = (ir + np.cos(ir*ir)).astype(np.double)

    xhigh = np.linspace(x[0], x[9], 100)
    yhigh = np.zeros(100, dtype=np.double)
    
    cdef gsl_interp_accel *acc = gsl_interp_accel_alloc()
    cdef gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, 10)

    try:
        gsl_spline_init(spline, <double*>x.data, <double*>y.data, 10)

        for i in range(xhigh.shape[0]):
            yhigh[i] = gsl_spline_eval(spline, xhigh[i], acc)

    finally:
         gsl_spline_free (spline);
         gsl_interp_accel_free (acc);

    import matplotlib.pyplot as plt
    plt.plot(x, y, 'r', xhigh, yhigh, 'g')
    plt.show()

def example2():
    """
    The example from the GSL documentation using the higher-level interface
    """
    cdef int i
    cdef double xi, yi
    cdef np.ndarray[double] x, y
    cdef np.ndarray[double] xhigh, yhigh

    ir = np.arange(10)
    x = (ir + 0.5 * np.sin(ir)).astype(np.double)
    y = (ir + np.cos(ir*ir)).astype(np.double)

    xhigh = np.linspace(x[0], x[9], 100)

    spline = Spline(x, y)
    yhigh = spline.evaluate(xhigh)


    import matplotlib.pyplot as plt
    plt.plot(x, y, 'r', xhigh, yhigh, 'g')
    plt.show()

             

