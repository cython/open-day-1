errors = []

cdef extern from *:
    ctypedef char const_char "const char"

cdef void _collecting_error_handler(const_char* reason,
                                    const_char* file,
                                    int line,
                                    int gsl_errno) with gil:
    print 'ERROR HANDLER CALLED'
    errors.append('%s:%d: GSL error %d: %s' % (
        (<bytes><char*>file).decode('ASCII'),
        line,
        gsl_errno,
        (<bytes><char*>reason).decode('ASCII')))        


def collect_errors():
    print 'starting to collect errors'
    gsl_set_error_handler(&_collecting_error_handler)
