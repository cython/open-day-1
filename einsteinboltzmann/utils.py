import contextlib
from time import clock

@contextlib.contextmanager
def print_time_taken(premsg=None, postmsg='Done (time taken: %.3f secs)'):
    if premsg is not None:
        print premsg
    t0 = clock()
    yield
    print postmsg % (clock() - t0)
