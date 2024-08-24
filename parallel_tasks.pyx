# parallel_tasks.pyx
from cython.parallel import prange
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef public void process_data(double[:] arr) nogil:
    cdef int i, n = arr.shape[0]
    with nogil:
        for i in prange(n, schedule='dynamic', num_threads=4):
            arr[i] *= 2  # Multiplica cada elemento por 2
