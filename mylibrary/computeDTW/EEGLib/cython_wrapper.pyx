from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np
ctypedef np.float32_t float

cdef extern from "EEG_lib.hpp":
    vector[float] _cppdistmtx "get_distance_matrix"(float S[], int nS,  float Q[], int nQ, 
                                int sEpoPat, int stride, int w, int verbose)

import numpy as np
def GetDistMtx(S, nS, Q, nQ, sEpoPat, stride, w, verbose):
    '''Compute the DTW distance matrix for time series S and Q after z-normalizing the series, using pattern=symmetric1 and partial eculidean distance (no square root)'''
    if not (S.flags['C_CONTIGUOUS']):
        print('Ooopss... we had to make a copy of the S array')
        S = np.ascontiguousarray(S) # Makes a contiguous copy of the numpy array.
    if not (Q.flags['C_CONTIGUOUS']):
        print('Ooopss... we had to make a copy of the Q array')
        Q = np.ascontiguousarray(Q) # Makes a contiguous copy of the numpy array.

    cdef float[::1] S_memview = S
    cdef float[::1] Q_memview = Q
    cdef int v
    if verbose==True:
        v=1
    else:
        v=0

    return _cppdistmtx(&S_memview[0], nS, &Q_memview[0], nQ, sEpoPat, stride, w, v)

cdef extern from "EEG_lib.hpp":
    vector[float] _cppdistmtxu "get_distance_matrix_u"(float S[], int nS,  float Q[], int nQ, 
                                int sEpoPat, int stride, int w, int verbose)

def GetDistMtxU(S, nS, Q, nQ, sEpoPat, stride, w, verbose):
    '''Compute the DTW distance matrix for time series S and Q without z-normalizing the series, using pattern=symmetric2 and full eculidean distance (including square root)'''
    if not (S.flags['C_CONTIGUOUS']):
        print('Ooopss... we had to make a copy of the S array')
        S = np.ascontiguousarray(S) # Makes a contiguous copy of the numpy array.
    if not (Q.flags['C_CONTIGUOUS']):
        print('Ooopss... we had to make a copy of the Q array')
        Q = np.ascontiguousarray(Q) # Makes a contiguous copy of the numpy array.

    cdef float[::1] S_memview = S
    cdef float[::1] Q_memview = Q
    cdef int v
    if verbose==True:
        v=1
    else:
        v=0

    return _cppdistmtxu(&S_memview[0], nS, &Q_memview[0], nQ, sEpoPat, stride, w, v)

cdef extern from "EEG_lib.hpp":
    vector[float] _cppdistmtxuns "get_distance_matrix_u_ns"(float S[], int nS,  float Q[], int nQ, 
                                int sEpoPat, int stride, int w, int verbose)

def GetDistMtxUNS(S, nS, Q, nQ, sEpoPat, stride, w, verbose):
    '''Compute the DTW distance matrix for time series S and Q without z-normalizing the series, using pattern=symmetric2 and partial eculidean distance (no square root)'''
    if not (S.flags['C_CONTIGUOUS']):
        print('Ooopss... we had to make a copy of the S array')
        S = np.ascontiguousarray(S) # Makes a contiguous copy of the numpy array.
    if not (Q.flags['C_CONTIGUOUS']):
        print('Ooopss... we had to make a copy of the Q array')
        Q = np.ascontiguousarray(Q) # Makes a contiguous copy of the numpy array.

    cdef float[::1] S_memview = S
    cdef float[::1] Q_memview = Q
    cdef int v
    if verbose==True:
        v=1
    else:
        v=0

    return _cppdistmtxuns(&S_memview[0], nS, &Q_memview[0], nQ, sEpoPat, stride, w, v)
