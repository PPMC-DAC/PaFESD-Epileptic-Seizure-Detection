import EEGLIB
import math
import numpy as np


def test_DistanceMatrix():
    # Datos de prueba
    nS=1000000
    nQ=100000
    S = np.cumsum(np.random.uniform(-0.5, 0.5, nS))
    Q = np.cumsum(np.random.uniform(-0.5, 0.5, nQ))
    S = S.astype(dtype=np.float32)
    Q = Q.astype(dtype=np.float32)

    stride = 256
    nEpoPat = 1280
    w=16

    # Llamar a la función compute_statistics desde el módulo
    distMtx = EEGLIB.GetDistMtx(S, nS, Q, nQ, nEpoPat, stride, w, True)
    dM=np.array(distMtx)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print("Distance Matrix partial output: ")
    print(dM[:5])

    # Llamar a la función compute_statistics desde el módulo
    distMtx = EEGLIB.GetDistMtxU(S, nS, Q, nQ, nEpoPat, stride, w, True)
    dM=np.array(distMtx)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print("Distance Matrix partial output: ")
    print(dM[:5])

# Testing C++ GetDistMtx function
test_DistanceMatrix()
