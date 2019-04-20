'''
Some misc functions.
'''
import numpy as np


def snr(sig, err):
    '''Signal to noise ratio.'''
    return np.sqrt(np.sum((sig / err)**2))


# def estA(cl, cl_t, err, cov):
#     '''Estimate A with error. \hat{C}_\ell = A*C_\ell'''
#     nb = len(cl)  # number of ell bins
#     A_ell = cl / cl_t
#     err_A_ell = err / cl_t
#     C_A = np.zeros(shape=(nb, nb))  # covariance matrix of A
#     for i in range(nb):
#         for j in range(nb):
#             C_A[i, j] = cov[i, j] / ()
