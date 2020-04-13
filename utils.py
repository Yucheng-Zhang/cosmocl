'''
Some useful functions.
'''
import numpy as np
from scipy.special import spherical_jn


def tab_jls(fo, ells, kchis):
    '''Tabulate the spherical bessel functions at sample points.'''
    jls = None

    for i, ell in enumerate(ells):
        print('ell : {:d}'.format(ell))
        jls_ = np.array([spherical_jn(ell, kchis)])
        if i == 0:
            jls = jls_
        else:
            jls = np.concatenate((jls, jls_))
        print(jls.shape)

    np.savez(fo, ells=ells, jls=jls)
