import numpy as np


def gen_fb(lmin, lmax, nb, fo, scale=None):
    '''Generate band power bin file.'''
    if scale == 'log':
        b1 = np.logspace(np.log10(lmin), np.log10(lmax),
                         num=nb+1, dtype='int32')
    else:
        b1 = np.linspace(lmin, lmax, num=nb+1, dtype='int32')

    b2 = b1 - 1
    b1[0], b2[-1] = lmin, lmax

    data = np.column_stack((b1[:-1], b2[1:]))
    np.savetxt(fo, data, fmt='%6d  %6d')
