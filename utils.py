'''
Some useful functions.
'''
import numpy as np
import h5py
from scipy.special import spherical_jn


def jl_2nd(ell, x=None, jls=None):
    '''Second order derivative of spherical Bessl function, j_ell(x).'''
    if jls is None:
        t1 = spherical_jn(ell-2, x)
        t2 = spherical_jn(ell, x)
        t3 = spherical_jn(ell+2, x)
    else:
        t1, t2, t3 = jls[0], jls[1], jls[2]

    t1 *= ell * (ell-1) / (2*ell-1) / (2*ell+1)
    t2 *= (2*ell**2-1) / (2*ell-1) / (2*ell+3)
    t3 *= (ell+1) * (ell+2) / (2*ell+1) / (2*ell+3)

    return t1 - t2 + t3


def tab_jls(fn, ells, ks, chis, verbose=True):
    '''tabulate j_ell(kchi) sample points into a hdf5 file'''
    f = h5py.File(fn, 'a')

    # write ks, chis & kchis
    if verbose:
        print('>> writing ks, chis, kchis and ells ...')
    _dset = f.create_dataset('ks', data=ks)
    _dset = f.create_dataset('chis', data=chis)
    kchis = np.einsum('i,j->ij', ks, chis)
    _dset = f.create_dataset('kchis', data=kchis)
    _dset = f.create_dataset('ells', data=ells)

    # jls
    for ell in ells:
        if verbose:
            print('>> ell = {:d}'.format(ell))
        jls = spherical_jn(ell, kchis)
        _dset = f.create_dataset('j_{:d}'.format(ell), data=jls)

    f.close()


def shot_noise(num, area, n_ells=None):
    '''Return the constant shot noise N_ell given 
       the area [deg^2] and total number counts of the survey.'''
    area_sr = (area / 41252.96) * 4 * np.pi
    sn = 1. / (num / area_sr)
    if n_ells is None:  # scalar
        return sn
    else:
        return np.full(n_ells, sn)


def cov_Gaussian(fsky, ells, cl13, cl24, cl14, cl23):
    '''Gaussian covariances between cl12 and cl34,
       cov(cl12, cl34) = (cl13 * cl24 + cl14 * cl23) / fsky / (2*l+1).
       The noises should be included in the cl's.'''

    return (cl13 * cl24 + cl14 * cl23) / fsky / (2 * ells + 1)
