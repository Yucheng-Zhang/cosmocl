'''
Some useful functions.
'''
import numpy as np
import h5py
from scipy.special import spherical_jn, spherical_yn


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


def tab_spherical_Bessel(fn, kind, ells, ks, rs, As=None, verbose=True):
    '''tabulate spherical Bessel function sample points into a hdf5 file.
       kind = 1, 2 or 3 for 1st, 2nd kind or both.
       As: the factor for linear combination of j_ell and y_ell (for shell limit).'''
    f = h5py.File(fn, 'a')

    # write ks, rs & krs
    if verbose:
        print('>> writing ks, rs, krs and ells ...')
    _dset = f.create_dataset('ks', data=ks)
    _dset = f.create_dataset('rs', data=rs)
    krs = np.einsum('i,j->ij', ks, rs)
    _dset = f.create_dataset('krs', data=krs)
    _dset = f.create_dataset('ells', data=ells)
    if As is not None:
        _dset = f.create_dataset('As', data=As)

    # spherical Bessels
    ss = '>> tabulating:'
    tjl, tyl = False, False
    if kind in [1, 3]:
        ss += ' j_ell'
        tjl = True
    if kind in [2, 3]:
        ss += ' y_ell'
        tyl = True
    print(ss)
    for ell in ells:
        if verbose:
            print('>> ell = {:d}'.format(ell))
        if tjl:
            fls = spherical_jn(ell, krs)
            _dset = f.create_dataset('j_{:d}'.format(ell), data=fls)
        if tyl:
            fls = spherical_yn(ell, krs)
            _dset = f.create_dataset('y_{:d}'.format(ell), data=fls)

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


def cov_Gaussian(fsky, ell, cl13, cl24, cl14, cl23):
    '''Gaussian covariances between cl12 and cl34,
       cov(cl12, cl34) = (cl13 * cl24 + cl14 * cl23) / fsky / (2*l+1).
       The noises should be included in the cl's.'''

    return (cl13 * cl24 + cl14 * cl23) / fsky / (2 * ell + 1)


def limber_lmin(cosmo, z1, z2):
    '''Estimate minimum ell for Limber approximation with chi/Delta(chi).'''
    chi = cosmo.chi(max(z1, z2))
    Delta_chi = abs(cosmo.chi(z2) - cosmo.chi(z1))
    return chi / Delta_chi - 0.5
