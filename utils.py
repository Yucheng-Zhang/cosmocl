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


def grp2dic(grp):
    '''Convert hdf5 group (with integer key) into python dictionary.'''
    dic = {}
    for (key, value) in grp.items():
        dic[int(key)] = value[:]

    return dic


def tab_j_ell(fn, ells, ks, rs, verbose=True):
    '''Tabulate j_ell at (k, r) sample points into a hdf5 file.'''
    f = h5py.File(fn, 'a')

    # write ks, rs & krs
    _dset = f.create_dataset('ells', data=ells)
    _dset = f.create_dataset('ks', data=ks)
    _dset = f.create_dataset('rs', data=rs)
    krs = np.einsum('k,r->kr', ks, rs)
    _dset = f.create_dataset('krs', data=krs)

    # spherical Bessels
    jl_grp = f.create_group('j')
    for ell in ells:
        if verbose:
            print('>> ell = {:d}'.format(ell))
        j_ells = spherical_jn(ell, krs)
        _dset = jl_grp.create_dataset('{:d}'.format(ell), data=j_ells)

    f.close()


def tab_J_ell(fn, ells, k_lns, A_lns, rs, verbose=True):
    '''Tabulate J_ell (shell radial eigenfunction) at (k_ln, r) sample points into a hdf5 file.'''
    f = h5py.File(fn, 'a')

    _dset = f.create_dataset('ells', data=ells)
    _dset = f.create_dataset('rs', data=rs)

    Jl_grp = f.create_group('J')
    for ell in ells:
        if verbose:
            print('>> ell = {:d}'.format(ell))
        _dset = f.create_dataset('k_{:d}ns'.format(ell), data=k_lns[ell])
        k_lnrs = np.einsum('k,r->kr', k_lns[ell], rs)
        _dset = f.create_dataset('k_{:d}nrs'.format(ell), data=k_lnrs)
        j_ells = spherical_jn(ell, k_lnrs)
        y_ells = spherical_yn(ell, k_lnrs)
        J_ells = j_ells + np.einsum('k,kr->kr', A_lns[ell], y_ells)
        _dset = Jl_grp.create_dataset('{:d}'.format(ell), data=J_ells)

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
