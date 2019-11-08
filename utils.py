'''
Some useful functions.
'''
import numpy as np
import healpy as hp
from cosmoknife.utils import get_ra_dec


def gen_fb(lmin, lmax, nb, fo, scale=None, re=False):
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

    if re:
        return data


def bin_cl(cl, ell, bins, weight='uniform'):
    '''Bin cl (one dim. array).'''
    nb = bins.shape[0]
    ell_b = np.mean(bins, axis=1)
    ell_err = (bins[:, 1] - bins[:, 0] + 1) / 2.
    cl_b = np.zeros(nb)
    for i, be in enumerate(bins):
        idx = np.where((ell >= be[0]) & (ell <= be[1]))[0]
        if weight == 'uniform':
            cl_b[i] = np.mean(cl[idx])

    return cl_b, ell_b, ell_err


def make_overdensity_map(nside, theta, phi, weight, mask, w_lim=0.9, powspec=False):
    '''Make overdensity map for point sources.'''
    npix = hp.nside2npix(nside)
    ppix = hp.ang2pix(nside, theta, phi)  # pixel indices of the targets
    mpix = np.where(mask >= w_lim)[0]  # pixels to use

    # get total map
    total_map = np.zeros(npix)
    pidx = np.isin(ppix, mpix)
    np.add.at(total_map, ppix[pidx], weight[pidx])

    # n-bar per pixel
    nbar = np.mean(total_map[mpix] / mask[mpix])

    # n-bar per steradians (for shot noise)
    area_pix = 4 * np.pi / npix
    nbar_st = nbar / area_pix

    # get overdensity signal
    signal = np.zeros(npix)
    signal[mpix] = total_map[mpix] / mask[mpix] / nbar - 1.

    # new mask
    # mask_n = np.zeros(npix)
    # mask_n[mpix] = 1.

    if powspec:  # make data file for PowSpec code
        theta_pix, phi_pix = hp.pix2ang(nside, np.arange(npix))
        ra, dec = get_ra_dec(theta_pix[mpix], phi_pix[mpix])
        area = np.full(mpix.shape[0], area_pix)
        noise = np.full(mpix.shape[0], 1./nbar)
        pow_data = np.column_stack((ra, dec, area, signal[mpix], noise))
        return signal, pow_data
    else:
        return signal
