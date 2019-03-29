'''
Functions for simple estimation with Healpy.
'''
import pymaster as nmt
import numpy as np
import healpy as hp
import sys
import os


def bin_cl(cl, bbs):
    '''Bin cl with bin file.'''
    ell_b, cl_b = [], []
    for bb in bbs:
        ell_b.append((bb[0] + bb[1]) / 2.)
        cl_b.append(np.mean(cl[bb[0]:bb[1]+1]))
    xerr = (bbs[:, 1] - bbs[:, 0] + 1) / 2.

    data = np.column_stack((ell_b, cl_b, xerr))

    return data


def main_hp(args):
    '''Main function for simple Healpy estimation.'''
    bbs = np.loadtxt(args.fb, dtype='int32')
    lmax = bbs[-1, -1]

    print('>> Loading mask 1: {}'.format(args.mask1))
    mask1 = hp.read_map(args.mask1)
    # if args.fwhm1 != -1:
    #     fwhm1 = args.fwhm1 * np.pi / 180  # get fwhm in radians
    #     print('>> Smoothing mask1, FWHM: {0:f} degrees'.format(args.fwhm1))
    #     mask1 = hp.smoothing(mask1, fwhm=fwhm1, pol=False)

    print('>> Loading map 1: {}'.format(args.map1))
    map1 = hp.read_map(args.map1) * mask1
    alm1 = hp.map2alm(map1, lmax=lmax, pol=False)

    if args.tp == 'cross':  # cross correlation
        print('>> Loading mask 2: {}'.format(args.mask2))
        mask2 = hp.read_map(args.mask2)
        # if args.fwhm2 != -1:
        #     fwhm2 = args.fwhm2 * np.pi / 180  # get fwhm in radians
        #     print('>> Smoothing mask2, FWHM: {0:f} degrees'.format(args.fwhm2))
        #     mask2 = hp.smoothing(mask2, fwhm=fwhm2, pol=False)

        print('>> Loading map 2: {}'.format(args.map2))
        map2 = hp.read_map(args.map2) * mask2
        alm2 = hp.map2alm(map2, lmax=lmax, pol=False)

    elif args.tp == 'auto':
        alm2 = None

    else:
        sys.exit('>> Wrong correlation type!')

    cl = hp.alm2cl(alm1, alms2=alm2)
    cl = cl / args.fsky

    data = bin_cl(cl, bbs)
    header = 'ell   cl   xerr'
    fn = args.focl
    np.savetxt(fn, data, header=header)
    print(':: Written to: {}'.format(fn))
