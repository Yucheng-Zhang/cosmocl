'''
Functions for estimation with NaMaster.
'''
import pymaster as nmt
import numpy as np
import healpy as hp
import sys
import os


def write_cls(ell, cl, clerr, fn, fb):
    '''Write [ell, cl, xerr]s to file.'''
    bbs = np.loadtxt(fb, dtype='int32')
    xerr = (bbs[:, 1] - bbs[:, 0] + 1) / 2.
    data = np.column_stack((ell, cl, xerr, clerr))
    header = 'ell   cl   xerr   yerr'
    np.savetxt(fn, data, header=header)
    print(':: Written to: {}'.format(fn))


def ini_field(mask, maps, fwhm, temp):
    '''Initialize pymaster field.'''
    if fwhm > 0:
        print('>> Computing Gaussian beam window function, FWHM = {0:f} [radians]'
              .format(fwhm))
        lmax = 3 * hp.get_nside(mask) - 1  # required by PyMaster
        bl = hp.gauss_beam(fwhm, lmax=lmax, pol=False)
    else:
        bl = None
    print('>> Initializing the field...')
    fld = nmt.NmtField(mask, [maps], beam=bl, templates=temp)
    return fld


def ini_bin(nside, fb, sbpws=False):
    '''Initialize the set of bins.'''
    # load the file which includes bin bounds
    # two columns [lmin,lmax], both included
    print('>> Loading bin file: {}'.format(fb))
    bbs = np.loadtxt(fb, dtype='int32')
    ells = np.arange(bbs[0, 0], bbs[-1, -1] + 1, dtype='int32')
    weights = np.zeros(len(ells))
    bpws = -1 + np.zeros_like(ells)  # array of bandpower indices
    ib = 0
    for i, bb in enumerate(bbs):
        nls = bb[1] - bb[0] + 1  # number of ells in the bin
        ie = ib + nls  # not included
        weights[ib:ie] = 1. / nls
        bpws[ib:ie] = i
        ib = ie

    if sbpws:
        data = np.column_stack((ells, weights, bpws))
        header = 'ells   weights   bandpower'
        np.savetxt('bandpowers.dat', data, header=header)

    print('>> Initializing bins...')
    b = nmt.NmtBin(nside, bpws=bpws, ells=ells, weights=weights)
    return b


# def gaussian_err(w, fcl, focov):
#     '''Gaussian estimate of the covariance -> error.'''
#     print('>> Computing Gaussian estimate of the covariance matrix...')
#     # load the predicted Cl from theoretical calculation
#     cl_t = np.loadtxt(fcl)[:, -1]

#     cw = nmt.NmtCovarianceWorkspace()
#     cw.compute_coupling_coefficients(w, w)
#     covar = nmt.gaussian_covariance(cw, cl_t, cl_t, cl_t, cl_t)
#     err = np.sqrt(np.array([covar[i, i] for i in range(len(covar[0]))]))

#     # write covariance matrix to file
#     header = 'covariance matrix'
#     np.savetxt(focov, covar, fmt='%.7e', header=header)
#     print(':: covariance matrix written to: {}'.format(focov))

#     return err


def est_cl(w, fld1, fld2, b, me='step', ccl=None):
    '''Estimate Cl.'''

    if me == 'full':
        print('>> Computing full master...')
        cl = nmt.compute_full_master(fld1, fld2, b, workspace=w)
        cl_decoupled = cl[0]

    elif me == 'step':
        if ccl is None:
            # compute the coupled full-sky angular power spectra
            # this is equivalent to Healpy.anafast on masked maps
            print('>> Computing coupled Cl...')
            cl_coupled = nmt.compute_coupled_cell(fld1, fld2)
        else:
            cl_coupled = [ccl]
        # decouple into bandpowers by inverting the binned coupling matrix
        print('>> Decoupling Cl...')
        cl_decoupled = w.decouple_cell(cl_coupled)[0]

    else:
        sys.exit('>> Wrong me.')

    # get the effective ells
    print('>> Getting effective ells...')
    ell = b.get_effective_ells()

    return ell, cl_decoupled


def main_master(args):
    '''Main function for NaMaster estimation.'''
    print('>> MASTER estimation.')
    ccl = None  # coupled C_l

    print('>> Loading mask 1: {}'.format(args.mask1))
    mask1 = hp.read_map(args.mask1)
    fwhm1 = args.fwhm1
    if fwhm1 != -1:
        fwhm1 = fwhm1 * np.pi / 180  # get fwhm in radians
        print('>> Smoothing mask1, FWHM: {0:f} degrees'.format(args.fwhm1))
        mask1 = hp.smoothing(mask1, fwhm=fwhm1, pol=False)

    print('>> Loading map 1: {}'.format(args.map1))
    map1 = hp.read_map(args.map1)
    if args.temp1 != '':
        print('>> Loading template 1: {}'.format(args.temp1))
        temp1 = [[hp.read_map(args.temp1)]]
    else:
        temp1 = None
    field1 = ini_field(mask1, map1, fwhm1, temp1)

    if args.tp == 'cross':  # cross correlation
        print('>> Loading mask 2: {}'.format(args.mask2))
        mask2 = hp.read_map(args.mask2)
        fwhm2 = args.fwhm2
        if fwhm2 != -1:
            fwhm2 = args.fwhm2 * np.pi / 180  # get fwhm in radians
            print('>> Smoothing mask2, FWHM: {0:f} degrees'.format(args.fwhm2))
            mask2 = hp.smoothing(mask2, fwhm=fwhm2, pol=False)

        print('>> Loading map 2: {}'.format(args.map2))
        map2 = hp.read_map(args.map2)
        if args.temp2 != '':
            print('>> Loading template 2: {}'.format(args.temp2))
            temp2 = [[hp.read_map(args.temp2)]]
        else:
            temp2 = None
        field2 = ini_field(mask2, map2, fwhm2, temp2)
        if args.eccl[0] == '1':
            print(':: Coupled C_l without multiplying mask on the map ::')
            ccl = hp.anafast(map1, map2=map2)

    elif args.tp == 'auto':  # auto correlation
        field2 = field1
        if args.eccl[1] == '1':
            print(':: Coupled C_l without multiplying mask on the map ::')
            ccl = hp.anafast(map1)

    else:
        sys.exit('>> Wrong correlation type!')

    b = ini_bin(args.nside, args.fb)

    # NmtWorkspace object used to compute and store the mode coupling matrix,
    # which only depends on the masks, not on the maps
    w = nmt.NmtWorkspace()
    fwsp = args.fwsp
    if os.path.isfile(fwsp):
        print('>> Loading workspace (coupling matrix) from : {}'.format(fwsp))
        w.read_from(fwsp)
    else:
        print('>> Computing coupling matrix...')
        w.compute_coupling_matrix(field1, field2, b)
        if fwsp != '' and args.swsp:
            w.write_to(fwsp)
            print(':: Workspace saved to : {}'.format(fwsp))

    # estimate the power spectrum
    ell, cl = est_cl(w, field1, field2, b, ccl=ccl)

    # compute Gaussian covariance
    if args.fcl != '':
        # clerr = gaussian_err(w, args.fcl, args.focov)
        pass
    else:
        clerr = np.array([0.] * len(cl))

    # take care of constant shot noise
    if args.shotnoise != 0:
        cl = cl - args.shotnoise

    write_cls(ell, cl, clerr, args.focl, args.fb)
