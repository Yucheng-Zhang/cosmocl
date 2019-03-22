'''
Estimate Cl with PyMaster.
'''
import pymaster as nmt
import argparse
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Estimate Cl with PyMaster.')

    parser.add_argument('-mask1', type=str, default='',
                        help='Input mask file 1.')
    parser.add_argument('-map1', type=str, default='',
                        help='Input map file 1.')
    parser.add_argument('-mask2', type=str, default='',
                        help='Input mask file 2.')
    parser.add_argument('-map2', type=str, default='',
                        help='Input map file 2.')
    parser.add_argument('-nside', type=int, default=2048,
                        help='Nside of the masks and maps.')

    parser.add_argument('-tp', type=str, default='auto',
                        help='auto or cross correlation.')

    parser.add_argument('-foutcl', type=str, default='out-cl.dat',
                        help='Output ell,cl file name.')

    parser.add_argument('-fb', type=str, default='fb.dat',
                        help='Data file which specifies the bins. \
                            Two columns [lmin, lmax], closed on both sides.')

    parser.add_argument('-savewsp', type=int, default=0,
                        help='Save the workspace or not, which might be very large.')
    parser.add_argument('-fwsp', type=str, default='',
                        help='Workspace file name.')

    args = parser.parse_args()


def ini_field(mask, maps):
    '''Initialize pymaster field.'''
    print('>> Initializing the field...')
    fld = nmt.NmtField(mask, [maps])
    return fld


def ini_bin(nside, fb):
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

    data = np.column_stack((ells, weights, bpws))
    header = 'ells   weights   bandpower'
    np.savetxt('bandpowers.dat', data, header=header)

    print('>> Initializing bins...')
    b = nmt.NmtBin(nside, bpws=bpws, ells=ells, weights=weights)
    return b


def est_cl(fld1, fld2, b, fwsp, swsp, me='full'):
    '''Estimate Cl.'''
    # NmtWorkspace object used to compute and store the mode coupling matrix,
    # which only depends on the masks, not on the maps
    w = nmt.NmtWorkspace()
    if os.path.isfile(fwsp):
        print('>> Loading workspace (coupling matrix) from : {}'.format(fwsp))
        w.read_from(fwsp)
    else:
        print('>> Computing coupling matrix...')
        w.compute_coupling_matrix(fld1, fld2, b)
        if fwsp != '' and swsp:
            w.write_to(fwsp)
            print(':: Workspace saved to : {}'.format(fwsp))

    if me == 'full':
        print('>> Computing full master...')
        cl = nmt.compute_full_master(fld1, fld2, b)
        cl_decoupled = cl[0]
    elif me == 'step':
        # compute the coupled full-sky angular power spectra
        # this is equivalent to Healpy.anafast on masked maps
        print('>> Computing coupled Cl...')
        cl_coupled = nmt.compute_coupled_cell(fld1, fld2)
        # decouple into bandpowers by inverting the binned coupling matrix
        print('>> Decoupling Cl...')
        cl_decoupled = w.decouple_cell(cl_coupled)[0]
    else:
        sys.exit('>> Wrong me.')

    # get the effective ells
    print('>> Getting effective ells...')
    ell = b.get_effective_ells()

    return ell, cl_decoupled


def write_cls(ell, cl, fn, fb):
    '''Write [ell, cl, xerr]s to file.'''
    bbs = np.loadtxt(fb, dtype='int32')
    xerr = (bbs[:, 1] - bbs[:, 0] + 1) / 2.
    data = np.column_stack((ell, cl, xerr))
    header = 'ell   cl   xerr'
    np.savetxt(fn, data, header=header)
    print(':: Written to: {}'.format(fn))


if __name__ == "__main__":
    '''Main function.'''
    print('>> Loading mask 1: {}'.format(args.mask1))
    mask1 = hp.read_map(args.mask1)
    print('>> Loading map 1: {}'.format(args.map1))
    map1 = hp.read_map(args.map1)
    field1 = ini_field(mask1, map1)

    if args.tp == 'cross':  # cross correlation
        print('>> Loading mask 2: {}'.format(args.mask2))
        mask2 = hp.read_map(args.mask2)
        print('>> Loading map 2: {}'.format(args.map2))
        map2 = hp.read_map(args.map2)
        field2 = ini_field(mask2, map2)
    elif args.tp == 'auto':  # auto correlation
        field2 = field1
    else:
        sys.exit('>> Wrong correlation type!')

    b = ini_bin(args.nside, args.fb)

    ell, cl = est_cl(field1, field2, b, args.fwsp, args.swsp)

    write_cls(ell, cl, args.foutcl, args.fb)
