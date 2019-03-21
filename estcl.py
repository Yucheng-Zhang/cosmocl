'''
Estimate Cl with PyMaster.
'''
import pymaster as nmt
import argparse
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys


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

#    parser.add_argument('-lmm', type=int, nargs='+',
#                        default=[2, 600], help='lmin and lmax')
    parser.add_argument('-nlb', type=int, default=1,
                        help='Number of ells per bin.')

    parser.add_argument('-foutcl', type=str,
                        default='out-cl.dat', help='Output ell,cl file name.')

    args = parser.parse_args()


def ini_field(mask, maps):
    '''Initialize pymaster field.'''
    fld = nmt.NmtField(mask, [maps])
    return fld


def ini_bin(nside, nlb):
    '''Initialize the set of bins.'''
    b = nmt.NmtBin(nside, nlb=nlb)
    return b


def est_cl(fld1, fld2, b):
    '''Estimate Cl.'''
    print('>> Estimating Cl...')

    # NmtWorkspace object used to compute and store the mode coupling matrix,
    # which only depends on the masks, not on the maps
    w = nmt.NmtWorkspace()
    print('>> Computing coupling matrix...')
    w.compute_coupling_matrix(fld1, fld2, b)

    # compute the coupled full-sky angular power spectra
    # this is equivalent to Healpy.anafast on masked maps
    print('>> Computing coupled Cl...')
    cl_coupled = nmt.compute_coupled_cell(fld1, fld2)

    # decouple into bandpowers by inverting the binned coupling matrix
    print('>> Decoupling Cl...')
    cl_decoupled = w.decouple_cell(cl_coupled)[0]

    # get the effective ells
    print('>> Getting ')
    ell = b.get_effective_ells()

    return ell, cl_decoupled


def write_cls(ell, cl, fn):
    '''Write [ell, cl]s to file.'''
    data = np.column_stack((ell, cl))
    header = 'ell   cl'
    np.savetxt(fn, data, header=header)
    print('>> Written to: {}'.format(fn))


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

    b = ini_bin(args.nside, args.nlb)

    ell, cl = est_cl(field1, field2, b)

    write_cls(ell, cl, args.foutcl)
