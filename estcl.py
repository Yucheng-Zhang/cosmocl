'''
Estimation of Cl.
'''
import pymaster as nmt
import numpy as np
import healpy as hp
import collections
import sys
import time


class estcl:
    '''Estimation of Cl, w/ PyMaster or Healpy.'''

    def __init__(self):

        # maps and masks
        self.fields = collections.OrderedDict()  # map & mask
        self.nside = None

        # bandpower settings
        self.bps = collections.OrderedDict()
        # bins, two columns [lmin, lmax], closed on both sides.
        self.bps['bins'] = None
        self.bps['ells'] = None
        self.bps['elle'] = None  # effective ell
        self.bps['lerr'] = None  # ell err
        self.bps['ibpws'] = None  # index of bandpower for each ell
        self.bps['weights'] = None  # weight of each cl over bin

        # power spectrum related
        self.cls = collections.OrderedDict()

    def read_map(self, nside, fields, fmaps, fmasks):
        '''Read maps and masks.'''
        self.nside = nside

        for i, field in enumerate(fields):
            print('>> Loading field: {0:s}'.format(field))
            self.fields[field] = collections.OrderedDict()
            print('> map {0:s}'.format(fmaps[i]))
            self.fields[field]['map'] = hp.read_map(fmaps[i])
            print('> mask {0:s}'.format(fmasks[i]))
            self.fields[field]['mask'] = hp.read_map(fmasks[i])
            # check nside
            if hp.get_nside(self.fields[field]['map']) != nside or \
                    hp.get_nside(self.fields[field]['mask']) != nside:
                sys.exit('!! exit: nside does not match !!')

    def ini_bins(self, fb, weight_m=0):
        '''Initialize bins.'''
        # read the bins, two columns [lmin, lmax], closed on both sides
        print('>> Loading bin file: {0:s}'.format(fb))
        self.bps['bins'] = np.loadtxt(fb, dtype=np.int)
        print('>> Bins initialized.')
        # generate the bandpower index and weight of each Cl
        ells = np.arange(self.bps['bins'][0, 0], self.bps['bins'][-1, -1] + 1,
                         dtype=np.int)
        self.bps['ells'] = ells
        self.bps['ibpws'] = -1 + np.zeros_like(ells, dtype=np.int)
        self.bps['weights'] = np.zeros_like(ells, dtype=np.float)

        # set the effective ell at the mean ell
        self.bps['elle'] = np.mean(self.bps['bins'], axis=1)
        self.bps['lerr'] = (self.bps['bins'][:, 1] -
                            self.bps['bins'][:, 0] + 1) / 2.

        for i, bb in enumerate(self.bps['bins']):
            idx = np.where((ells >= bb[0]) & (ells <= bb[1]))[0]
            self.bps['ibpws'][idx] = i
            if weight_m == 0:  # uniform
                w_, w_e = 1., 1.
            elif weight_m == 1:  # ell
                w_, w_e = ells[idx], self.bps['elle'][i]
            elif weight_m == 2:  # ell*(ell+1)
                w_ = ells[idx] * (1. + ells[idx])
                w_e = self.bps['elle'][i] * (self.bps['elle'][i] + 1)
            else:
                sys.exit('!! exit: wrong weight_m !!')

            self.bps['weights'][idx] = w_ / w_e / idx.shape[0]

    def write_bins(self, fo):
        '''Output the bandpower bins information.'''
        header = 'ell     bandpower     weight'
        data = np.column_stack((self.bps['ells'], self.bps['ibpws'],
                                self.bps['weights']))
        fmt = '%5d   %5d   %15.7e'
        np.savetxt(fo, data, header=header, fmt=fmt)
        print(':: Bins written to: {0:s}'.format(fo))

    def have_fld(self, fld):
        '''Check if fld is in self.fields.'''
        return fld in self.fields.keys()

    def have_cl(self, cl):
        '''Check if Cl is in self.cls.'''
        return cl in self.cls.keys()

    def est_nmt(self, f1, f2, cl_label, re=False):
        '''Estimate w/ PyMaster, f1 & f2(f1) cross(auto) Cl.'''
        print('>> Estimating {0:s} with PyMaster...'.format(cl_label))
        t0 = time.time()
        # check f1 & f2
        if not self.have_fld(f1) or not self.have_fld(f2):
            sys.exit('!! exit: no such field !!')

        b = nmt.NmtBin(self.nside, bpws=self.bps['ibpws'],
                       ells=self.bps['ells'], weights=self.bps['weights'])

        fld1 = nmt.NmtField(self.fields[f1]['mask'],
                            [self.fields[f1]['map']])
        if f2 == f1:  # auto
            fld2 = fld1
        else:  # cross
            fld2 = nmt.NmtField(self.fields[f2]['mask'],
                                [self.fields[f2]['map']])

        w = nmt.NmtWorkspace()
        print('> computing coupling matrix')
        w.compute_coupling_matrix(fld1, fld2, b)
        print('> computing coupled cl')
        cl_coupled = nmt.compute_coupled_cell(fld1, fld2)
        print('> decoupling cl')
        cl_decoupled = w.decouple_cell(cl_coupled)[0]

        self.cls[cl_label] = cl_decoupled

        print('<< time elapsed: {0:.2f} s'.format(time.time()-t0))

        if re:  # return the Cl
            return cl_decoupled

    def est_hp(self, f1, f2, cl_label):
        '''Estimate w/ Healpy, f1 & f2(f1) cross(auto) Cl.'''
        pass

    def get_cl(self, cl_label):
        '''Return the Cl.'''
        if not self.have_cl(cl_label):
            sys.exit('!! exit: no such cl : {0:s}'.format(cl_label))
        return self.cls[cl_label]

    def get_eff_ells(self):
        '''Return the effective ells.'''
        return self.bps['elle']

    def get_ell_width(self):
        '''Return the bin width.'''
        return self.bps['lerr']

    def write_cl(self, fo, cl_label):
        '''Write Cls to file.'''
        if not self.have_cl(cl_label):
            sys.exit('!! exit: no such cl : {0:s}'.format(cl_label))

        header = 'ell     {0:s}     xerr'.format(cl_label)
        fmt = '%10g   %23.15e   %10g'
        data = np.column_stack((self.bps['elle'], self.cls[cl_label],
                                self.bps['lerr']))
        np.savetxt(fo, data, header=header, fmt=fmt)
