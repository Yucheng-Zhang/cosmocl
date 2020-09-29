'''
Theoretical computation of power spectra in SFB basis.
'''
import numpy as np
from scipy.special import spherical_jn
import h5py

from .utils import jl_2nd
from .constants import *


class pssfb:
    '''Theoretical computation of power spectra in SFB basis.'''

    def __init__(self, cosmo, fn_jls, fn_Jls=None):

        self.cosmo = cosmo
        self.gflds = {}  # galaxy fields

        # ------ k, r & j_ell sample points ------
        self.fn_jls = fn_jls
        self.fss = h5py.File(fn_jls, 'r')
        self.ks = self.fss['ks'][:]
        self.rs = self.fss['chis'][:]
        self.krs = self.fss['krs'][:]  # outer product of ks and chis
        self.ell_ss = self.fss['ells'][:]
        # z sample points corresponding to rs
        self.zs = self.cosmo.z_at_chi(self.rs)

    def add_galaxy_field(self, name, dim=):
        '''Add a galaxy field to the class.
           dim: dimension, 2 or 3.'''
        self.gflds[name] = {}

    def get_Jls(self, ell):
        '''Get '''

    def get_jls(self, ell, der=0):
        '''Get j_ell samples from data file or spherical_jn.'''
        if der == 0:
            if ell in self.ell_ss:
                return self.fss['j_{:d}'.format(ell)][:]
            else:
                return spherical_jn(ell, self.krs)

        if der == 2:
            jls_ = {}
            for i, ell_ in enumerate([ell-2, ell, ell+2]):
                if ell_ in self.ell_ss:
                    jls_[i] = self.fss['j_{:d}'.format(ell_)][:]
                else:
                    jls_[i] = spherical_jn(ell_, self.krs)
            return jl_2nd(ell, jls=jls_)

    def c_Delta(self, ell, W, dim, rmin=None, rmax=None):
        '''Compute transfer function for a field.
           W: redshift kernel;
           dim: dimension, 2D or 3D.'''
        if dim == 2:
            pass
        elif dim == 3:
            pass
        else:
            pass

    def c_ps(self):
        '''Compute the '''
