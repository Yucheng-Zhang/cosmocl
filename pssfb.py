'''
Theoretical computation of power spectra in SFB basis.
'''
import numpy as np
import h5py

from .utils import jl_2nd
from .constants import *


def W_k(cosmo, r, z):
    '''CMB lensing kernel, W_kappa(r).'''
    r_CMB = cosmo.chi_CMB
    fac = 3./2. * cosmo.Om0 * (cosmo.H0 / C_LIGHT)**2
    return fac * (1 + z) * r * (1 - r / r_CMB) * cosmo.D(z, interp=True)


class gfield:
    '''Galaxy field.'''

    def __init__(self, fg, bg, zmin, zmax):
        self.fg = fg
        self.bg = bg
        self.zmin, self.zmax = zmin, zmax

    def W_g(self, cosmo, z):
        '''Galaxy redshift kernel, W_g(r(z)).'''
        return cosmo.H(z)/C_LIGHT * self.fg(z) * self.bg(z) * cosmo.D(z, interp=True)


class cc:

    def __init__(self, cosmo, fg=None, bg=None, zg1=None, zg2=None):

        # ------ cosmology ------
        self.cosmo = cosmo
        self.chi_CMB = cosmo.chi(Z_CMB)


class cps_sfb:
    '''Theoretical computation of power spectra in SFB basis.'''

    def __init__(self):
        pass
