'''
Models for different fields.
'''
import numpy as np
from .constants import *


def W_k(cosmo, r, z):
    '''CMB lensing kernel, W_kappa(r).'''
    r_CMB = cosmo.chi_CMB
    fac = 3./2. * cosmo.Om0 * (cosmo.H0 / C_LIGHT)**2
    return fac * (1 + z) * r * (1 - r / r_CMB) * cosmo.D(z, interp=True)


class galaxy:
    '''Galaxy field.'''

    def __init__(self, bg, fg, phig, zmin, zmax):

        self.bg = bg  # bias, b_g(z)
        self.fg = fg  # normalized redshift distribution, f_g(z)
        self.phig = phig  # 3D radial selection function, phi_g(r)
        self.zmin, self.zmax = zmin, zmax

    def W_g_2D(self, cosmo, r, z):
        '''2D redshift kernel.'''
        return cosmo.H(z)/C_LIGHT * self.fg(z) * self.bg(z) * cosmo.D(z, interp=True)

    def W_g_3D(self, cosmo, r, z):
        '''3D redshift kernel.'''
        return r**2 * self.phig(r) * self.bg(z) * cosmo.D(z, interp=True)

    def Delta_b_PNG(self, cosmo, z, k, arr=False):
        '''Scale-dependent bias correction due to PNG.'''
        fac = 3 * cosmo.Om0 * DELTA_C * (cosmo.H0 / C_LIGHT)**2
        fz = (1 - 1 / self.bg(z)) / cosmo.D_unnorm(z, interp=True)
        fk = 1 / (k**2 * cosmo.Tk(k))
        if arr:
            return fac * np.einsum('k,z->kz', fk, fz)
        else:
            return fac * fk * fz
