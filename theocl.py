'''
Theoretical calculation of Cl.
'''
import numpy as np
import scipy.integrate as spint
import argparse
import sys
import multiprocessing as mp
from joblib import Parallel, delayed

from cosmopy.cosmoLCDM import cosmoLCDM
from cosmopy.utils import gen_fg_z

C_LIGHT = 299792.458  # speed of light in km/s


class theocls:
    '''Theoretical power spectra, w/ Limber approximation.'''

    def __init__(self, lmin, lmax, z1, z2, fg, b, cosmo):
        self.lmin, self.lmax = lmin, lmax
        self.z1, self.z2 = z1, z2
        self.fg = fg  # redshift distribution function
        self.b = b  # linear bias function
        self.cosmo = cosmo  # cosmoLCDM instance

        kmax = (self.lmax + 100) / self.cosmo.z2chi(self.z1)
        self.cosmo.gen_pk(kmax, self.z1, self.z2)

        self.ells = np.arange(lmin, lmax+1, 1, dtype='int32')

        self.num_cpus = mp.cpu_count()

    def get_ells(self):
        return self.ells

    # ------ power spectra ------ #

    def c_clkg(self):
        '''Compute C_l^kg.'''
        def clkg_kernel(z, ell):
            chi_z = self.cosmo.z2chi(z)
            p1 = (1 + z) * self.cosmo.w_z(z) * self.fg(z) / chi_z**2
            p2 = self.cosmo.pk.P(z, ell/chi_z) * self.b(z)
            return p1*p2

        def target(ell):
            return spint.quad(clkg_kernel, self.z1, self.z2, args=(ell,), full_output=1)[0]

        print('>> Computing C_l^kg...')
        clkgs = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        fac = 3. * self.cosmo.H0**2 * self.cosmo.Om0 / (2. * C_LIGHT**2)
        clkgs = np.array(clkgs) * fac

        return clkgs

    def c_clgg(self):
        '''Compute C_l^gg.'''
        def clgg_kernel(z, ell):
            chi_z = self.cosmo.z2chi(z)
            p1 = self.cosmo.H_z(z) * self.fg(z)**2 / chi_z**2
            p2 = self.cosmo.pk.P(z, ell/chi_z) * self.b(z)**2
            return p1*p2

        def target(ell):
            return spint.quad(clgg_kernel, self.z1, self.z2, args=(ell,), full_output=1)[0]

        clggs = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        clggs = np.array(clggs) / C_LIGHT

        return clggs
