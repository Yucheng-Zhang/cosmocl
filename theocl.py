'''
Theoretical calculation of Cl.
'''
import numpy as np
import scipy.integrate as spint
import multiprocessing as mp
from joblib import Parallel, delayed
import sys

from cosmopy.cosmoLCDM import cosmoLCDM
from cosmopy.utils import gen_fg_z

C_LIGHT = 299792.458  # speed of light in km/s
Z_CMB = 1100


class theocls:
    '''Theoretical linear power spectra, w/ Limber approximation.'''

    def __init__(self, ells, z1, z2, fg, b, cosmo):
        self.lmin, self.lmax = np.amin(ells), np.amax(ells)
        self.z1, self.z2 = z1, z2
        self.fg = fg  # redshift distribution function
        self.b = b  # linear bias function
        self.cosmo = cosmo  # cosmoLCDM instance

        self.ells = ells

        self.num_cpus = mp.cpu_count()
        print('>> Number of CPUs: {0:d}'.format(self.num_cpus))

    def set_pk(self, zmin, zmax, kmax, extrap_kmax=None):
        self.cosmo.gen_pk(zmin, zmax, kmax, extrap_kmax=extrap_kmax)
        print(':: PK interpolator settings ::')
        print(':: zmin = {0:g}, zmax = {1:g}'.format(zmin, zmax))
        print(':: kmax = {0:g}, extrap_kmax = {1:g}'.format(kmax, extrap_kmax))
        print(':: (for points out of range, the returned value is the boundary value)')
        print(':: NOTE: integral over z from 0 will start from chi2z(ell/extrap_kmax)')

        if extrap_kmax is None:
            self.kmax = kmax
        else:
            self.kmax = extrap_kmax

    def set_chi2z(self, zmin, zmax, dz):
        check = np.amax(self.ells) / self.kmax / self.cosmo.z2chi(zmax)
        print('-- check = {0:.2f}'.format(check))
        if check > 1:
            sys.exit('-- set_chi2z: zmax not large enough')

        self.cosmo.gen_chi2z(zmin=zmin, zmax=zmax, dz=dz)

    def get_ells(self):
        return self.ells

    # ------ power spectra ------ #

    def c_clkg(self):
        '''Compute C_l^kg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.z2chi(z)
            p1 = (1 + z) * self.cosmo.w_z(z) * self.fg(z) / chi_z**2
            p2 = self.cosmo.pk.P(z, ell/chi_z) * self.b(z)
            return p1*p2

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,), full_output=1)[0]

        print('>> Computing C_l^kg...')
        clkgs = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        fac = 3. * self.cosmo.H0**2 * self.cosmo.Om0 / (2. * C_LIGHT**2)
        clkgs = np.array(clkgs) * fac

        return clkgs

    def c_clgg(self):
        '''Compute C_l^gg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.z2chi(z)
            p1 = self.cosmo.H_z(z) * self.fg(z)**2 / chi_z**2
            p2 = self.cosmo.pk.P(z, ell/chi_z) * self.b(z)**2
            return p1*p2

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,), full_output=1)[0]

        print('>> Computing C_l^gg...')
        clggs = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        clggs = np.array(clggs) / C_LIGHT

        return clggs

    # ------ C_Gamma calibration ------ #

    def c_clmg(self):
        '''Compute C_l^mg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.z2chi(z)
            p1 = self.cosmo.H_z(z) * self.fg(z)**2 / chi_z**2
            p2 = self.cosmo.pk.P(z, ell/chi_z) * self.b(z)
            return p1*p2

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,), full_output=1)[0]

        print('>> Computing C_l^mg...')
        clmgs = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        clmgs = np.array(clmgs) / C_LIGHT

        return clmgs

    def c_qlgg(self):
        '''Compute Q_l^gg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.z2chi(z)
            p1 = (1 + z) * self.cosmo.w_z(z) * self.fg(z) / chi_z**2
            p2 = self.cosmo.pk.P(z, ell/chi_z) * self.b(z)**2
            return p1*p2

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,), full_output=1)[0]

        print('>> Computing Q_l^gg...')
        qlggs = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        qlggs = np.array(qlggs) / 2.

        return qlggs

    # ------ Magnification bias calibration ------ #

    def c_clg1g2(self, s):
        '''Compute C_l^g1g2.'''
        def kernel(z2, z1, ell):
            chi_z2 = self.cosmo.z2chi(z2)
            p1 = (1 + z2) * self.cosmo.w_z(z2, zs=z1) * \
                self.fg(z1) * self.fg(z2) / chi_z2**2
            p2 = self.cosmo.pk.P(z2, ell/chi_z2) * self.b(z2)
            return p1*p2

        def target(ell):
            return spint.dblquad(kernel, self.z1, self.z2, lambda x: self.z1, lambda x: x, args=(ell,))[0]

        print('>> Computing C_l^g1g2...')
        clg1g2s = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        fac = 3. * self.cosmo.H0**2 * self.cosmo.Om0 / C_LIGHT**2
        fac = fac * (5./2. * s - 1)

        clg1g2s = np.array(clg1g2s) * fac

        return clg1g2s

    def c_clg2g2(self, s):
        '''Compute C_l^g2g2.'''
        def kernel(z3, z2, z1, ell):
            chi_z3 = self.cosmo.z2chi(z3)
            p1 = self.fg(z1) * self.fg(z2) * (1 + z3)**2 * self.cosmo.w_z(z3, zs=z1) * \
                self.cosmo.w_z(z3, zs=z2) / self.cosmo.H_z(z3) / chi_z3**2
            p2 = self.cosmo.pk.P(z3, ell/chi_z3)
            return p1*p2

        def target(ell):
            iz = self.cosmo.chi2z(ell/self.kmax)
            return spint.tplquad(kernel, self.z1, self.z2, lambda x: self.z1, lambda x: self.z2,
                                 lambda x, y: iz, lambda x, y: min(x, y), args=(ell,))[0]

        print('>> Computing C_l^g2g2...')
        clg2g2s = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        fac = (3 * self.cosmo.H0**2 * self.cosmo.Om0 /
               C_LIGHT**2)**2 * (5./2. * s - 1)**2 * C_LIGHT

        clg2g2s = np.array(clg2g2s) * fac

        return clg2g2s

    def c_clkg2(self, s):
        '''Compute C_l^kg2.'''
        def kernel(z2, z1, ell):
            chi_z2 = self.cosmo.z2chi(z2)
            p1 = (1 + z2)**2 * self.cosmo.w_z(z2, zs=z1) * self.cosmo.w_z(z2) * \
                self.fg(z1) / self.cosmo.H_z(z2) / chi_z2**2
            p2 = self.cosmo.pk.P(z2, ell/chi_z2)
            return p1*p2

        def target(ell):
            iz = self.cosmo.chi2z(ell/self.kmax)
            return spint.dblquad(kernel, self.z1, self.z2, lambda x: iz, lambda x: x, args=(ell,))[0]

        print('>> Computing C_l^kg2...')
        clkg2s = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        fac = 1./2. * (3 * self.cosmo.H0**2 * self.cosmo.Om0 /
                       C_LIGHT**2)**2 * (5./2. * s - 1) * C_LIGHT
        clkg2s = np.array(clkg2s) * fac

        return clkg2s

    def c_clkk(self):
        '''Compute C_l^kk.'''
        def kernel(z, ell):
            chi_z = self.cosmo.z2chi(z)
            w_z = self.cosmo.w_z(z)
            p1 = (1 + z)**2 * w_z**2 / self.cosmo.H_z(z) / chi_z**2
            p2 = self.cosmo.pk.P(z, ell/chi_z)
            return p1*p2

        def target(ell):
            iz = self.cosmo.chi2z(ell/self.kmax)
            return spint.quad(kernel, iz, Z_CMB, args=(ell,), full_output=1)[0]

        print('>> Computing C_l^kk...')
        clkks = Parallel(n_jobs=self.num_cpus)(
            delayed(target)(ell) for ell in self.ells)

        fac = 1./4. * (3 * self.cosmo.H0**2 *
                       self.cosmo.Om0 / C_LIGHT**2)**2 * C_LIGHT
        clkks = np.array(clkks) * fac

        return clkks
