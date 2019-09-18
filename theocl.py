'''
Theoretical calculation of Cl.
'''
import numpy as np
import scipy.integrate as spint
import multiprocessing as mp
from joblib import Parallel, delayed
import sys
import time

from cosmopy.cosmoLCDM import cosmoLCDM
from cosmopy.utils import gen_fg_z

C_LIGHT = 299792.458  # speed of light in km/s
Z_CMB = 1100
DELTA_C = 1.686


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

    def set_interp_pk(self, zmin, zmax, kmax, extrap_kmax=None):
        self.cosmo.gen_interp_pk(zmin, zmax, kmax, extrap_kmax=extrap_kmax)
        print(':: PK interpolator settings ::')
        print(':: zmin = {0:g}, zmax = {1:g}'.format(zmin, zmax))
        print(':: kmax = {0:g}'.format(kmax))
        if extrap_kmax is not None:
            print(':: extrap_kmax = {0:g}'.format(extrap_kmax))
        print(
            ':: (note: for points out of range, the returned value is the boundary value)')

        if extrap_kmax is None:
            self.kmax = kmax
        else:
            self.kmax = extrap_kmax

        print('-- checking integral over z from zero')
        print('- ell = {0:g} : z_zero = {1:g}'.format(
            self.ells[0], self.cosmo.interp_chi2z(self.ells[0] / self.kmax)))
        print('- ell = {0:g} : z_zero = {1:g}'.format(
            self.ells[-1], self.cosmo.interp_chi2z(self.ells[-1] / self.kmax)))

    def get_ells(self):
        return self.ells

    # ------ power spectra ------ #

    def c_clkg(self):
        '''Compute C_l^kg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            p1 = (1 + z) * self.cosmo.interp_w_z(z) * \
                self.fg(z) * self.b(z) / chi_z**2
            return p1 * self.cosmo.interp_pk.P(z, ell/chi_z)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-4)[0]

        print('>> Computing C_l^kg...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 3. * self.cosmo.H0**2 * self.cosmo.Om0 / (2. * C_LIGHT**2)
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clgg(self):
        '''Compute C_l^gg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            p1 = self.cosmo.H_z(z) * self.fg(z)**2 * self.b(z)**2 / chi_z**2
            return p1 * self.cosmo.interp_pk.P(z, ell/chi_z)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-3)[0]

        print('>> Computing C_l^gg...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        cl = np.array(cl) / C_LIGHT

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    # ------ C_Gamma calibration ------ #

    def c_clm(self):
        '''Compute C_l^m.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = self.cosmo.H_z(z) * self.fg(z)**2 / chi_z**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-3)[0]

        print('>> Computing C_l^m...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        cl = np.array(cl) / C_LIGHT

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clmg(self):
        '''Compute C_l^mg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = self.cosmo.H_z(z) * self.b(z) * self.fg(z)**2 / chi_z**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-6)[0]

        print('>> Computing C_l^mg...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        cl = np.array(cl) / C_LIGHT

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_qlm(self):
        '''Compute Q_l^m.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = (1 + z) * self.cosmo.interp_w_z(z) * \
                self.fg(z) / chi_z**2  # * self.cosmo.f_growth_z(z)
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-4)[0]

        print('>> Computing Q_l^m...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        cl = np.array(cl) / 2.

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_qlmg(self):
        '''Compute Q_l^mg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = (1 + z) * self.cosmo.interp_w_z(z) * \
                self.fg(z) * self.b(z) / chi_z**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-4)[0]

        print('>> Computing Q_l^mg...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        cl = np.array(cl) / 2.

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_qlgg(self):
        '''Compute Q_l^gg.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = (1 + z) * self.cosmo.interp_w_z(z) * \
                self.fg(z) * self.b(z)**2 / chi_z**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-4)[0]

        print('>> Computing Q_l^gg...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        cl = np.array(cl) / 2.

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    # ------ Magnification bias calibration ------ #

    def c_clg1g2(self, s):
        '''Compute C_l^g1g2.'''
        def kernel(z2, z1, ell):
            chi_z2 = self.cosmo.interp_z2chi(z2)
            p1 = (1 + z2) * self.cosmo.interp_w_z(z2, zs=z1) * \
                self.fg(z1) * self.fg(z2) * self.b(z2) / chi_z2**2
            return p1 * self.cosmo.interp_pk.P(z2, ell/chi_z2)

        def target(ell):
            return spint.dblquad(kernel, self.z1, self.z2, lambda x: self.z1, lambda x: x,
                                 args=(ell,), epsabs=0, epsrel=1e-3)[0]

        print('>> Computing C_l^g1g2...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 3. * self.cosmo.H0**2 * self.cosmo.Om0 / C_LIGHT**2
        fac = fac * (5./2. * s - 1)

        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clg2g2(self, s):
        '''Compute C_l^g2g2.'''
        def kernel(z3, z2, z1, ell):
            chi_z3 = self.cosmo.interp_z2chi(z3)
            p1 = self.fg(z1) * self.fg(z2) * (1 + z3)**2 * self.cosmo.interp_w_z(z3, zs=z1) * \
                self.cosmo.interp_w_z(z3, zs=z2) / \
                self.cosmo.H_z(z3) / chi_z3**2
            return p1 * self.cosmo.interp_pk.P(z3, ell/chi_z3)

        def target(ell):
            iz = self.cosmo.interp_chi2z(ell/self.kmax)
            return spint.tplquad(kernel, self.z1, self.z2, lambda x: self.z1, lambda x: self.z2,
                                 lambda x, y: iz, lambda x, y: min(x, y),
                                 args=(ell,), epsabs=0, epsrel=1e-3)[0]

        print('>> Computing C_l^g2g2...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = (3 * self.cosmo.H0**2 * self.cosmo.Om0 /
               C_LIGHT**2)**2 * (5./2. * s - 1)**2 * C_LIGHT

        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clkg2(self, s):
        '''Compute C_l^kg2.'''
        def kernel(z2, z1, ell):
            chi_z2 = self.cosmo.interp_z2chi(z2)
            p1 = (1 + z2)**2 * self.cosmo.interp_w_z(z2, zs=z1) * self.cosmo.interp_w_z(z2) * \
                self.fg(z1) / self.cosmo.H_z(z2) / chi_z2**2
            return p1 * self.cosmo.interp_pk.P(z2, ell/chi_z2)

        def target(ell):
            iz = self.cosmo.interp_chi2z(ell/self.kmax)
            return spint.dblquad(kernel, self.z1, self.z2, lambda x: iz, lambda x: x,
                                 args=(ell,), epsabs=0, epsrel=1e-3)[0]

        print('>> Computing C_l^kg2...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 1./2. * (3 * self.cosmo.H0**2 * self.cosmo.Om0 /
                       C_LIGHT**2)**2 * (5./2. * s - 1) * C_LIGHT
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clkk(self):
        '''Compute C_l^kk.'''
        def kernel(z, ell):
            chi_z = self.cosmo.z2chi(z)
            k = (ell + 1./2.) / chi_z
            w_z = self.cosmo.w_z(z)
            p1 = (1 + z)**2 * w_z**2 / self.cosmo.H_z(z) / chi_z**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            iz = self.cosmo.interp_chi2z((ell+1./2.)/self.kmax)
            return spint.quad(kernel, iz, Z_CMB, args=(ell,),
                              epsabs=0, epsrel=1e-6)[0]

        print('>> Computing C_l^kk...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 1./4. * (3 * self.cosmo.H0**2 *
                       self.cosmo.Om0 / C_LIGHT**2)**2 * C_LIGHT
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    # ------ fNL & RSD ------ #

    def K_ell(self, ell):
        '''A function of ell due to RSD.'''
        p1 = (2*np.power(ell, 2) + 2*ell - 1) / \
            ((2*ell-1) * (2*ell+3) * np.sqrt(ell+1./2.))
        p2 = - ell * (ell-1) / (2 * (2*ell-1) * np.power(ell-3./2., 3./2.))
        p3 = - (ell+1) * (ell+2) / (2 * (2*ell+3) * np.power(ell+5./2., 3./2.))
        return p1 + p2 + p3

    def c_clkg_nl(self):
        '''Compute C_l^kg,NL.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = self.fg(z) * (self.b(z) - 1) * self.cosmo.interp_w_z(z) / \
                self.cosmo.interp_Tk(k) / self.cosmo.interp_D_z(z)
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-4)[0] / (ell+1./2.)**2

        print('>> Computing C_l^kg,nl...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 9./2. * (self.cosmo.H0 / C_LIGHT)**4 * \
            self.cosmo.Om0**2 * DELTA_C
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clkg_r(self):
        '''Compute C_l^kg,RSD.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = self.cosmo.f_growth_z(z) * self.fg(z) * \
                self.cosmo.interp_w_z(z) / chi_z**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-4)[0] * self.K_ell(ell) * np.sqrt(ell+1./2.)

        print('>> Computing C_l^kg,RSD...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 3./2. * (self.cosmo.H0/C_LIGHT)**2 * self.cosmo.Om0
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clgg_0f(self):
        '''Compute C_l^gg,0f.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = self.fg(z)**2 * self.cosmo.H_z(z) * self.b(z) * (self.b(z) - 1) / \
                self.cosmo.interp_Tk(k) / self.cosmo.interp_D_z(z)
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-3)[0] / (ell+1./2.)**2

        print('>> Computing C_l^gg,0f...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 3. * self.cosmo.Om0 * self.cosmo.H0**2 * DELTA_C / C_LIGHT**3
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clgg_ff(self):
        '''Compute C_l^gg,ff.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = self.cosmo.H_z(z) * self.fg(z)**2 * chi_z**2 * (self.b(z)-1)**2 / \
                self.cosmo.interp_Tk(k)**2 / self.cosmo.interp_D_z(z)**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-3)[0] / (ell+1./2.)**4

        print('>> Computing C_l^gg,ff...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 9. * self.cosmo.Om0**2 * self.cosmo.H0**4 * DELTA_C**2 / C_LIGHT**5
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clgg_0r(self):
        '''Compute C_l^gg,0r.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = np.power(chi_z, -2) * self.b(z) * \
                self.cosmo.f_growth_z(z) * self.cosmo.H_z(z) * self.fg(z)**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-3)[0] * self.K_ell(ell) * np.sqrt(ell+1./2.)

        print('>> Computing C_l^gg,0r...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 1. / C_LIGHT
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clgg_rr(self):
        '''Compute C_l^gg,rr.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = np.power(chi_z, -2) * self.cosmo.f_growth_z(z)**2 * \
                self.cosmo.H_z(z) * self.fg(z)**2
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-3)[0] * (ell+1./2.) * self.K_ell(ell)**2

        print('>> Computing C_l^gg,rr...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 1. / C_LIGHT
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl

    def c_clgg_fr(self):
        '''Compute C_l^gg,fr.'''
        def kernel(z, ell):
            chi_z = self.cosmo.interp_z2chi(z)
            k = (ell + 1./2.) / chi_z
            p1 = (self.b(z)-1) * self.cosmo.f_growth_z(z) * self.fg(z)**2 * self.cosmo.H_z(z) / \
                self.cosmo.interp_Tk(k) / self.cosmo.interp_D_z(z)
            return p1 * self.cosmo.interp_pk.P(z, k)

        def target(ell):
            return spint.quad(kernel, self.z1, self.z2, args=(ell,),
                              epsabs=0, epsrel=1e-3)[0] * self.K_ell(ell) / np.power(ell+1./2., 3./2.)

        print('>> Computing C_l^gg,fr...')
        tt0 = time.time()
        cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
                                            for ell in self.ells)

        fac = 3. * self.cosmo.Om0 * self.cosmo.H0**2 * DELTA_C / C_LIGHT**3
        cl = np.array(cl) * fac

        print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
        return cl
