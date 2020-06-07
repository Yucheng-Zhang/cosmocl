'''
Theoretical computation of C_ell.
'''
import numpy as np
from scipy import integrate, interpolate
from scipy.special import spherical_jn
import h5py

import multiprocessing as mp
from joblib import Parallel, delayed
import sys
import time
from tqdm import tqdm

from .utils import jl_2nd
from .constants import *


class ccl:
    '''Theoretical computation of cl.'''

    def __init__(self, cosmo, fn_jls, fg=None, bg=None, zg1=None, zg2=None):

        # ------ cosmology model ------
        self.cosmo = cosmo  # cosmopy.cosmology.flatLCDM instance
        self.chi_CMB = cosmo.chi(Z_CMB)

        # ------ k, chi & j_ell sample points ------
        self.fn_jls = fn_jls
        self.fss = h5py.File(fn_jls, 'r')
        self.ks = self.fss['ks'][:]
        self.chis = self.fss['chis'][:]
        self.kchis = self.fss['kchis'][:]  # outer product of ks and chis
        self.ell_ss = self.fss['ells'][:]
        # z sample points corresponding to chis
        self.zs = self.cosmo.z_at_chi(self.chis)

        # ------ values at sample points ------
        # window functions
        self.bW_Xs = {'kappa': None, 'g': None}
        # transfer functions
        self.Delta_Xs = {'kappa': {}, 'g': {},
                         'g_NL': {}, 'g_R': {}, 'g_M': {}}
        # matter power spectrum at z=0
        self.Pk0s = self.cosmo.Pk(self.ks)
        # for f_NL
        self.betas = None

        # ------ galaxy survey details ------
        self.fg = fg  # galaxy redshift distribution, f_g(z)
        self.bg = bg  # galaxy linear bias function, b_g(z)
        self.zg1, self.zg2 = zg1, zg2  # redshift bin of the galaxy survey
        self.chig1 = self.cosmo.chi(zg1)
        self.chig2 = self.cosmo.chi(zg2)

        # ------ set redshift masks for all fields ------
        # redshift mask
        self.zms = {}
        self.zms['kappa'] = (self.zs <= Z_CMB)
        self.zms['g'] = (self.zs >= self.zg1) & (self.zs <= self.zg2)

        # Limber approximation related
        self.g_set = ['g', 'g_NL', 'g_R', 'g_M']  # with galaxy window function

        self.num_cpus = mp.cpu_count()
        print('>> Number of CPUs: {:d}'.format(self.num_cpus))

    def get_jls(self, ell, der=0):
        '''Get j_ell samples from data file or spherical_jn.'''
        if der == 0:
            if ell in self.ell_ss:
                return self.fss['j_{:d}'.format(ell)][:]
            else:
                return spherical_jn(ell, self.kchis)

        if der == 2:
            jls_ = {}
            for i, ell_ in enumerate([ell-2, ell, ell+2]):
                if ell_ in self.ell_ss:
                    jls_[i] = self.fss['j_{:d}'.format(ell_)][:]
                else:
                    jls_[i] = spherical_jn(ell_, self.kchis)
            return jl_2nd(ell, jls=jls_)

    # ------ kernel functions ------ #

    def bar_W(self, X, chis, zs):
        '''Bar{W}_X(chi) function.'''
        if X == 'kappa':
            fac = 3./2. * self.cosmo.Om0 * (self.cosmo.H0 / C_LIGHT)**2
            return fac * (1 + zs) * chis * (1 - chis / self.chi_CMB) * \
                self.cosmo.D(zs, interp=True)
        if X == 'g':
            return self.cosmo.H(zs)/C_LIGHT * self.fg(zs) * self.bg(zs) * \
                self.cosmo.D(zs, interp=True)

    def c_bar_W(self, X):
        '''Compute Bar{W}_X(chi) at chi sample points.'''
        if self.bW_Xs[X] is None:
            zs = self.zs[self.zms[X]]
            chis = self.chis[self.zms[X]]
            self.bW_Xs[X] = self.bar_W(X, chis, zs)

    # ------ f_NL related ------ #

    def beta(self, ks, zs):
        '''beta(k, z) function.'''
        fac = 3 * self.cosmo.Om0 * DELTA_C * (self.cosmo.H0 / C_LIGHT)**2
        fzs = (1 - 1 / self.bg(zs)) / self.cosmo.D_unnorm(zs, interp=True)
        fks = 1 / (ks**2 * self.cosmo.Tk(ks))
        if np.isscalar(ks) and np.isscalar(zs):
            return fac * fks * fzs
        else:
            return fac * np.einsum('k,c->kc', fks, fzs)

    def c_beta(self):
        '''Compute beta(k, z) at all (k, chi) sample points.'''
        if self.betas is None:
            zs = self.zs[self.zms['g']]
            self.betas = self.beta(self.ks, zs)

    # ------ transfer functions ------ #

    def c_Delta(self, X, ell):
        '''Compute Delta_{X, ell}(k) at k sample points.'''

        if ell not in self.Delta_Xs[X].keys():  # not computed before

            if X in ['kappa', 'g']:
                self.c_bar_W(X)
                jls = self.get_jls(ell)[:, self.zms[X]]
                chis = self.chis[self.zms[X]]
                self.Delta_Xs[X][ell] = np.trapz(np.einsum('c,kc->kc', self.bW_Xs[X], jls),
                                                 chis, axis=-1)
            if X == 'g_NL':
                self.c_bar_W('g')
                self.c_beta()
                jls = self.get_jls(ell)[:, self.zms['g']]
                chis = self.chis[self.zms['g']]
                self.Delta_Xs[X][ell] = np.trapz(np.einsum('c,kc,kc->kc', self.bW_Xs['g'],
                                                           self.betas, jls), chis, axis=-1)

            if X == 'g_R':
                self.c_bar_W('g')
                jl_2nds = self.get_jls(ell, der=2)[:, self.zms['g']]
                chis = self.chis[self.zms['g']]
                self.Delta_Xs[X][ell] = - np.trapz(np.einsum('c,kc->kc', self.bW_Xs['g'], jl_2nds),
                                                   chis, axis=-1)

            if X == 'g_M':
                pass

    def Delta_Limber(self, X, ell, k):
        '''Compute Delta_{X, ell}(k) with Limber approximation.'''
        chi_lk = (ell + 0.5) / k
        z_lk = self.cosmo.z_at_chi(chi_lk)

        if X in ['kappa', 'g']:
            return np.sqrt(np.pi/(2*ell+1)) * self.bar_W(X, chi_lk, z_lk) / k

        if X == 'g_NL':
            return np.sqrt(np.pi/(2*ell+1)) * self.bar_W(X, chi_lk, z_lk) * self.beta(k, z_lk) / k

        if X == 'g_R':
            t1 = ell * (ell-1) / (2*ell-1) / (2*ell+1) / np.sqrt(2*ell-3) * \
                self.bar_W('g', (ell-1.5)/k, self.cosmo.z_at_chi((ell-1.5)/k))
            t2 = (2*ell**2-1) / (2*ell-1) / (2*ell+3) / np.sqrt(2*ell+1) * \
                self.bar_W('g', chi_lk, z_lk)
            t3 = (ell+1) * (ell+2) / (2*ell+1) / (2*ell+3) / np.sqrt(2*ell+5) * \
                self.bar_W('g', (ell+2.5)/k, self.cosmo.z_at_chi((ell+2.5)/k))
            return - np.sqrt(np.pi) * (t1 - t2 + t3) / k

        if X == 'g_M':
            pass

    # ------ angular power spectra ------ #

    def c_clxy_exact(self, X, Y, ell):
        '''Compute C_l^XY at ell with exact integration.'''
        self.c_Delta(X, ell)
        self.c_Delta(Y, ell)
        Deltas = self.Delta_Xs[X][ell] * self.Delta_Xs[Y][ell]
        return 2/np.pi * np.trapz(self.ks**2 * Deltas * self.Pk0s,
                                  self.ks, axis=-1)

    def c_clxy_limber(self, X, Y, ells, limber_kmax):
        '''Compute C_l^XY at ells with Limber approximation.'''
        def integrand(k, ell):
            return k**2 * self.Delta_Limber(X, ell, k) * self.Delta_Limber(Y, ell, k) * \
                self.cosmo.Pk(k)

        def c_one(ell):
           # integral limits
            if X in self.g_set or Y in self.g_set:
                kmin = (ell + 0.5) / self.chig2
                kmax = min(limber_kmax, (ell + 0.5) / self.chig1)
            else:
                kmin = (ell + 0.5) / self.chi_CMB
                kmax = limber_kmax

            return 2./np.pi * integrate.quad(integrand, kmin, kmax, args=(ell,),
                                             epsabs=0, epsrel=1e-4)[0]

        # return Parallel(n_jobs=self.num_cpus)(delayed(c_one)(ell)
        #                                       for ell in ells)
        return np.array([c_one(ell) for ell in ells])

    def c_clxy(self, X, Y, ells, progbar=True, use_Limber=False, Limber_kmax=5.):
        '''Compute C_l^XY.'''
        res = np.full_like(ells, 0., dtype=np.float)

        if use_Limber:  # using Limber approximation
            ells_ = tqdm(ells) if progbar else ells
            res = self.c_clxy_limber(X, Y, ells_, Limber_kmax)
        else:  # exact integration
            enum_ells = enumerate(tqdm(ells)) if progbar else enumerate(ells)
            for i, ell in enum_ells:
                res[i] = self.c_clxy_exact(X, Y, ell)

        return res


# class ccl_Limber:
#     '''Theoretical computation of cl w/ Limber approximation.'''

#     def __init__(self, ells, z1, z2, fg, b, cosmo):
#         self.lmin, self.lmax = np.amin(ells), np.amax(ells)
#         self.z1, self.z2 = z1, z2
#         self.fg = fg  # redshift distribution function
#         self.b = b  # linear bias function
#         self.cosmo = cosmo  # cosmoLCDM instance

#         self.ells = ells

#         self.num_cpus = mp.cpu_count()
#         print('>> Number of CPUs: {0:d}'.format(self.num_cpus))

#     def set_interp_pk(self, zmin, zmax, kmax, extrap_kmax=None):
#         self.cosmo.gen_interp_pk(zmin, zmax, kmax, extrap_kmax=extrap_kmax)
#         print(':: PK interpolator settings ::')
#         print(':: zmin = {0:g}, zmax = {1:g}'.format(zmin, zmax))
#         print(':: kmax = {0:g}'.format(kmax))
#         if extrap_kmax is not None:
#             print(':: extrap_kmax = {0:g}'.format(extrap_kmax))
#         print(
#             ':: (note: for points out of range, the returned value is the boundary value)')

#         if extrap_kmax is None:
#             self.kmax = kmax
#         else:
#             self.kmax = extrap_kmax

#         print('-- checking integral over z from zero')
#         print('- ell = {0:g} : z_zero = {1:g}'.format(
#             self.ells[0], self.cosmo.interp_chi2z(self.ells[0] / self.kmax)))
#         print('- ell = {0:g} : z_zero = {1:g}'.format(
#             self.ells[-1], self.cosmo.interp_chi2z(self.ells[-1] / self.kmax)))

#     def get_ells(self):
#         return self.ells

#     # ------ power spectra ------ #

#     def c_clkg(self):
#         '''Compute C_l^kg.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             p1 = (1 + z) * self.cosmo.interp_w_z(z) * \
#                 self.fg(z) * self.b(z) / chi_z**2
#             return p1 * self.cosmo.interp_pk.P(z, ell/chi_z)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-4)[0]

#         print('>> Computing C_l^kg...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 3. * self.cosmo.H0**2 * self.cosmo.Om0 / (2. * C_LIGHT**2)
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clgg(self):
#         '''Compute C_l^gg.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             p1 = self.cosmo.H_z(z) * self.fg(z)**2 * self.b(z)**2 / chi_z**2
#             return p1 * self.cosmo.interp_pk.P(z, ell/chi_z)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-3)[0]

#         print('>> Computing C_l^gg...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         cl = np.array(cl) / C_LIGHT

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     # ------ C_Gamma calibration ------ #

#     def c_clm(self):
#         '''Compute C_l^m.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = self.cosmo.H_z(z) * self.fg(z)**2 / chi_z**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-3)[0]

#         print('>> Computing C_l^m...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         cl = np.array(cl) / C_LIGHT

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clmg(self):
#         '''Compute C_l^mg.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = self.cosmo.H_z(z) * self.b(z) * self.fg(z)**2 / chi_z**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-6)[0]

#         print('>> Computing C_l^mg...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         cl = np.array(cl) / C_LIGHT

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_qlm(self):
#         '''Compute Q_l^m.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = (1 + z) * self.cosmo.interp_w_z(z) * \
#                 self.fg(z) / chi_z**2 * self.cosmo.f_growth_z(z)
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-4)[0]

#         print('>> Computing Q_l^m...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         cl = np.array(cl) / 2.

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_qlmg(self):
#         '''Compute Q_l^mg.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = (1 + z) * self.cosmo.interp_w_z(z) * \
#                 self.fg(z) * self.b(z) / chi_z**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-4)[0]

#         print('>> Computing Q_l^mg...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         cl = np.array(cl) / 2.

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_qlgg(self):
#         '''Compute Q_l^gg.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = (1 + z) * self.cosmo.interp_w_z(z) * \
#                 self.fg(z) * self.b(z)**2 / chi_z**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-4)[0]

#         print('>> Computing Q_l^gg...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         cl = np.array(cl) / 2.

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     # ------ Magnification bias calibration ------ #

#     def c_clg1g2(self, s):
#         '''Compute C_l^g1g2.'''
#         def kernel(z2, z1, ell):
#             chi_z2 = self.cosmo.interp_z2chi(z2)
#             p1 = (1 + z2) * self.cosmo.interp_w_z(z2, zs=z1) * \
#                 self.fg(z1) * self.fg(z2) * self.b(z2) / chi_z2**2
#             return p1 * self.cosmo.interp_pk.P(z2, ell/chi_z2)

#         def target(ell):
#             return integrate.dblquad(kernel, self.z1, self.z2, lambda x: self.z1, lambda x: x,
#                                      args=(ell,), epsabs=0, epsrel=1e-3)[0]

#         print('>> Computing C_l^g1g2...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 3. * self.cosmo.H0**2 * self.cosmo.Om0 / C_LIGHT**2
#         fac = fac * (5./2. * s - 1)

#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clg2g2(self, s):
#         '''Compute C_l^g2g2.'''
#         def kernel(z3, z2, z1, ell):
#             chi_z3 = self.cosmo.interp_z2chi(z3)
#             p1 = self.fg(z1) * self.fg(z2) * (1 + z3)**2 * self.cosmo.interp_w_z(z3, zs=z1) * \
#                 self.cosmo.interp_w_z(z3, zs=z2) / \
#                 self.cosmo.H_z(z3) / chi_z3**2
#             return p1 * self.cosmo.interp_pk.P(z3, ell/chi_z3)

#         def target(ell):
#             iz = self.cosmo.interp_chi2z(ell/self.kmax)
#             return integrate.tplquad(kernel, self.z1, self.z2, lambda x: self.z1, lambda x: self.z2,
#                                      lambda x, y: iz, lambda x, y: min(x, y),
#                                      args=(ell,), epsabs=0, epsrel=1e-3)[0]

#         print('>> Computing C_l^g2g2...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = (3 * self.cosmo.H0**2 * self.cosmo.Om0 /
#                C_LIGHT**2)**2 * (5./2. * s - 1)**2 * C_LIGHT

#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clkg2(self, s):
#         '''Compute C_l^kg2.'''
#         def kernel(z2, z1, ell):
#             chi_z2 = self.cosmo.interp_z2chi(z2)
#             p1 = (1 + z2)**2 * self.cosmo.interp_w_z(z2, zs=z1) * self.cosmo.interp_w_z(z2) * \
#                 self.fg(z1) / self.cosmo.H_z(z2) / chi_z2**2
#             return p1 * self.cosmo.interp_pk.P(z2, ell/chi_z2)

#         def target(ell):
#             iz = self.cosmo.interp_chi2z(ell/self.kmax)
#             return integrate.dblquad(kernel, self.z1, self.z2, lambda x: iz, lambda x: x,
#                                      args=(ell,), epsabs=0, epsrel=1e-3)[0]

#         print('>> Computing C_l^kg2...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 1./2. * (3 * self.cosmo.H0**2 * self.cosmo.Om0 /
#                        C_LIGHT**2)**2 * (5./2. * s - 1) * C_LIGHT
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clkk(self):
#         '''Compute C_l^kk.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             w_z = self.cosmo.w_z(z)
#             p1 = (1 + z)**2 * w_z**2 / self.cosmo.H_z(z) / chi_z**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             iz = self.cosmo.interp_chi2z((ell+1./2.)/self.kmax)
#             return integrate.quad(kernel, iz, Z_CMB, args=(ell,),
#                                   epsabs=0, epsrel=1e-6)[0]

#         print('>> Computing C_l^kk...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 1./4. * (3 * self.cosmo.H0**2 *
#                        self.cosmo.Om0 / C_LIGHT**2)**2 * C_LIGHT
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     # ------ fNL & RSD ------ #

#     def K_ell(self, ell):
#         '''A function of ell due to RSD.'''
#         p1 = (2*np.power(ell, 2) + 2*ell - 1) / \
#             ((2*ell-1) * (2*ell+3) * np.sqrt(ell+1./2.))
#         p2 = - ell * (ell-1) / (2 * (2*ell-1) * np.power(ell-3./2., 3./2.))
#         p3 = - (ell+1) * (ell+2) / (2 * (2*ell+3) * np.power(ell+5./2., 3./2.))
#         return p1 + p2 + p3

#     def c_clkg_nl(self):
#         '''Compute C_l^kg,NL.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = self.fg(z) * (self.b(z) - 1) * self.cosmo.interp_w_z(z) / \
#                 self.cosmo.interp_Tk(k) / self.cosmo.interp_D_z(z)
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-4)[0] / (ell+1./2.)**2

#         print('>> Computing C_l^kg,nl...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 9./2. * (self.cosmo.H0 / C_LIGHT)**4 * \
#             self.cosmo.Om0**2 * DELTA_C
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clkg_r(self):
#         '''Compute C_l^kg,RSD.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = self.cosmo.f_growth_z(z) * self.fg(z) * \
#                 self.cosmo.interp_w_z(z) / chi_z**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-4)[0] * self.K_ell(ell) * np.sqrt(ell+1./2.)

#         print('>> Computing C_l^kg,RSD...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 3./2. * (self.cosmo.H0/C_LIGHT)**2 * self.cosmo.Om0
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clgg_0f(self):
#         '''Compute C_l^gg,0f.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = self.fg(z)**2 * self.cosmo.H_z(z) * self.b(z) * (self.b(z) - 1) / \
#                 self.cosmo.interp_Tk(k) / self.cosmo.interp_D_z(z)
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-3)[0] / (ell+1./2.)**2

#         print('>> Computing C_l^gg,0f...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 3. * self.cosmo.Om0 * self.cosmo.H0**2 * DELTA_C / C_LIGHT**3
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clgg_ff(self):
#         '''Compute C_l^gg,ff.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = self.cosmo.H_z(z) * self.fg(z)**2 * chi_z**2 * (self.b(z)-1)**2 / \
#                 self.cosmo.interp_Tk(k)**2 / self.cosmo.interp_D_z(z)**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-3)[0] / (ell+1./2.)**4

#         print('>> Computing C_l^gg,ff...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 9. * self.cosmo.Om0**2 * self.cosmo.H0**4 * DELTA_C**2 / C_LIGHT**5
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clgg_0r(self):
#         '''Compute C_l^gg,0r.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = np.power(chi_z, -2) * self.b(z) * \
#                 self.cosmo.f_growth_z(z) * self.cosmo.H_z(z) * self.fg(z)**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-3)[0] * self.K_ell(ell) * np.sqrt(ell+1./2.)

#         print('>> Computing C_l^gg,0r...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 1. / C_LIGHT
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clgg_rr(self):
#         '''Compute C_l^gg,rr.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = np.power(chi_z, -2) * self.cosmo.f_growth_z(z)**2 * \
#                 self.cosmo.H_z(z) * self.fg(z)**2
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-3)[0] * (ell+1./2.) * self.K_ell(ell)**2

#         print('>> Computing C_l^gg,rr...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 1. / C_LIGHT
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl

#     def c_clgg_fr(self):
#         '''Compute C_l^gg,fr.'''
#         def kernel(z, ell):
#             chi_z = self.cosmo.interp_z2chi(z)
#             k = (ell + 1./2.) / chi_z
#             p1 = (self.b(z)-1) * self.cosmo.f_growth_z(z) * self.fg(z)**2 * self.cosmo.H_z(z) / \
#                 self.cosmo.interp_Tk(k) / self.cosmo.interp_D_z(z)
#             return p1 * self.cosmo.interp_pk.P(z, k)

#         def target(ell):
#             return integrate.quad(kernel, self.z1, self.z2, args=(ell,),
#                                   epsabs=0, epsrel=1e-3)[0] * self.K_ell(ell) / np.power(ell+1./2., 3./2.)

#         print('>> Computing C_l^gg,fr...')
#         tt0 = time.time()
#         cl = Parallel(n_jobs=self.num_cpus)(delayed(target)(ell)
#                                             for ell in self.ells)

#         fac = 3. * self.cosmo.Om0 * self.cosmo.H0**2 * DELTA_C / C_LIGHT**3
#         cl = np.array(cl) * fac

#         print('>> Time elapsed: {0:.2f} s'.format(time.time() - tt0))
#         return cl
