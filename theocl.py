'''
Theoretical computation of C_ell.
'''
import numpy as np
from scipy import integrate, interpolate
from scipy.special import spherical_jn
import h5py

import multiprocessing as mp
from tqdm import tqdm

from .utils import jl_2nd
from .constants import *


class cc:
    '''Some quantities and functions used in exact integration and Limber approximation.'''

    def __init__(self, cosmo, fg=None, bg=None, zg1=None, zg2=None):

        # ------ cosmology ------
        self.cosmo = cosmo
        self.chi_CMB = cosmo.chi(Z_CMB)

        # ------ galaxy survey details ------
        self.fg = fg  # galaxy redshift distribution, f_g(z)
        self.bg = bg  # galaxy linear bias function, b_g(z)
        self.zg1, self.zg2 = zg1, zg2  # redshift bin of the galaxy survey
        self.chig1 = self.cosmo.chi(zg1)
        self.chig2 = self.cosmo.chi(zg2)

        # ------ Limber approximation related ------
        self.g_set = ['g', 'g_NL', 'g_R', 'g_M']  # with galaxy window function

        self.num_cpus = mp.cpu_count()
        print('>> Number of CPUs: {:d}'.format(self.num_cpus))

    # ------ window functions ------

    def bar_W(self, X, chis, zs):
        '''Bar{W}_X(chi) function.'''
        if X == 'kappa':
            fac = 3./2. * self.cosmo.Om0 * (self.cosmo.H0 / C_LIGHT)**2
            return fac * (1 + zs) * chis * (1 - chis / self.chi_CMB) * \
                self.cosmo.D(zs, interp=True)
        if X == 'g':
            return self.cosmo.H(zs)/C_LIGHT * self.fg(zs) * self.bg(zs) * \
                self.cosmo.D(zs, interp=True)

    # ------ fNL related ------

    def beta(self, ks, zs, arr=False):
        '''beta(k, z) function.'''
        fac = 3 * self.cosmo.Om0 * DELTA_C * (self.cosmo.H0 / C_LIGHT)**2
        fzs = (1 - 1 / self.bg(zs)) / self.cosmo.D_unnorm(zs, interp=True)
        fks = 1 / (ks**2 * self.cosmo.Tk(ks))
        if arr:
            return fac * np.einsum('k,c->kc', fks, fzs)
        else:
            return fac * fks * fzs

    # ------ Limber approximation related ------

    def Delta_Limber(self, X, ell, k):
        '''Compute Delta_{X, ell}(k) with Limber approximation.'''

        chi_lk = (ell + 0.5) / k
        z_lk = self.cosmo.z_at_chi(chi_lk)

        if X in ['kappa', 'g']:
            return np.sqrt(np.pi/(2*ell+1)) * self.bar_W(X, chi_lk, z_lk) / k

        if X == 'g_NL':
            return np.sqrt(np.pi/(2*ell+1)) * self.bar_W('g', chi_lk, z_lk) * self.beta(k, z_lk) / k

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


def c_clxy_limber(cc, X, Y, ells, limber_kmax):
    '''Compute C_l^XY at ell with Limber approximation.'''
    def integrand(k, ell):
        return k**2 * cc.Delta_Limber(X, ell, k) * cc.Delta_Limber(Y, ell, k) * cc.cosmo.Pk(k)

    def c_one(ell):
        # integral limits
        if X in cc.g_set or Y in cc.g_set:
            kmin = (ell + 0.5) / cc.chig2
            kmax = min(limber_kmax, (ell + 0.5) / cc.chig1)
        else:
            kmin = (ell + 0.5) / cc.chi_CMB
            kmax = limber_kmax

        return 2./np.pi * integrate.quad(integrand, kmin, kmax, args=(ell,),
                                         epsabs=0, epsrel=1e-4)[0]

    res = np.array([c_one(ell) for ell in ells])

    return res


class ccl:
    '''Theoretical computation of cl.'''

    def __init__(self, cc, fn_jls):

        self.cc = cc

        # ------ k, chi & j_ell sample points ------
        self.fn_jls = fn_jls
        self.fss = h5py.File(fn_jls, 'r')
        self.ks = self.fss['ks'][:]
        self.chis = self.fss['chis'][:]
        self.kchis = self.fss['kchis'][:]  # outer product of ks and chis
        self.ell_ss = self.fss['ells'][:]
        # z sample points corresponding to chis
        self.zs = self.cc.cosmo.z_at_chi(self.chis)

        # ------ values at sample points ------
        # window functions
        self.bW_Xs = {'kappa': None, 'g': None}
        # transfer functions
        self.Delta_Xs = {'kappa': {}, 'g': {},
                         'g_NL': {}, 'g_R': {}, 'g_M': {}}
        # matter power spectrum at z=0
        self.Pk0s = self.cc.cosmo.Pk(self.ks)
        # for f_NL
        self.betas = None

        # ------ set redshift masks for all fields ------
        # redshift mask
        self.zms = {}
        self.zms['kappa'] = (self.zs <= Z_CMB)
        self.zms['g'] = (self.zs >= self.cc.zg1) & (self.zs <= self.cc.zg2)

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

    # ------ window functions ------ #

    def c_bar_W(self, X):
        '''Compute Bar{W}_X(chi) at chi sample points.'''
        if self.bW_Xs[X] is None:
            zs = self.zs[self.zms[X]]
            chis = self.chis[self.zms[X]]
            self.bW_Xs[X] = self.cc.bar_W(X, chis, zs)

    # ------ f_NL related ------ #

    def c_beta(self):
        '''Compute beta(k, z) at all (k, chi) sample points.'''
        if self.betas is None:
            zs = self.zs[self.zms['g']]
            self.betas = self.cc.beta(self.ks, zs, arr=True)

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

    # ------ angular power spectra ------ #

    def c_clxy_exact(self, X, Y, ell):
        '''Compute C_l^XY at ell with exact integration.'''
        self.c_Delta(X, ell)
        self.c_Delta(Y, ell)
        Deltas = self.Delta_Xs[X][ell] * self.Delta_Xs[Y][ell]
        return 2/np.pi * np.trapz(self.ks**2 * Deltas * self.Pk0s,
                                  self.ks, axis=-1)

    def c_clxy(self, X, Y, ells, progbar=True, use_Limber=False, Limber_kmax=5.):
        '''Compute C_l^XY.'''
        res = np.full_like(ells, 0., dtype=np.float)

        if use_Limber:  # using Limber approximation
            ells_ = tqdm(ells) if progbar else ells
            res = c_clxy_limber(self.cc, X, Y, ells_, Limber_kmax)
        else:  # exact integration
            enum_ells = enumerate(tqdm(ells)) if progbar else enumerate(ells)
            for i, ell in enum_ells:
                res[i] = self.c_clxy_exact(X, Y, ell)

        return res
