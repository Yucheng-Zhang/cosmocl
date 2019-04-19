'''
Theoretical calculation of Cl.
'''
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as spint
from astropy.cosmology import LambdaCDM
import camb
import argparse
import sys
import multiprocessing as mp

C_LIGHT = 299792.458  # speed of light in km/s
Z_CMB = 1100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Theoretical calculation of Cl.')
    parser.add_argument('-cal', type=str, help='Calculate what.')

    parser.add_argument('-H0', type=float, default=70.0,
                        help='Hubble constant.')
    parser.add_argument('-Omm0', type=float, default=0.3,
                        help='Matter density.')
    parser.add_argument('-Omb0', type=float, default=0.043,
                        help='Baryon density.')
    parser.add_argument('-Tcmb0', type=float, default=2.7255,
                        help='CMB temperature.')
    parser.add_argument('-ns', type=float, default=0.96,
                        help='Scalar index.')
    parser.add_argument('-As', type=float, default=2.4675e-9,
                        help='Scalar amplitude.')

    parser.add_argument('-bkg', type=float, default=1.0,
                        help='Bias for clkg.')
    parser.add_argument('-bgg', type=float, default=1.0,
                        help='Bias for clgg.')

    parser.add_argument('-lmm', type=int, nargs='+', default=[2, 1024],
                        help='Min and max of ell.')
    parser.add_argument('-z1z2', type=float, nargs='+',
                        help='Redshift range.')

    parser.add_argument('-dataz', type=str, help='Galaxy redshift file.')

    parser.add_argument('-fo', type=str, default='output-cl.dat',
                        help='Output file name.')

    args = parser.parse_args()


class cosmoLCDM:
    '''LambdaCDM cosmology.'''

    def __init__(self, H0, Om0, Ode0):
        self.H0 = H0
        self.cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
        self.h = self.H0 / 100.
        self.Om0 = Om0

    def H_z(self, z):
        '''Hubble parameter at redshift z, i.e. H(z)'''
        return self.H0 * self.cosmo.efunc(z)

    def z2chi(self, z):
        '''Get comoving distance in [Mpc] from redshift'''
        return self.cosmo.comoving_distance(z).value

    def w_z(self, z):
        '''CMB lensing kernel.'''
        chi_z = self.z2chi(z)
        chi_cmb = self.z2chi(Z_CMB)
        return chi_z * (1. - chi_z / chi_cmb)


def gen_fg_z(dataz, z1, z2, bins=200):
    '''Generate the redshift distribution function fg(z).'''
    hist, bin_edges = np.histogram(
        dataz, bins=bins, range=(z1, z2), density=True)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    xs = np.concatenate(([z1], bin_mids, [z2]))
    ys = np.concatenate(([hist[0]], hist, [hist[-1]]))

    fg_z = spi.interp1d(xs, ys, kind='linear',
                        bounds_error=False, fill_value=(0., 0.))

    return fg_z


def gen_pk(pars, kmax, z1, z2):
    '''Generate matter power spectrum Pmm(z, k).'''
    pk = camb.get_matter_power_interpolator(pars, zmin=z1-0.1, zmax=z2+0.1,
                                            kmax=kmax, nonlinear=True,
                                            hubble_units=False, k_hunit=False)

    return pk


def clkg(lmin, lmax, cosmo, z1, z2, fg, pk, bkg):
    '''C_l^kg, LCDM & Limber approximation.'''
    def clkg_kernel(z, ell):
        chi_z = cosmo.z2chi(z)
        p1 = (1 + z) * cosmo.w_z(z) * fg(z) / chi_z**2
        p2 = pk.P(z, ell/chi_z) * bkg
        return p1*p2

    ells = np.arange(lmin, lmax+1, 1, dtype='int32')
    clkgs = np.zeros(len(ells))
    errs = np.zeros(len(ells))

    for i, ell in enumerate(ells):
        print('>> Integrating ell {0:d}...'.format(ell))
        y, err = spint.quad(clkg_kernel, z1, z2, args=(ell,))
        clkgs[i], errs[i] = y, err

    fac = 3. * cosmo.H0**2 * cosmo.Om0 / (2. * C_LIGHT**2)
    clkgs = clkgs * fac

    return ells, clkgs


def clgg(lmin, lmax, cosmo, z1, z2, fg, pk, bgg):
    '''C_l^gg, LCDM & Limber approximation.'''
    def clgg_kernel(z, ell):
        chi_z = cosmo.z2chi(z)
        p1 = cosmo.H_z(z) * fg(z)**2 / chi_z**2
        p2 = pk.P(z, ell/chi_z) * bgg**2
        return p1*p2

    def target(ell):
        print('>> Integrating ell {0:d}...'.format(ell))
        y, err = spint.quad(clgg_kernel, z1, z2, args=(ell,))
        return y

    ells = np.arange(lmin, lmax+1, 1, dtype='int32')

    pool = mp.Pool(mp.cpu_count())
    clggs = pool.map(target, ells)
    pool.close()
    pool.join()

    clggs = np.array(clggs)
    clggs = clggs / C_LIGHT

    return ells, clggs


if __name__ == "__main__":
    lmin, lmax = args.lmm[0], args.lmm[1]
    z1, z2 = args.z1z2[0], args.z1z2[1]

    # cosmological parameters
    H0, Omm0, Omb0, Tcmb0 = args.H0, args.Omm0, args.Omb0, args.Tcmb0
    ns, As = args.ns, args.As
    bkg, bgg = args.bkg, args.bgg
    h = H0 / 100.
    Omc0 = Omm0 - Omb0

    cosmo = cosmoLCDM(H0, Omm0, 1.-Omm0)

    # set up cosmology for CAMB
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Omb0*h**2, omch2=Omc0*h**2, TCMB=Tcmb0)
    pars.InitPower.set_params(As=As, ns=ns)
    # generate PK
    kmax = (lmax + 100) / cosmo.z2chi(z1)
    pk = gen_pk(pars, kmax, z1, z2)
    print('>> Matter power spectrum function generated with CAMB.')

    # generate fg(z)
    print('>> Loading redshift data: {}'.format(args.dataz))
    dataz = np.loadtxt(args.dataz)
    fg = gen_fg_z(dataz, z1, z2)
    print('>> Galaxy redshift distribution function generated.')

    if args.cal == 'clgg':
        print('>> Calculating clgg...')
        ell, cl = clgg(lmin, lmax, cosmo, z1, z2, fg, pk, bgg)
    elif args.cal == 'clkg':
        print('>> Calculating clkg...')
        ell, cl = clkg(lmin, lmax, cosmo, z1, z2, fg, pk, bkg)
    else:
        sys.exit('>> Set -cal to one of [clgg, clkg].')

    data = np.column_stack((ell, cl))
    header = 'ell   cl'
    np.savetxt(args.fo, data, fmt='%6d %15.7e', header=header)
    print(':: Written to file: {}'.format(args.fo))
