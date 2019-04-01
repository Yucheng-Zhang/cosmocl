'''
Estimate Cl.
'''
import argparse
import sys
from estnmt import main_master
from esthp import main_hp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Estimate Cl.')

    parser.add_argument('-est', type=str, default='nmt',
                        help='Estimator: nmt for NaMaster, \
                            hp for simple estimation with healpy')
    parser.add_argument('-tp', type=str, default='auto',
                        help='auto or cross correlation.')
    parser.add_argument('-fb', type=str, default='fb.dat',
                        help='Data file which specifies the bins. \
                            Two columns [lmin, lmax], closed on both sides.')

    parser.add_argument('-nside', type=int, default=1024,
                        help='Nside of the masks and maps.')

    parser.add_argument('-mask1', type=str, default='',
                        help='Input mask file 1.')
    parser.add_argument('-map1', type=str, default='',
                        help='Input map file 1.')
    parser.add_argument('-alm1', type=str, default='',
                        help='Input alm file 1.')
    parser.add_argument('-fwhm1', type=float, default=-1,
                        help='Full Width Half Max of the Gaussian beam \
                            [in degree] for emission map.')

    parser.add_argument('-mask2', type=str, default='',
                        help='Input mask file 2.')
    parser.add_argument('-map2', type=str, default='',
                        help='Input map file 2.')
    parser.add_argument('-alm2', type=str, default='',
                        help='Input alm file 2.')
    parser.add_argument('-fwhm2', type=float, default=-1,
                        help='Full Width Half Max of the Gaussian beam \
                            [in degree] for emission map.')

    #--- output related ---#
    parser.add_argument('-focl', type=str, default='out-cl.dat',
                        help='Output ell,cl file name.')

    #--- for PyMaster estimation ---#
    parser.add_argument('-savewsp', type=int, default=0,
                        help='Save the workspace or not, which might be very large.')
    parser.add_argument('-fwsp', type=str, default='',
                        help='Workspace file name.')

    #--- for simple HealPy estimation ---#
    parser.add_argument('-fsky', type=float, default=1.0,
                        help='fsky if estimated with healpy.')

    args = parser.parse_args()


if __name__ == "__main__":
    '''Main function.'''
    if args.est == 'nmt':
        main_master(args)

    elif args.est == 'hp':
        main_hp(args)

    else:
        sys.exit('Wrong estimator.')
