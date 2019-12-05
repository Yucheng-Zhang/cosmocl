# cosmocl

## estimation of angular power spectra

- source: `estcl.py`
- estimator: `PyMaster` or `Healpy`
- Each field has map and mask, which can be updated separately.
- Mode coupling matrix can be saved to save time if masks not changed.

## theoretical alculation of angular power spectra

- source: `theocl.py`
- example usage:
```python
import numpy as np
from cosmopy.cosmoLCDM import cosmoLCDM
from cosmopy.utils import gen_fg_z
from cosmocl.theocl import theocls

# cosmological parameters
H0 = 67.66
Om0 = 0.3111
Ob0 = 0.04897
Tcmb0 = 2.7255
As = 2.105e-9
ns = 0.9665

# linear bias
def bias(z):
#    return 2.233021
    return 0.53 + 0.29 * np.power(1+z, 2)

# redshift distribution from all redshifts
dataz = np.loadtxt('../cata/dataz.dat')
z1, z2 = 0.8, 2.2 # redshift range of the galaxy sample
fg = gen_fg_z(dataz, z1, z2, bins=28)

cosmo = cosmoLCDM(H0, Om0, Ob0, Tcmb0, As, ns)

ells = np.arange(2, 1101, 1)

cls = theocls(ells, z1, z2, fg, bias, cosmo)
cls.cosmo.gen_interp_chiz(zmin=0, zmax=z2+0.1, dz=0.001)
cls.set_interp_pk(zmin=z1-0.1, zmax=z2+0.1, kmax=10)

clkg = cls.c_clkg()

np.savetxt('theo_clkg.dat', np.column_stack((ells, clkg)),
             fmt='%6d %15.7e', header='ell   C_l^kg')
```