# cosmocl

- Code for theoretical calculation and estimation of the angular power spectra.
- Dependencies:
  - `numpy`, `scipy`, `h5py`
  - `tqdm`
    - https://github.com/tqdm/tqdm
    - `pip install tqdm`
  - `healpy`
    - https://github.com/healpy/healpy
    - `pip install healpy`
  - `pymaster`
    - https://github.com/LSSTDESC/NaMaster
    - `conda install -c conda-forge namaster`

---

## Theoretical calculation
- source: `theocl.py`
- See [this notebook](./demo/theo_cal.ipynb) for example calculations.

---

## Estimation
(need update)

- source: `estcl.py`
- estimator: `PyMaster` or `Healpy`
- Each field has map and mask, which can be updated separately.
- Mode coupling matrix can be saved to save time if masks not changed.
