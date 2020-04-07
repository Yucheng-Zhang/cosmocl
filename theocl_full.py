'''
Theoretical computation of C_ell
w/o using Limber approximation.
'''
import numpy as np
import scipy.integrate as spint
import multiprocessing as mp
from joblib import Parallel, delayed
import sys
import time

