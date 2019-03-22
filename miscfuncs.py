'''
Some misc functions.
'''
import numpy as np
import matplotlib.pyplot as plt


def plot_cell(fn):
    '''Plot Cl.'''
    print('>> Plotting Cl : {}'.format(fn))
    data = np.loadtxt(fn)
    plt.errorbar(data[:, 0], data[:, 1], xerr=data[:, 2], fmt='.', capsize=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
