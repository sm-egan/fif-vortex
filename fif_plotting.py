# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:00:45 2022

@author: shann
"""
import numpy as np
from scipy.constants import pi
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
from fif_aligned_lattice import make_lattice, plot_grid

def get_ldos(evals, evecs, sites, dw, wrange = None):
    N = len(evals)
    if wrange is None:
        emin = np.min(evals)
        emax = np.max(evals)
        wrange = np.linspace(emin, emax, int((emax - emin)/dw), endpoint=False)
    
    #print(wrange)
    #evals_sub = evals
    evecs_sub = evecs[sites]
    
    ldos = []
    
    for w in wrange:
        inrange = (evals > w)*(evals < w + dw)
        if np.sum(inrange) > 0:
            if evecs_sub.ndim > 1:
                ldos.append(np.sum(np.abs(evecs_sub[:, inrange])**2, axis = 0))
            else:
                ldos.append(np.sum(np.abs(evecs_sub[inrange])**2))
        else:
            ldos.append(0)
    
    #plt.plot(wrange, ldos)
    return wrange, np.array(ldos)

if __name__ == '__main__':
    mu = 0.9
    
    filename = 'data/fif_eigh_vortex_R32_mu{0:.2f}'.format(mu).replace('.','') + '.npz'
    infile = np.load(filename)
    evals = infile['arr_0']
    evecs = infile['arr_1']
    
    N = len(evals)//2
    title1 = r'$N =$ {0}, $\mu=$ {1}'.format(N, mu)
    savename = filename.split('/')[1].split('.npz')[0]
    if 'vortex' in filename:
        savename = savename + 'vortex'
    
    lattice = make_lattice(32)
    center = np.where(np.all(lattice == np.array([0,0]), axis = 1))[0][0]
    
    wrangec, ldosc = get_ldos(evals, evecs, center, 0.1)
    wrangee, ldose = get_ldos(evals, evecs, 0, 0.1)
    
    fig, (axc, axe) = plt.subplots(ncols = 2, figsize = (10, 6))
    axc.plot(wrangec, ldosc)
    axc.set_xlabel(r'$\omega$', size = 'xx-large')
    axc.set_ylabel(r'$\rho(\omega, \mathbf{r})$', size = 'xx-large')
    axc.set_xlim(-6,6)
    axc.set_title('LDOS at origin')
    
    axe.plot(wrangee, ldose)
    axe.set_xlim(-6, 6)
    axe.set_title('LDOS at edge')
    axe.set_xlabel(r'$\omega$', size = 'xx-large')
    
    if 'vortex' in savename:
        fig.suptitle('With vortex', size = 'xx-large')
    else:
        fig.suptitle('Without vortex', size = 'xx-large')
    fig.tight_layout()
    
    zero_evals = np.where(np.abs(evals) < 0.03)[0]
    
    fig, ax = plt.subplots(figsize = (4,10))
    ax.plot(np.arange(0, len(evals))[np.abs(evals) > 0.03], evals[np.abs(evals) > 0.03], 'o')
    ax.plot(zero_evals, evals[zero_evals], 'x', c='r', markersize = 12, label = r'$|E| < 0.03$')
    ax.set_xlim(3150, 3190)
    ax.set_ylim(-0.15,0.15)
    ax.set_xlabel('Eigenstate index', size='x-large')
    ax.set_ylabel('Energy', size='xx-large')
    ax.set_title(title1, size = 'x-large')
    ax.legend(loc = 2, prop = {'size': 14})
    plt.show()
    
    #if len(zero_evals) < 12: