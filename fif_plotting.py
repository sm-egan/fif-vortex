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
from fif_aligned_lattice import make_lattice

def get_ldos(evals, evecs, sites, dw, wrange = None):
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
    infile = np.load('fif_eigh_R32.npz')
    evals = infile['arr_0']
    evecs = infile['arr_1']
    
    lattice = make_lattice(32)
    center = np.where(np.all(lattice == np.array([0,0]), axis = 1))[0][0]
    
    wrange, ldos = get_ldos(evals, evecs, center + 20, 0.1)
    
    plt.plot(wrange, ldos)
    plt.xlabel()
    #plt.xlim(-1,1)
    
    
    
    