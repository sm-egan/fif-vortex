# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:00:45 2022

@author: shann
"""
import numpy as np
from scipy.constants import pi
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import os
from fif_aligned_lattice import *
from scipy.interpolate import interp1d
import matplotlib.cm as cm

matplotlib.rcParams["mathtext.rm"] = 'serif'
matplotlib.rcParams["mathtext.fontset"] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Palatino'

# set the font globally
plt.rcParams.update({'font.family':'serif'})

def get_ldos(evals, evecs, sites, wrange = None):
    '''
    Parameters
    ----------
    evals : numpy.ndarray
        1d array containing energy eigenvalues of real space Hamiltonian.
    evecs : numpy.ndarray
        2d array containing energy eigenstates.  evecs[:,i] is the eigenvectors for evals[i]
    sites : numpy.ndarray
        Lattice sites in real space where we want to compute the LDOS.
    dw : float
        fixed interval for size of "bins" in omega.  Only used if wrange not given.
    wrange : numpy.ndarray, optional
        array of "bins" in positive omega over which to compute LDOS. The default is None.

    Returns
    -------
    numpy.ndarray
        Array same size as wrange which gives spectral density for each omega at the chosen sites.

    '''
    N = len(evals) // 2
    if wrange is None:
        emin = np.min(evals)
        emax = np.max(evals)
        wrange = np.linspace(emin, emax, int((emax - emin)/0.1), endpoint=False)
    
    #subset of eigenvector amplitudes for the chosen sites
    evecs_sub = evecs[sites]
    evals = evals
    print('Length of evecs_sub is {}'.format(len(evecs_sub)))
    
    ldos = []
    
    for wind in range(0, len(wrange) - 1):
        inrange = (evals >= wrange[wind])*(evals < wrange[wind + 1])
        if np.sum(inrange) > 0:
            # If we are computing the ldos for multiple sites
            if evecs_sub.ndim > 1:
                # Computes the average contribution to the ldos for all sites and adds it to the array
                ldos.append(np.sum(np.abs(evecs_sub[:, inrange])**2)/len(sites))
            else:
                ldos.append(np.sum(np.abs(evecs_sub[inrange])**2))
        else:
            ldos.append(0)
    
    #plt.plot(wrange, ldos)
    return wrange, np.array(ldos)

def est_gap_size(evals, evecs, sites, dw, wrange = None):
    omegas, ldos = get_ldos(evals, evecs, sites, dw, wrange)
    gapstates = np.where(ldos < 1e-5)[0]
    print(gapstates)
    return omegas[gapstates[-1]] - omegas[gapstates[0]]

if __name__ == '__main__':
    mu = 0.8
    R = 32
    dw1 = 0.238*2/100
    wmax1 = 0.238
    dw2 = dw1
    wmax2 = 3
    wrange = np.append(np.linspace(-wmax2, -wmax1, int((wmax2-wmax1)/dw2), endpoint=False), np.linspace(-wmax1,wmax1,int(2*wmax1/dw1)))
    wrange = np.append(wrange, np.linspace(wmax1 + dw2, wmax2 + dw2, int((wmax2-wmax1)/dw2)))
    
    filename = 'data/fif_eigh_vortex_R{0}_mu{1:.2f}'.format(R, mu).replace('.','') + '.npz'
    infile = np.load(filename)
    evals = infile['arr_0']
    evecs = infile['arr_1']
    
    N = len(evals)//2
    title1 = r'$N =$ {0}, $\mu=$ {1}'.format(N, mu)
    savename = filename.split('/')[1].split('.npz')[0]
    if 'vortex' in filename:
        savename = savename + 'vortex'
    
    lattice = make_lattice(R)
    lattnn1 = find_nn(lattice)
    lattnn2 = find_nn(lattice, 2)
    
    origin = len(lattice) // 2
    core_sites = lattnn1[origin]
    edge_sites = np.where(np.invert(np.all(lattnn1 > -1, axis = 1)))[0]
    corner_sites = np.where(np.sum(lattnn1 > -1, axis = 1) == 3)[0]
    
    wrangec, ldosc = get_ldos(evals, evecs, core_sites, wrange)
    wrangee, ldose = get_ldos(evals, evecs, corner_sites, wrange)
    wrange3, ldos3 = get_ldos(evals, evecs, np.arange(0,3169), wrange)
    
    ###############################################
    ########### Plot ldos with subplots ###########
    
    fig, (axc, axe) = plt.subplots(ncols = 2, sharey=True, figsize = (10, 6))
    axc.axhline(0, c='k', linestyle='--', linewidth = 1)
    axc.plot(wrangec[0:len(ldosc)], ldosc, color = cm.gnuplot2(0.45), linewidth = 2, solid_capstyle='round')
    axc.axvspan(-0.238, 0.238, color='grey', alpha=0.4)
    
    axc.set_xlabel(r'$\omega$', size = 'xx-large')
    axc.set_ylabel(r'$\rho(\omega, \mathbf{r})$', size = 'xx-large')
    axc.set_xlim(-wmax2, wmax2)
    axc.set_title('LDOS near origin')
    
    axe.axhline(0, c='k', linestyle='--', linewidth = 0.7)
    axe.plot(wrangee[0:len(ldose)], ldose, color = cm.gnuplot2(0.4), linewidth = 2)
    axe.axvspan(-0.238, 0.238, color='grey', alpha=0.4, label = 'Bulk gap')
    
    axe.set_xlim(-wmax2, wmax2)
    axe.set_title('LDOS at edge')
    axe.set_xlabel(r'$\omega$', size = 'xx-large')
    
    if 'vortex' in savename:
        fig.suptitle('With vortex', size = 'xx-large')
    else:
        fig.suptitle('Without vortex', size = 'xx-large')
    fig.tight_layout()
    plt.show()
    
    #################################################
    ###### ldos plot with both edge and centre ######
    fig, ax = plt.subplots(figsize = (14,7))
    ax.axhline(0, c='k', linestyle='--', linewidth = 1)
    ax.axvspan(-0.238, 0.238, color='grey', alpha=0.4, label = 'Bulk gap')
    #ax.plot(wrangee[0:len(ldose)], ldose, color = cm.gnuplot2(0.65), linewidth = 4, linestyle = '--', label = 'Edge')
    #ax.plot(wrangec[0:len(ldosc)], ldosc, color = cm.gnuplot2(0.2), linewidth = 4, label='Vortex core')
    ax.plot(wrange3[0:len(ldos3)], ldos3, color = cm.gnuplot2(0.5), linewidth = 4, label='All sites')

    ax.set_xlabel(r'$\omega$', fontsize = 40)
    ax.set_ylabel(r'$\rho(\omega, \mathbf{r})$', fontsize=40)
    
    ax.tick_params(axis='both', which='major', labelsize=25)
    
    ax.set_xlim(-2.5,2.5)
    ax.set_ylim(0,0.002)
    ax.legend(fontsize = 25)
    fig.tight_layout()
    #fig.savefig('plots/LDOS.pdf')
    plt.show()
    
    ############################################
    ##### Replot ldos with spline ##############
    
    # splinec = interp1d(wrange[0:len(ldosc)], ldosc, kind = 'cubic')
    # splinee = interp1d(wrange[0:len(ldose)], ldose, kind = 'cubic')
    
    # xspline = np.linspace(-wmax2, wmax2, 1000)
    # ysplinec = splinec(xspline)
    # ysplinee = splinee(xspline)
    
    # fig, (axc, axe) = plt.subplots(ncols = 2, figsize = (10, 6))
    # axc.plot(xspline, ysplinec)
    # axc.set_xlabel(r'$\omega$', size = 'xx-large')
    # axc.set_ylabel(r'$\rho(\omega, \mathbf{r})$', size = 'xx-large')
    # #axc.set_xlim(-4,4)
    # axc.set_title('LDOS near origin')
    
    # axe.plot(xspline, ysplinee)
    # #axe.set_xlim(-4, 4)
    # axe.set_title('LDOS at edge')
    # axe.set_xlabel(r'$\omega$', size = 'xx-large')
    
    # if 'vortex' in savename:
    #     fig.suptitle('With vortex', size = 'xx-large')
    # else:
    #     fig.suptitle('Without vortex', size = 'xx-large')
    # fig.tight_layout()
    
    #############################################
    ########### Plot eigenvalues ################
    
    # zero_evals = np.where(np.abs(evals) < 0.03)[0]
    
    # erange = 0.45
    # einrange = evals[np.abs(evals) < erange]
    # eindex = np.arange(0, len(evals))[np.abs(evals) < erange]
    
    # fig, ax = plt.subplots(figsize = (10,4))
    # ax.plot(eindex[np.abs(einrange) > 0.03], einrange[np.abs(einrange) > 0.03], 'o')
    # ax.plot(zero_evals, evals[zero_evals], 'x', c='r', label = r'$|E| < 0.03$')

    ##############################################
    ############## Plot the MZMs #################
    
    zmodes = np.argsort(np.abs(evals))[0:2]
    
    mzm1 = np.abs((evecs[:, zmodes[0]][N:] + evecs[:, zmodes[1]][N:])/np.sqrt(2))**2
    mzm2 = np.abs((evecs[:, zmodes[0]][N:] - evecs[:, zmodes[1]][N:])/np.sqrt(2))**2
    
    msize = 12
    x = []
    y = []
    for row in lattice:
        x.append(LatticeVec(row, 'tri').get_x())
        y.append(LatticeVec(row, 'tri').get_y())
    
    plot_grid(lattice, colour = mzm1/np.max(mzm1), pointsize = msize, norm = 1/2, show_labels = False, savename = 'MZM_edge.svg')
    plt.show()
    plot_grid(lattice, colour = mzm2/np.max(mzm2), pointsize = msize, norm = 1/2, show_labels = False, savename = 'MZM_core.svg')
    plt.show()
    # ax.set_xlabel('Eigenstate index', size='x-large')
    # ax.set_ylabel('Energy', size='xx-large')
    # ax.set_title(title1, size = 'x-large')
    # ax.legend(loc = 2, prop = {'size': 14})
    # plt.show()
    
    
    #if len(zero_evals) < 12: