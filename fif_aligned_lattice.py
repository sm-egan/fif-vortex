# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:08:40 2022

@author: shann
"""
import numpy as np
from scipy.constants import pi
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os.path
import time
import warnings

matplotlib.rcParams["mathtext.rm"] = 'serif'
matplotlib.rcParams["mathtext.fontset"] = 'cm'

R = 2
paulis = np.array([
        np.array([[1+0j,0+0j],[0+0j,1+0j]]),
        np.array([[0+0j,1+0j],[1+0j,0+0j]]),
        np.array([[0+0j,0-1j],[0+1j,0j]]),
        np.array([[1+0j,0+0j],[0+0j,-1+0j]])
        ])
basis = np.array([[1, 0], [-1/2, np.sqrt(3)/2]])

class LatticeVec:
    def __init__(self, point = np.array([0.,0.]), grid = 'tri'):
        # Gcoords is a two-element array which specifies the coefficient of GM1, GM2, respectively
        self.coords = point
        # Defines two basis vectors in x,y coordinates
        if isinstance(grid, np.ndarray):
            self.basis = grid
        elif isinstance(grid, str):
            if grid == 'square':
                # Default is the Moire reciprocal vectors GM1 and GM2
                self.basis = np.array([[1, 0], [0, 1]])
            elif grid == 'tri':
                self.basis = np.array([[1, 0], [-1/2, np.sqrt(3)/2]])
            else: 
                self.basis = None
                raise Exception('No basis defined for the LatticeVec')
    
    # The add and subtract can handle any LatticeVecs regardless of whether they have the same basis
    # The output is a LatticeVec with the same basis as the first vector (i.e. if A+B, return will be in same basis as A)
    def __add__(self, kprime):
        if np.all(self.basis == kprime.basis):
            return LatticeVec(self.coords + kprime.coords)
        else:
            ksum = np.array([self.get_x() + kprime.get_x(), self.get_y() + kprime.get_y()])
            #print(ksum)
            
            n = (ksum[0]*self.basis[1][1] - ksum[1]*self.basis[1][0])/(self.basis[0][0]*self.basis[1][1] - self.basis[0][1]*self.basis[1][0])
            m = (ksum[0]*self.basis[0][1] - ksum[1]*self.basis[0][0])/(self.basis[0][1]*self.basis[1][0] - self.basis[0][0]*self.basis[1][1])
            
            #print('n = ' +str(n) + ' , m = ' + str(m))
            
            return LatticeVec(np.array([n,m]), self.basis)
    
    def __sub__(self, kprime):
        if np.all(self.basis == kprime.basis):
            return LatticeVec(self.coords - kprime.coords)
        else:
            kdif = np.array([self.get_x() - kprime.get_x(), self.get_y() - kprime.get_y()])
            #print(kdif)
            
            n = (kdif[0]*self.basis[1][1] - kdif[1]*self.basis[1][0])/(self.basis[0][0]*self.basis[1][1] - self.basis[0][1]*self.basis[1][0])
            m = (kdif[0]*self.basis[0][1] - kdif[1]*self.basis[0][0])/(self.basis[0][1]*self.basis[1][0] - self.basis[0][0]*self.basis[1][1])
            
            #print('n = ' +str(n) + ' , m = ' + str(m))
            
            return LatticeVec(np.array([n,m]), self.basis)
    
    def __mul__(self, scalar):
        #Default is multiplication by a number
        try:
            return LatticeVec(scalar*self.coords)
        except (TypeError, ValueError):
            print("LatticeVec objects can only be multiplied by a scalar or array with shape (2,)")
    
    __rmul__ = __mul__ 
        
    def __div__(self, scalar):
        try:
            return LatticeVec(self.coords/scalar)
        except (TypeError, ValueError):
            print("LatticeVec objects can only be divided by a scalar or array with shape (2,)")
                
    def get_x(self):
        return self.coords[0]*self.basis[0][0] + self.coords[1]*self.basis[1][0]
    
    def get_y(self):
        return self.coords[0]*self.basis[0][1] + self.coords[1]*self.basis[1][1]
    
    def get_len(self):
        return np.sqrt(self.get_x()**2 + self.get_y()**2)
    
    def dot(self, kprime):
        return self.get_x()*kprime.get_x() + self.get_y()*kprime.get_y()

#Initialize an array with all the (m,n) labels
def make_lattice(maxL):
    lim = maxL
    mesh = np.vstack((np.mgrid[-lim:lim+1, -lim:lim+1][0].flatten(), np.mgrid[-lim:lim+1, -lim:lim+1][1].flatten())).T
    
    opposites = np.sum(np.sign(mesh), axis=1) == 0
    abssum = np.sum(np.abs(mesh), axis=1) > maxL
    selection = np.invert(opposites*abssum)
    
    return mesh[selection]

nn1 = np.array([[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]])
nn2 = np.array([[2,1],[1,2],[-1,1],[-2,-1],[-1,-2],[1,-1]])

def find_nn(lattice, deg = 1):
    nnarr = np.full((len(lattice), 6), None)
    ind = None
    if deg == 1:
        nn = nn1
    elif deg == 2:
        nn = nn2
        
    for lattind in range(0, len(lattice)):
        # If all the values have already been set, skip to next row
        if np.all(nnarr[lattind] != None):
            continue
        
        for nnind in range(0,6):
            ind = np.where(np.all(lattice == lattice[lattind] + nn[nnind], axis=1))[0]
            if ind.size > 0:
                nnarr[lattind, nnind] = ind[0]
                
                # If match comes later in loop, update its info too
                if ind[0] > lattind:
                    nnarr[ind[0], (nnind +  3) % 6] = lattind
            else:
                # If no match found, set index to N (invalid)
                nnarr[lattind, nnind] = -1
    return nnarr.astype(int)

def N_unitcells(r):
    return 1 + 3*r*(r+1)

def A_hex(r):
    return 3*np.sqrt(3)/2*r**2

def H_diag(mu):
    N = len(lattice)
    block = sp.block_diag((-mu,)*N)
    return sp.kron(paulis[3], block)

def H_hop(t):
    N = len(lattice)
    hops = np.array([0,2,4])
    indlist = np.array([])
    indptr = [0]
    
    for i in range(0, N):
        nninds = lattnn1[i][hops]
        # print(nninds)
        nninds = nninds[nninds > -1]
        # print(nninds)
        indlist = np.append(indlist, nninds)
        indptr.append(indptr[-1] + len(nninds))
    
    # print(indlist)
    #print(indptr)
    data = np.array([-t]*len(indlist))
    
    #print('H_hop indlist has {} elements'.format(len(indlist)))
    halfmat = sp.csr_matrix((data, indlist, indptr), shape = (N, N))
    return sp.kron(paulis[3], halfmat + halfmat.transpose())

def H_sc1(Delta1, vortex = -1, vorticity =  1, plot = False):
    # Bonds with +1 order param
    bonds = np.array([0,2,4])
    indlist = np.array([])
    if vortex > -1:
        print('Adding vortex to H_sc2')
        vplist = np.array([])
    indptr = [0]
    N = len(lattice)
    
    for i in range(0, N):
        nninds = lattnn1[i][bonds]
        nninds = nninds[nninds > -1]
        
        if vortex > -1:
            vps = vortex_phase(vortex, i, nninds, vorticity)
            vplist = np.append(vplist, vps)
        
        indlist = np.append(indlist, nninds)
        indptr.append(indptr[-1] + len(nninds))
    
    if vortex > -1:
        data = -Delta1*vplist
    else:
        data = np.array([-Delta1]*len(indlist))
    #print('H_sc1 indlist has {} elements'.format(len(indlist)))
    halfmat = sp.csr_matrix((data, indlist, indptr), shape = (N, N))
    halfmat = halfmat - halfmat.transpose() 
    if plot:
        plt.pcolor(np.real(halfmat.toarray()))
        plt.title(r'Re $H_{SC1}$')
        plt.show()
        plt.pcolor(np.imag(halfmat.toarray()))
        plt.title(r'Im $H_{SC1}$')
        plt.show()       
        
    '''
    Used this method to put blocks together before I learned about sp.bmat
    bmat is 30-50% faster
    
    #data2 = np.array([halfmat.toarray(), halfmat.getH().toarray()])
    #return sp.bsr_matrix((data2, np.array([1,0]), np.array([0,1,2])), shape=(2*N, 2*N))
    '''
    return sp.bmat([[None, halfmat], [halfmat.getH(), None]]).tocsr()

def H_sc2(Delta2, vortex = -1, vorticity = 1, plot = False):
    # Bonds with +i order param
    bonds = np.array([1,3,5])
    indlist = np.array([])
    if vortex > -1:
        print('Adding vortex to H_sc2')
        vplist = np.array([])
    indptr = [0]
    N = len(lattice)
    
    for i in range(0, N):
        nninds = lattnn2[i][bonds]
        nninds = nninds[nninds > -1]
        
        if vortex > -1:
            vps = vortex_phase(vortex, i, nninds, vorticity)
            vplist = np.append(vplist, vps)
        
        indlist = np.append(indlist, nninds)
        indptr.append(indptr[-1] + len(nninds))
    
    if vortex > -1:
        data = -Delta2*1j*vplist
    else:
        data = np.array([-(Delta2*1j)]*len(indlist))
    #print('H_sc2 indlist has {} elements'.format(len(indlist)))
    halfmat = sp.csr_matrix((data, indlist, indptr), shape = (N, N))
    halfmat = halfmat - halfmat.transpose()
    
    if plot:
        plt.pcolor(np.real(halfmat.toarray()))
        plt.title(r'Re $H_{SC1}$')
        plt.show()
        plt.pcolor(np.imag(halfmat.toarray()))
        plt.title(r'Im $H_{SC1}$')
        plt.show()  
        
    #data2 = np.array([halfmat.toarray(), halfmat.getH().toarray()])
    #return sp.bsr_matrix((data2, np.array([1,0]), np.array([0,1,2])), shape=(2*N, 2*N))
    return sp.bmat([[None, halfmat], [halfmat.getH(), None]]).tocsr()
    
    # block = -(Delta2/2)*1j*paulis[1]
    # data = np.array([block]*len(indlist))
    # print('H_sc1 indlist has {} elements'.format(len(indlist)))
    # halfmat = sp.bsr_matrix((data, indlist, indptr), shape = (2*N, 2*N))
    # return halfmat + np.conj(halfmat.transpose())

def H_fif(mu, t, Delta1, Delta2, vortex = -1, vorticity = 1):
    start = time.time()
    sc1 = H_sc1(Delta1, vortex, vorticity)
    stop = time.time()
    print('H_sc1 took {0:5f} seconds to build'.format(stop - start))
    
    start = time.time()
    sc2 = H_sc2(Delta2, vortex, vorticity)
    stop = time.time()
    print('H_sc2 too {0:5f} seconds to build'.format(stop-start))
    return H_diag(mu) + H_hop(t) + sc1 + sc2

def vortex_phase(vsite, ind, neighbours, vorticity = 1):
    bondmid = (lattice[ind] + lattice[neighbours])/2
    distvec = bondmid - lattice[vsite]
    #print(distvec)
    
    distx = distvec[:, 0]*basis[0,0] + distvec[:, 1]*basis[1,0]
    disty = distvec[:, 0]*basis[0,1] + distvec[:,1]*basis[1,1]
    
    phases = np.exp(1j*vorticity*np.arctan2(disty, distx))
    '''
    Goal:
        Return an array the same length as neighbours which gives the phase multiplier
        for each bond due to the vortex.
    Next steps:
        1. Calculate the angle of the bond centre relative to the vortex centre
            a. Maybe make an auxiliary function called get_angle() to do this?
        2. Once you have the above, simply insert it into the complex exponential and return ?
      
    '''
    return phases
    
        
# def vortex_phase(vsite, vorticity, raw = True):
#     '''
#     Creates sparse matrix which can be multiplied by the Hamiltonian to give
#     phase winding on the order parameter

#     Parameters
#     ----------
#     vsite : int
#         Lattice site where vortex appears.
#     vorticity : int
#         Number of times the phase winds around the vortex.

#     Returns
#     -------
#     None.

#     '''
#     N = len(lattice)
#     bonds = np.arange(0,6)
#     phase1 = np.exp(1j*vorticity*pi/3*bonds)
#     phase2 = np.exp(1j*vorticity*pi*(1/3*bonds + 1/6))
    
#     row1 = np.array([vsite]*6)
#     row2 = np.array([vsite]*6)
    
#     col1 = lattnn1[vsite]
#     col2 = lattnn2[vsite]
    
#     data1 = phase1 
#     data2 = phase2 
#     if not raw:
#         data1 -= 1
#         data2 -= 1
#     upper1 = sp.csr_matrix((data1, (row1, col1)), shape = (N, N))
#     upper1 = upper1 + upper1.transpose()
    
#     upper2 = sp.csr_matrix((data2, (row2, col2)), shape = (N, N))
#     upper2 = upper2 + upper2.transpose()
    
#     halfmat = upper1 + upper2
    
#     return sp.bmat([[None, halfmat],[halfmat.getH(), None]]).tocsr()

def plot_grid(mesh, colour = None, grid = 'tri', title='', clims = None, savename = None, pointsize = None, cmap = 'YlGnBu', norm = None, show_labels = True):
    xlist = []
    ylist = []
    
    for row in mesh:
        xlist.append(LatticeVec(row, grid).get_x())
        ylist.append(LatticeVec(row, grid).get_y())
    
    if grid == 'tri':
        mark = 'h'
    elif grid == 'FBZ':
        mark = 'H'
    else:
        mark = 'o'
        
    if pointsize is not None:
        size = pointsize
    else:
        # Change size in proportion to R=8 case, which for a (9,8) size figure looks best wiht pointsize=700 (FBZ case)
        size = 70
    if colour is None:
        plt.scatter(xlist, ylist, s=size, marker=mark)
    else:
        if np.any(colour < 0) and norm == 'log':
            warnings.warn('WARNING: log scale does not work with negative values.  Linear scale will be used')
            norm = None
        if clims == None:
            minmax = np.max(colour)
            if norm is None:
                plt.scatter(xlist, ylist, c = colour, cmap = cmap, s = size, marker = mark)
            elif norm == 'log':
                plt.scatter(xlist, ylist, c = colour, norm=colors.LogNorm(vmin = colour.min(), vmax = colour.max()), cmap = cmap, s = size, marker = mark)
            elif type(norm) is float:
                plt.scatter(xlist, ylist, c = colour, norm=colors.PowerNorm(norm), cmap = cmap, s = size, marker = mark)
            #plt.pcolormesh(xlist, ylist, colour, cmap = cmap, shading = 'nearest')
        else:
            plt.scatter(xlist, ylist, c = colour, cmap = 'YlGnBu', s = size, marker = mark)
           
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize = 15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show_labels:
        plt.xlabel(r'$x$', size='xx-large')
        plt.ylabel(r'$y$', size='xx-large')
    else:
        plt.tick_params(left = False, labelleft = False , labelbottom = False, bottom = False)
    plt.title(title, size='xx-large')
    
    if savename is not None:
        print('Saving figure')
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/' + savename, bbox_inches = 'tight')
        
def is_hermitian(A):
    return not np.any((A != A.getH()).toarray())

def PH_exchange(A):
    P = sp.kron(paulis[1], sp.identity(N))
    return P * A.conjugate() * P

def Pdiff(A):
    return A + PH_exchange(A)
    
if __name__ == '__main__':
    #plot_grid(lattice, pointsize = 500)
    R = 32
    k = 2**5
    t = 1
    mu = 0.8
    Delta1 = 0.5*t
    Delta2 = Delta1/(3*np.sqrt(3))
    use_sparse = False
    vorticity = 1
    
    saveplots = False
    
    lattice = make_lattice(R)
    lattnn1 = find_nn(lattice)
    lattnn2 = find_nn(lattice, 2)
    N = len(lattice)
    msize = 12*A_hex(32)/A_hex(R)
    
    if vorticity is not None:
        vsite = len(lattice) // 2
        H = H_fif(t, mu, Delta1, Delta2, vsite, vorticity)
    else:
        vsite = -1
        H = H_fif(t, mu, Delta1, Delta2, vsite)
        
        
    
    #title1 = r'$R =$ {0}, $N =$ {1}, $\mu=$ {2}'.format(R, N, mu)
    title1 = r'$N =${0}, $\mu=${1}'.format(N, mu)
    if use_sparse:
        savename = 'fif_eigsh'
        evals, evecs = eigsh(H, k = k, which='SM')
        evals = np.sort(evals)
        title1 = title1 + r', $k =$ {}'.format(k)
    else:
        #Takes about 13 minutes for R = 32
        savename = 'fif_eigh'
        evals, evecs = eigh(H.toarray())
    if vorticity is not None:
        if vorticity > 0:
            savename = savename + '_vortex'
        elif vorticity < 0:
            savename = savename + '_antivortex'
    np.savez('data/' + savename + '_R{0}_mu{1:.2f}'.format(R, mu).replace('.',''), evals, evecs)
    
    zero_evals = np.where(np.abs(evals) < 0.05)[0]

    for ind in zero_evals:
        state = evecs[:, ind][N:]
        amp = np.abs(state)**2
        norm = np.sum(amp)
        title2 = title1 + ', $E =${0:.4f}'.format(evals[ind])
        
        if saveplots:
            plot_grid(lattice, colour = amp/norm, pointsize = msize, title = title2, savename = savename + '_ind{}'.format(ind) + '.pdf')
        else:
            plot_grid(lattice, colour = amp/norm, pointsize = msize, title = title2)
    
    # Isolate the states closest to zero
    zmodes = np.argsort(np.abs(evals))[0:2]
    for ind in zmodes:
        state = evecs[:, ind][N:]
        title2 = title1 + ', $E =${0:.4f}'.format(evals[ind])
        #title2 = title1 + ', $E =$ {0:.4f}'.format(evals[ind])

    zmodes1 = (evecs[:, zmodes[0]][N:] + evecs[:, zmodes[1]][N:])/np.sqrt(2)
    zmodes2 = (evecs[:, zmodes[0]][N:] - evecs[:, zmodes[1]][N:])/np.sqrt(2)
    
    amp1 = np.abs(zmodes1)**2
    amp2 = np.abs(zmodes2)**2
    
    plot_grid(np.delete(lattice, vsite, axis = 0), colour = np.delete(amp1, vsite), pointsize = msize, title = title2, norm = 'log')
    plot_grid(lattice, colour = amp1, pointsize = msize, title = title2)
    plot_grid(np.delete(lattice, vsite, axis = 0), colour = np.delete(amp2, vsite), pointsize = msize, title = title2, norm = 'log')
    plot_grid(lattice, colour = amp2, pointsize = msize, title = title2)
    
    # spectral_rev = plt.cm.get_cmap('Spectral').reversed()
    
    # plot_grid(lattice, colour = np.real(zmodes1), pointsize = msize, title = title2, cmap = spectral_rev)
    # plot_grid(lattice, colour = np.imag(zmodes1), pointsize = msize, title = title2, cmap = spectral_rev)
    
    # plot_grid(lattice, colour = np.real(zmodes2), pointsize = msize, title = title2, cmap = spectral_rev)
    # plot_grid(lattice, colour = np.imag(zmodes2), pointsize = msize, title = title2, cmap = spectral_rev)