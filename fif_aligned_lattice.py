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
import matplotlib.pyplot as plt
import os.path

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
nn2 = np.array([[2,-1],[1,2],[-1,1],[-2,-1],[-1,-2],[1,-1]])

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
    
    print('H_hop indlist has {} elements'.format(len(indlist)))
    halfmat = sp.csr_matrix((data, indlist, indptr), shape = (N, N))
    return sp.kron(paulis[3], halfmat + halfmat.transpose())

def H_sc1(Delta1, plot = False):
    # Bonds with +1 order param
    bonds = np.array([0,2,4])
    indlist = np.array([])
    indptr = [0]
    N = len(lattice)
    
    for i in range(0, N):
        nninds = lattnn1[i][bonds]
        nninds = nninds[nninds > -1]
        
        indlist = np.append(indlist, nninds)
        indptr.append(indptr[-1] + len(nninds))
        
    data = np.array([-Delta1]*len(indlist))
    print('H_sc1 indlist has {} elements'.format(len(indlist)))
    halfmat = sp.csr_matrix((data, indlist, indptr), shape = (N, N))
    if plot:
        plt.pcolor(halfmat.toarray())
        plt.show()
    halfmat = halfmat - halfmat.transpose()
    if plot:
        plt.pcolor(halfmat.toarray())
        plt.show()
    data2 = np.array([halfmat.toarray(), halfmat.getH().toarray()])
    return sp.bsr_matrix((data2, np.array([1,0]), np.array([0,1,2])), shape=(2*N, 2*N))

def H_sc2(Delta2, plot = False):
    # Bonds with +i order param
    bonds = np.array([1,3,5])
    indlist = np.array([])
    indptr = [0]
    N = len(lattice)
    
    for i in range(0, N):
        nninds = lattnn2[i][bonds]
        nninds = nninds[nninds > -1]
        
        indlist = np.append(indlist, nninds)
        indptr.append(indptr[-1] + len(nninds))
    
        data = np.array([-Delta1]*len(indlist))
    
    data = np.array([-(Delta2*1j)]*len(indlist))
    print('H_sc2 indlist has {} elements'.format(len(indlist)))
    halfmat = sp.csr_matrix((data, indlist, indptr), shape = (N, N))
    if plot:
        plt.pcolor(np.imag(halfmat.toarray()))
        plt.show()
    halfmat = halfmat - halfmat.transpose()
    if plot:
        plt.pcolor(np.imag(halfmat.toarray()))
        plt.show()
    data2 = np.array([halfmat.toarray(), halfmat.getH().toarray()])
    return sp.bsr_matrix((data2, np.array([1,0]), np.array([0,1,2])), shape=(2*N, 2*N))
    
    # block = -(Delta2/2)*1j*paulis[1]
    # data = np.array([block]*len(indlist))
    # print('H_sc1 indlist has {} elements'.format(len(indlist)))
    # halfmat = sp.bsr_matrix((data, indlist, indptr), shape = (2*N, 2*N))
    # return halfmat + np.conj(halfmat.transpose())

def H_fif(mu, t, Delta1, Delta2):
    return H_diag(mu) + H_hop(t) + H_sc1(Delta1) + H_sc2(Delta2)

def plot_grid(mesh, colour = None, grid = 'tri', title='', clims = None, savename = None, pointsize=None):
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
        if clims == None:
            minmax = np.max(colour)
            
            plt.scatter(xlist, ylist, c=colour, cmap = 'YlGnBu', s=size, marker=mark, vmin=0, vmax=minmax)
        else:
            plt.scatter(xlist, ylist, c=colour, cmap = 'YlGnBu', s=size, marker=mark, vmin=clims[0], vmax=clims[1])
           
    plt.colorbar().set_label('Amplitude', size = 'x-large')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel(r'$x$', size='xx-large')
    plt.ylabel(r'$y$', size='xx-large')
    plt.title(title, size='xx-large')
    
    if savename is not None:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/' + savename)
    
    plt.show()
    
def is_hermitian(A):
    return not np.any((A != A.getH()).toarray())
    
if __name__ == '__main__':
    #plot_grid(lattice, pointsize = 500)
    R = 32
    k = 2**3
    t = 1
    mu = 1
    Delta1 = 0.2
    Delta2 = 0.1
    use_sparse = False
    
    lattice = make_lattice(R)
    lattnn1 = find_nn(lattice)
    lattnn2 = find_nn(lattice, 2)
    N = len(lattice)
    
    H = H_fif(t, mu, Delta1, Delta2)
    title1 = r'$R =$ {0}, $N =$ {1}'.format(R, N)
    if use_sparse:
        evals, evecs = eigsh(H, k = k, which='SM')
        evals = np.sort(evals)
        np.savez('fif_eigsh_R{0}_k{1}'.format(R, k), evals, evecs)
        title1 = title1 + r', $k =$ {}'.format(k)
    else:
        #Takes about 13 minutes for R = 32
        evals, evecs = eigh(H.toarray())
        np.savez('fif_eigh_R{0}'.format(R), evals, evecs)
    
    plt.plot(evals, 'o')
    #plt.ylim(-0.5, 0.5)
    plt.xlabel('Eigenstate index', size='x-large')
    plt.ylabel('Energy', size='x-large')
    plt.title(title1)
    plt.show()
    
    zero_evals = np.where(np.abs(evals) < 0.04)[0]
    if len(zero_evals) < 12:
        for ind in zero_evals:
            state = evecs[:, ind][0:N]
            amp = np.abs(state)
            norm = np.sum(amp**2)
            title2 = title1 + ', $E =$ {0:.2f}'.format(evals[ind])
            plot_grid(lattice, colour = amp/np.sqrt(norm), pointsize=10, title = title2)
