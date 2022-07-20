# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:58:54 2022

@author: Pawel Janas (Python 3.8)

Student Number: 19318421

Description: Cubic Lattices for Point Generation
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def cluster_dist3D(L, x, y, z): # 3D plot
    """ Create a 3D plot of the galaxy distribution """
    plt.figure(dpi=150, figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, color='r', edgecolor='k', zorder=10)
    ax.set_title('Clustered Distribution', fontsize=15)
    
def get_coords(L, N):
    """
    Create arrays for randomly generated 3D coordinates.

    Parameters
    ----------
    L : float
        .
    N : int
        Number of points (galaxies) to generate coordinates for.

    Returns
    -------
    x : array-like
        1D x-coordinate array.
    y : array-like
        1D y-coordinate array.
    z : array-like
        1D z-coordinate array.

    """
    x = np.random.random(N) * L
    y = np.random.random(N) * L
    z = np.random.random(N) * L
    return x, y, z

def distances_to_each(N, data1, data2, mode):
    """N and arr must have the same length """
    if (mode == 'DD'): # data-data distance computation
        n_RR = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                n_RR[i,j] = math.dist(data1[i,:], data1[j,:])
        return n_RR
    
    elif (mode == 'DR'): # data random distance computation (or data1-data2)
        n_DR = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                n_DR[i,j] = math.dist(data1[i,:], data2[j,:])
        return n_DR
    
def remove_low_pairs(n_DD, rmax):
    """ Removes one of each pair if one of the points fall within rmax of the
    other. This algorithm is to be run on the random data to produce a negative
    correlation function xi(r) at $$r < r_{max}$$."""
    n = len(n_DD)
    min_dist_points = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (n_DD[i,j] < rmax):
                min_dist_points[i,j] = 1

    points_to_del = np.argwhere(min_dist_points == 1)[:,0]
    n_DD = np.delete(n_DD, points_to_del, axis=0)
    return n_DD

def xi_r_calc_alt(n_RR1, n_RR2):
    """ Calculates the correlation function estimator  """
    data1 = n_RR1.flatten()
    data2 = n_RR2.flatten()
    
    r = np.linspace(0, 1, len(data1)//10)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
            
    # handle div by 0...
    xi_r_binned = (binned_data1[0]+(binned_data1[0]==0) * \
                   binned_data2[0]+(binned_data2[0]==0)) / \
        (binned_data2[0]+(binned_data2[0]==0))**2 - 1

    return xi_r_binned

L = 1
N = 1000 # cubed number (choose from 8, 27, 64, 125, 216, 343, 512, 729, 1000)
random_data = np.transpose(get_coords(L, N))
xy = random_data[:,:2] # 2d rand data

# generate data on lattice
steps = int(np.cbrt(N))
x_ax = np.linspace(0, L, steps)
y_ax = np.linspace(0, L, steps)
z_ax = np.linspace(0, L, steps)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

for j in range(steps):
    z[j*steps**2:(j+1)*steps**2] = z_ax[j]
    
for i in range(steps): # fix x, generate y  
    x[i*steps:(i+1)*steps] = x_ax[i]

for i in range(steps): # gen y for each x
    y[i*steps:(i+1)*steps] = y_ax

for i in range(steps):
    x[i*steps**2:(i+1)*steps**2] = x[0:steps**2]
    y[i*steps**2:(i+1)*steps**2] = y[0:steps**2]
        
data = np.transpose([x, y, z])
   
n_DD = distances_to_each(N, data, data, 'DD')
n_DR = distances_to_each(N, data, random_data, 'DR')
n_DD = np.ma.masked_equal(n_DD, 0) # mask previous array to exclude when = 0
n_DDf = n_DD.flatten()

rmax = L / steps
plt.figure(dpi=150, figsize=(5,5))
plt.hist(n_DDf, bins=int(np.sqrt(N)))
plt.xlim(0,2)
plt.title(r'$n_{DD}$ Distribution')
plt.axvline(rmax, color='r', linestyle='--', label=r'$r_{max}$')
plt.legend(loc='best')
plt.tight_layout()

cluster_dist3D(L, x, y, z)

xi_r = xi_r_calc_alt(n_DD, n_DR)

r = np.linspace(0, 1, len(data)**2//10 - 1)
plt.figure(dpi=150)
plt.plot(r[1:], xi_r[1:])
plt.xlabel(r'$r$')
plt.ylabel(r'$\xi$')

N_1d = 100
x_1d = np.linspace(0, 1, N_1d)
n_DD1 = np.zeros((N_1d, N_1d))
for i in range(len(x_1d)):
    for j in range(len(x_1d)):
        n_DD1[i,j] = np.sqrt((x_1d[i]-x_1d[j])**2)

n_DR1 = np.zeros((N_1d, N_1d))
for i in range(N_1d):
    for j in range(N_1d):
        n_DR1[i,j] = np.sqrt((x_1d[i]-random_data[j,0])**2)

xi_r_1d = xi_r_calc_alt(n_DD1, n_DR1)

r1 = np.linspace(0, 1, N_1d**2//10 - 1)
plt.figure(dpi=150)
plt.plot(r1[1:], xi_r_1d[1:])
plt.title(r'One Dimensional Lattice')
plt.xlabel(r'$r$')
plt.ylabel(r'$\xi$')

