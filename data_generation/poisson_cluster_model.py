# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:36:55 2022

@author: Pawel Janas (Python 3.10.5)

Description: Data generation for Poisson Cluster Model
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def sample_spherical(npoints, radius, centre, ndim=3):
    """ Generate a vector from three standard normal distributions to avoid 
    'pole density'. Normalise the vector to have magnitude equal to radius
    desired. Also set the centre of the sphere in space as desired. """
    vec = np.random.randn(ndim, npoints) # select random points
    vec /= np.linalg.norm(vec, axis=0) # normalise vector
    vec *= radius # set vector length to desired radius
    vec[0,:] += centre[0] # translate sphere to centre coordinates
    vec[1,:] += centre[1]
    vec[2,:] += centre[2]
    return vec

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

def poisson_cluster_dist3D(L, x, y, z): # 3D plot
    """ Create a 3D plot of the Poisson distribution """
    plt.figure(dpi=150, figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, color='r', edgecolor='k', zorder=10)
    ax.set_title('Clustered Poisson Distribution', fontsize=15)

def xi_r_calc_alt(n_RR1, n_RR2):
    """ Calculates the correlation function estimator  """
    data1 = n_RR1.flatten()
    data2 = n_RR2.flatten()
    
    r = np.linspace(0, 1, len(data1)//100)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
    
    data1.sort() # sort n_DD arrays for full xi_r analyses
    data2.sort()
    
    xi_r_full = np.empty(len(data1)) # for all values
    for i in range(len(data1)):
        if (data2[i] == 0 or data1[i]==data2[i]):
            xi_r_full[i] = 0
        else:
            xi_r_full[i] = (data1[i] * data2[i]) / data2[i]**2 - 1
            
    # handle div by 0...
    xi_r_binned = (binned_data1[0]+(binned_data1[0]==0) * \
                   binned_data2[0]+(binned_data2[0]==0)) / \
        (binned_data2[0]+(binned_data2[0]==0))**2 - 1

    return xi_r_binned, xi_r_full

def distances_to_each(N, data1, data2, mode):
    """N and arr must have the same length """
    if (mode == 'DD'): # data-data distance computation
        n_RR = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                n_RR[i,j] = math.dist(data1[i,:], data1[j,:])
        return n_RR
    
    elif (mode == 'DR'): # data random distance computation
        n_DR = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                n_DR[i,j] = math.dist(data1[i,:], data2[j,:])
        return n_DR

def file_write(x, y, z, file_name):
    """ Function for writing data to a .txt file. """
    out_file = open(file_name, 'w') # open file for writing
    # write values to file
    print('{0:>15}'.format('x'),
          '{0:>15}'.format('y'),
          '{0:>15}'.format('z'), file=out_file)
    for i in range(len(x)):
        print('{0:>15.7g}'.format(x[i]),
              '{0:>15.7g}'.format(y[i]), 
              '{0:>15.7g}'.format(z[i]), file=out_file)

def remove_low_pairs(poisson_data, rmax):
    """ Removes one of each pair if one of the points fall within rmax of the
    other. This algorithm is to be run on the random data to produce a negative
    correlation function xi(r) at $$r < r_{max}$$."""
    n = len(poisson_data)
    n_RR = distances_to_each(n, poisson_data, poisson_data, 'DD')
    min_dist_points = np.zeros((n, n))
    for i in range(len(n_RR)):
        for j in range(len(n_RR)):
            if (i != j) and (n_RR[i,j] < rmax):
                    min_dist_points[i,j] = 1

    points_to_del = np.argwhere(min_dist_points == 1)[:,0]
    poisson_data = np.delete(poisson_data, points_to_del, axis=0)
    return poisson_data

#---------------------------MAIN PROGRAM START-----------------------------#
L = 1
N = 30 # number of galaxies to place within each sphere
N_pois = 100 # number of random points
RMAX = 0.07 # max radius of sphere 0.07

points_to_generate = N_pois//10 # number of points to place spheres on (N_pois//10 later)

data_random = np.transpose(get_coords(L, N_pois))
# data_random = remove_low_pairs(data_random, RMAX) # (seems to have no effect)

N_pois = len(data_random)
x = np.zeros(N_pois + points_to_generate * N)
y = np.zeros(N_pois + points_to_generate * N)
z = np.zeros(N_pois + points_to_generate * N)
x[:N_pois] = data_random[:,0]
y[:N_pois] = data_random[:,1]
z[:N_pois] = data_random[:,2]

number_of_rows = data_random.shape[0]

data = np.transpose([x, y, z])

sp_data = [] # split up data to run loop only on segments where generating new data
sp_data.append(data[:N_pois])
sp_data.append(np.array_split(data[N_pois:], points_to_generate))

# generate clusters around a random point
for i in range(points_to_generate):
    rand_indice = np.random.choice(number_of_rows)
    centre = data_random[rand_indice,:]
    for j in range(N): 
        r = np.random.random() * RMAX # set random radius
        sp_data[1][i][j,0], sp_data[1][i][j,1], \
            sp_data[1][i][j,2] = sample_spherical(1, r, centre)

# recombine the data into single array
data = sp_data[0]
for k in range(points_to_generate):
    ar_to_add = sp_data[1][k]
    data = np.concatenate([data, ar_to_add])

poisson_cluster_dist3D(L, data[:,0], data[:,1], data[:,2])

rand_data = np.transpose(get_coords(L, len(data))) # rand data for n_DR calc

n_DR = distances_to_each(len(data), data, rand_data, 'DR')
n_DD = distances_to_each(len(data), data, rand_data, 'DD')

xi_r_binned, xi_r_full = xi_r_calc_alt(n_DD, n_DR)

#--------------------------PLOTTING----------------------------------------#
ar = np.linspace(0, 1, len(x)**2//100 - 1)
plt.figure(dpi=150)
plt.plot(ar[1:], xi_r_binned[1:], color='deepskyblue')
avg = pd.Series(xi_r_binned[1:]).rolling(window=20).mean()
plt.plot(ar[1:], avg, color='k', label=r'$\langle \xi \rangle$')
plt.axvline(RMAX, color='r', linestyle='--', label=r'$r_{max}$')
plt.title(' Binned two point correlation function $\\xi(r)$ \nfor Poisson Clusters', \
          color='b')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Frequency of $\xi(r)$')
plt.legend(loc='best')
plt.tight_layout()

plt.figure(dpi=150)
plt.plot(np.linspace(0, 1, len(xi_r_full)), xi_r_full, 'g', label=r'$\xi(r)$')
plt.axvline(RMAX, color='r', linestyle='--', label=r'$r_{max}$')
plt.title(r'Two point correlation function for Poisson Clusters', color='b')
plt.ylabel(r'$\xi(r)$')
plt.xlabel(r'Distance $r$')
plt.legend(loc='best')
plt.tight_layout()

# file_write(data[:,0], data[:,1], data[:,2], 'poisson_MASTER1.txt')








