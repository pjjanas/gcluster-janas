# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:06:30 2022

@author: Pawel Janas (Python 3.10.5)

Description: Create master data file for Pairwise cluster model.
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
        Width of the cube for which the galaxies are placed in.
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

def pairwise_cluster_dist3D(L, x, y, z): # 3D plot
    """ Create a 3D plot of the Poisson distribution """
    plt.figure(dpi=150, figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, color='r', edgecolor='k', zorder=10)
    ax.set_title('Clustered Pairwise Distribution', fontsize=15)

def pairwise_cluster_dist2D(L, x, y): # 2D plot
    """ Create a 2D plot of the Poisson distribution """
    plt.figure(dpi=150, figsize=(6, 6))
    plt.plot(x, y, 'g+')
    plt.title('Clustered Pairwise Distribution (2D)', fontsize=15)

def xi_r_calc(n_RR1, n_RR2):
    """ Calculates the correlation function estimator  """
    data1 = n_RR1.flatten()
    data2 = n_RR2.flatten()
    
    r = np.linspace(0, 0.5, len(data1)//100)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
    
    xi_r_full = np.empty(len(data1)) # for all values
    for i in range(len(data1)):
        if (data2[i] == 0 or data1[i]==data2[i]):
            xi_r_full[i] = 0
        else:
            xi_r_full[i] = data1[i] / data2[i] - 1
            
    # handle div by 0...
    xi_r_binned = (binned_data1[0]+(binned_data1[0]==0)) / \
        (binned_data2[0]+(binned_data2[0]==0)) - 1

    return xi_r_binned, xi_r_full

def xi_r_calc_alt(n_RR1, n_RR2):
    """ Calculates the correlation function estimator  """
    data1 = n_RR1.flatten()
    data2 = n_RR2.flatten()
    
    r = np.linspace(0, 1, len(data1)//10)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
    
    data1.sort()
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

def xi_r_plot(N, xi_r_binned, xi_r_full):
    fig, ax = plt.subplots(2, 1, dpi=150, figsize=(7,7))
    ax[0].plot(np.linspace(0, 1, len(xi_r_full)), xi_r_full, 'g', label=r'Full $\xi(r)$ set')
    fig.suptitle('Two-point correlation function', \
                    color='b', fontsize=16)
    ax[0].set_ylabel(r'$\xi(r)$')
    fig.supxlabel(r'Distance $r$', fontsize=13)
    ar = np.linspace(0, 1, N**2//10 - 1)
    ax[1].plot(ar[1:], xi_r_binned[1:], color='deepskyblue', label=r'Binned $\xi(r)$')
    ax[1].set_ylabel(r'Frequency $\xi(r)$')
    avg1 = pd.Series(xi_r_binned[1:]).rolling(window=10).mean()
    ax[1].plot(ar[1:], avg1, color='k', label=r'$\langle \xi \rangle$')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    fig.tight_layout()

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

a, b, c = 0, 0, 0
centre = np.array([a, b, c])
N = 1 # number of points to place around each node
r = 0.05
xi, yi, zi = sample_spherical(100, r, centre)

L = 1
N_pois = 1000 # number of galaxies to form clusters around
x_pois, y_pois, z_pois = get_coords(L, N_pois)


x = np.zeros(N_pois + N_pois//10)
y = np.zeros(N_pois + N_pois//10)
z = np.zeros(N_pois + N_pois//10)
x[:N_pois] = x_pois
y[:N_pois] = y_pois
z[:N_pois] = z_pois

points_to_generate = N_pois//10 # points to generate pairs on

data_random = np.transpose([x_pois, y_pois, z_pois])

#-------------------------SPHERICAL NODE MODEL-----------------------------#

# x = np.empty(N_pois + N_pois * N) # init arrays for full clustered data set
# y = np.empty(N_pois + N_pois * N)
# z = np.empty(N_pois + N_pois * N)
# for i in range(N_pois):
#     centre = np.array([x_pois[i], y_pois[i], z_pois[i]])
#     x[N_pois+i*N:N_pois+(i+1)*N], y[N_pois+i*N:N_pois+(i+1)*N], \
#         z[N_pois+i*N:N_pois+(i+1)*N] = sample_spherical(N, r, centre)
#-------------------------------------------------------------------------#

#------------------PAIRWISE MODEL (10% of galaxies have a pair)-----------#

number_of_rows = data_random.shape[0] # find number of rows on axis 0
for i in range(points_to_generate):
    rand_indice = np.random.choice(number_of_rows)
    centre = data_random[rand_indice,:] # get random galaxy point
    
    x[N_pois+i*N:N_pois+(i+1)*N], y[N_pois+i*N:N_pois+(i+1)*N], \
        z[N_pois+i*N:N_pois+(i+1)*N] = sample_spherical(N, r, centre)
#--------------------------------------------------------------------------#

data = np.transpose([x, y, z])
rand_data = np.transpose(get_coords(L, len(data)))

pairwise_cluster_dist3D(L, x, y, z)
pairwise_cluster_dist2D(L, x, y)
# file_write(x, y, z, 'pairwise_MAST1000.txt') # write master data file

n_DD = distances_to_each(len(data), data, data, 'DD')
n_DR = distances_to_each(len(data), data, rand_data, 'DR')
n_RR = distances_to_each(len(data), rand_data, rand_data, 'DD')

xi_r_binned, xi_r_full = xi_r_calc_alt(n_DD, n_DR)

xi_r_plot(len(data), xi_r_binned, xi_r_full)

    
    