# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:42:53 2022

@author: Pawel Janas (Python 3.8)

Student Number: 19318421

Description: Computation time for calculating xi(r) for increasing N.
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt

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

def xi_r_calc(n_RR1, n_RR2):
    """ Calculates the correlation function estimator  """
    data1 = n_RR1.flatten()
    data2 = n_RR2.flatten()
    xi_r = np.zeros(len(data1))
    for i in range(len(data1)):
        if (data2[i] == 0 or data1[i]==data2[i]):
            xi_r[i] = 0
        else:
            xi_r[i] = data1[i] / data2[i] - 1
    return xi_r

iterations = 5
nmin, nmax = 10, 5000
N_big = np.linspace(nmin, nmax, iterations+1, dtype=(int))

x1, y1, z1 = get_coords(1, nmax)
x2, y2, z2 = get_coords(1, nmax)
data1 = np.transpose([x1,y1,z1])
data2 = np.transpose([x2,y2,z2])

split_data1 = [] # split the data to add galaxies as we go, galaxies up to n_min
split_data1.append(data1[:nmin])                         # are not split.
split_data1.append(np.array_split(data1[nmin:], iterations+1))

split_data2 = [] # split the data to add galaxies as we go, galaxies up to n_min
split_data2.append(data2[:nmin])                         # are not split.
split_data2.append(np.array_split(data2[nmin:], iterations+1))

test_data1 = np.zeros((len(data1), 3))
test_data2 = np.zeros((len(data2), 3))
test_data1[:nmin] = split_data1[0]
test_data2[:nmin] = split_data2[0]

comp_times = np.zeros(len(N_big))

for i in range(iterations+1):
    start_time = time.perf_counter()
    chunk = len(split_data1[1][i])
    
    print('N: ', N_big[i])
    print
    
    test_data1[nmin+i*chunk:nmin+(i+1)*chunk,:] = split_data1[1][i]
    test_data2[nmin+i*chunk:nmin+(i+1)*chunk,:] = split_data2[1][i]
    n_RR1_multiN = np.zeros((len(test_data1[:nmin+(i+1)*chunk,:]), \
                             len(test_data1[:nmin+(i+1)*chunk,:])))
    n_RR2_multiN = np.zeros((len(test_data1[:nmin+(i+1)*chunk,:]), \
                             len(test_data1[:nmin+(i+1)*chunk,:])))
    
    n_RR1_multiN = distances_to_each(len(n_RR1_multiN), test_data1[:nmin+(i+1)*chunk,:],\
                                     data2, mode='DD')
    n_RR2_multiN = distances_to_each(len(n_RR2_multiN), test_data2[:nmin+(i+1)*chunk,:],\
                                     data2, mode='DD')
    xi_r_multiN = xi_r_calc(n_RR1_multiN, n_RR2_multiN)
    print('Iteration: ', i)
    print('\n')
    comp_times[i] = (time.perf_counter() - start_time)

plt.figure(dpi=150)
plt.title('Computation Times with Increasing N', color='purple')
plt.plot(N_big, comp_times, color='firebrick', label=r'$t(N)$')
plt.ylabel('Time (seconds)')
plt.xlabel('N')
plt.legend(loc='best')
plt.tight_layout()


    