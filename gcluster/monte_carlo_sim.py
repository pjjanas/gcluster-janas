# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:36:15 2022

@author: Pawel Janas (Python 3.10.5)

Description: Monte Carlo simulations for Clustered Patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# change default font to make graphs presentable
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['agg.path.chunksize'] = 10000 # for large computations for plotting

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

def get_thinned(full_data, N, replace=False):
    """ Function for getting thinned out data of size N. Use replcae=True if 
    you don't mind getting duplicates (False by default). """
    thinned_data = np.zeros((N, full_data.shape[1]))
    
    if replace == True:   
        num_of_rows = full_data.shape[0]
        for i in range(N): # select random points from total data set
            rand_indice = np.random.choice(num_of_rows)
            thinned_data[i,:] = full_data[rand_indice,:]
            
    else:
        rand_indices = np.random.choice(full_data.shape[0], size=N, replace=False)
        for i in range(N):
            thinned_data[i,:] = full_data[rand_indices[i],:]
            
    return thinned_data

def make_sphere(data, a, b, c):
    """
    Creates a virtual sphere around some point in space centered on
    (a, b, c) and calculates r^2 for each data point. 
    To be passed to countinSphere() function for counting points within.

    Parameters
    ----------
    data : array-like
        2D coordinate array with columns being x, y, z coordinates respectively.
    a : float
        x-coordinate of origin of sphere.
    b : float
        y-coordinate of origin of sphere.
    c : float
        z-coordinate of origin of sphere.

    Returns
    -------
    p : array-like
        Returns the array of $r^2$ terms.

    """
    # sphere centred on (a,b,c)
    # compute r**2 for each data point
    p = np.zeros(len(data[:,0])) # number of r**2 points to fill
    for i in range(len(data[:,0])):
        p[i] = (data[i,0]-a)**2 + (data[i,1]-b)**2 + (data[i,2]-c)**2
        
    p.sort() # sort r**2 array
    return p

def countinSphere(p, N, rad):
    """
    Counts the number of points within some radius rad of some point in space.

    Parameters
    ----------
    p : array
        1D input array of $r^2$ terms found in make_sphere().
    N : int
        The number of points to search through and check whether they are 
        inside the sphere.
    rad : float
        The radius of the sphere in which the points will be counted.

    Returns
    -------
    The number of points lying inside some sphere.

    """
    # find number of points within radius rad
    # point will lie in sphere if x^2 + y^2 + z^2 <= r^2
    # using binary search
    start = 0
    end = N-1
    while((end - start) > 1):
        mid = (start + end) // 2
        tp = np.sqrt(p[mid]) # test point
        
        if (tp > rad):
            end = mid - 1
        else:
            start = mid
    
    tp1 = np.sqrt(p[start]) # setup test-points to start checking if they're <=r
    tp2 = np.sqrt(p[end])
    if (tp1 > rad):
        return 0
    elif (tp2 <= rad):
        return end + 1
    else:
        return start + 1
    
def galaxy_count(N_sph, N, r, data):
    """
    Counts the galaxies within spheres placed at random points within the
    catalogue, for increasing radius r. Then calculates the average over all 
    spheres and stores the values in an array of size (len(r)).

    Parameters
    ----------
    N_sph : int
        The number of spheres to generate placed at random points on the 
        catalogue.
    N : int
        The total number of data points to be passed to countinSphere for
        checking.
    r : array
        1 dimensional input array of radius values to make spheres with.
    data : array
        3 columned two-dimensional array containing x, y and z coordinates of 
        the data.

    Returns
    -------
    P_V_avg : array
        Returns array of counts of points within each radius with increasing r.

    """
    p = np.zeros((data.shape[0], N_sph))
    P_V = np.zeros((N_sph, len(r)))
    
    num_rows = data.shape[0]
    # calculate number of points within each sphere
    for i in range(len(r)):
        for j in range(N_sph):
            rand_indice = np.random.choice(num_rows)
            p[:,j] = make_sphere(data, data[rand_indice,0], \
                                  data[rand_indice,1], data[rand_indice,2])
                
            P_V[j,i] = countinSphere(p[:,j], N, r[i])
    
    # now average all counts for increasing r
    P_V_avg = np.zeros(len(r))
    for i in range(len(r)):
        P_V_avg[i] = np.mean(P_V[:,i])
        
    return P_V_avg

def distances_to_each(N, data1, data2, mode):
    """N and arr must have the same length """
    if (mode == 'DD'): # data-data distance computation
        n_RR = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                n_RR[i,j] = math.dist(data1[i,:], data1[j,:])
        # mask the array to exclude 0 entries
        n_RR = np.ma.masked_equal(n_RR, 0)
        return n_RR
    
    elif (mode == 'DR'): # data random distance computation (or data1-data2)
        n_DR = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                n_DR[i,j] = math.dist(data1[i,:], data2[j,:])
        n_DR = np.ma.masked_equal(n_DR, 0) # mask again
        return n_DR

def xi_r_calc_alt(n_DD, n_DR, n_RR, Nbins):
    """ Calculates the correlation function estimator  """
    data1 = n_DD.compressed() # compressed instead of flatten() to remove masked
    data2 = n_DR.compressed()                                      # entries
    data3 = n_RR.compressed()
    
    r = np.linspace(0.01, 1, Nbins)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
    binned_data3 = np.histogram(data3, r)
            
    # handle div by 0...
    xi_r_binned = ((binned_data1[0]) * \
                   (binned_data3[0])) / \
        ((binned_data2[0]+(binned_data2[0]==0))**2) - 1

    return xi_r_binned

def xi_r_calc_basic(n_DD, n_RR, Nbins):
    """ Calculates the correlation function estimator  """
    data1 = n_DD.compressed() # compressed instead of flatten() to remove masked
    data2 = n_RR.compressed()                                      # entries

    r = np.linspace(0.01, 1, Nbins)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
            
    # handle div by 0...
    xi_r_binned = binned_data1[0] / (binned_data2[0]+(binned_data2[0]==0)) - 1

    # return binned data for debugging purposes
    return xi_r_binned, binned_data1, binned_data2

def cluster_dist3D(L, x, y, z): # 3D plot
    """ Create a 3D plot of the galaxy distribution """
    plt.figure(dpi=150, figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, color='r', edgecolor='k', zorder=10)
    ax.set_title('Clustered Distribution', fontsize=15)
    
def xi_r_plot(xi_r_binned, Nbins):
    fig, ax = plt.subplots(1, 1, dpi=150)
    # ax[0].plot(np.linspace(0, 1, len(xi_r_full)), xi_r_full, 'g', label=r'Full $\xi(r)$ set')
    # fig.suptitle('Two-point correlation function', \
    #                 color='b', fontsize=16)
    # ax[0].set_ylabel(r'$\xi(r)$')
    ax.set_xlabel(r'Distance $r$', fontsize=10)
    ar = np.linspace(0.01, 1, Nbins - 1)
    ax.plot(ar[1:], xi_r_binned[1:], color='deepskyblue', label=r'Binned $\xi(r)$')
    ax.set_ylabel(r'$\xi$', fontsize=10)
    avg1 = pd.Series(xi_r_binned[1:]).rolling(window=5).mean()
    ax.plot(ar[1:], avg1, color='k', label=r'$\langle \xi \rangle$')
    # ax[0].legend(loc='best')
    ax.legend(loc='best')
    fig.tight_layout()
    
#----------------------------MAIN PROGRAM STARTS HERE------------------------#
MC_RUNS = 1000 # number of times to simulate
L = 1 # width of box

file_name = 'powerlaw_HIGHN1.txt'
data = np.loadtxt(file_name, skiprows=1)

N = 100 # number of points to gather from data
Nbins = 500 # for plotting and binning n_DD etc.

n_DD_full = np.zeros((N*MC_RUNS, N)) # init arrays for storing stacked n_DD etc
n_DR_full = np.zeros((N*MC_RUNS, N))
n_RR_full = np.zeros((N*MC_RUNS, N))

radii = np.linspace(0.005, 0.1, 50) # radius for galaxy_count()
N_spheres = 5
P_V = np.zeros((len(radii), MC_RUNS))

for i in range(MC_RUNS):
    if (i % 10 == 0):
        progress = i / MC_RUNS * 100
        print(str('Progress = ') + str(progress) + str('%')) # print progress
    
    new_data = get_thinned(data, N) # get thinned data
    # new_data = np.transpose(get_coords(L, N)) # for random catalogue
    random_data = np.transpose(get_coords(L, N))
    
    n_DD_full[i*N:(i+1)*N,:] = distances_to_each(N, new_data, random_data, 'DD')
    n_DR_full[i*N:(i+1)*N,:] = distances_to_each(N, new_data, random_data, 'DR')
    n_RR_full[i*N:(i+1)*N,:] = distances_to_each(N, random_data, random_data, 'DD')
    
    P_V[:,i] = galaxy_count(N_spheres, N, radii, new_data) # gain count per r
    
    if (np.log10(i+1) % 1 == 0): # plotting at each power of 10
    # this is a bit of waste of memory but it wouldn't work otherwise; problems
    # with making masked arrays with partial arrays
        n_DD_masked = np.zeros(((i+1)*N, N)) # init arrays
        n_DR_masked = np.zeros(((i+1)*N, N))
        n_RR_masked = np.zeros(((i+1)*N, N))
        
        n_DD_masked = n_DD_full[:(i+1)*N,:]
        n_DR_masked = n_DR_full[:(i+1)*N,:]
        n_RR_masked = n_RR_full[:(i+1)*N,:]
        
        n_DD_masked = np.ma.masked_equal(n_DD_masked, 0)
        n_DR_masked = np.ma.masked_equal(n_DR_masked, 0)
        n_RR_masked = np.ma.masked_equal(n_RR_masked, 0)
        
        xi_r_inter = xi_r_calc_alt(n_DD_masked, n_DR_masked, \
                                    n_RR_masked, Nbins)
        # xi_r_plot(xi_r_inter, factor)
        # plt.title(f' Zoomed Plot of $\\xi(r)$ for N: {N}, MC Run: {i+1}', fontsize = 16)
        # plt.ylim(-5,5)
        xi_r_plot(xi_r_inter, Nbins)
        plt.title(f'Plot of $\\xi(r)$ for N: {N}, MC Run: {i+1}', fontsize = 16)
        # plt.savefig('Plots/MonteCarlo Runs/pairwise_'+str(i+1)+'_bins'+str(factor), \
        #             bbox_inches='tight')
    

# mask the n_DD arrays to exclude 0 entries
n_DD_full = np.ma.masked_equal(n_DD_full, 0)
n_DR_full = np.ma.masked_equal(n_DR_full, 0)
n_RR_full = np.ma.masked_equal(n_RR_full, 0)

# =========================CALC/PLOTTING CELL COUNTS===========================
P_V_full = np.zeros(len(radii))
for i in range(len(radii)):
    P_V_full[i] = np.mean(P_V[i,:])
    
plt.figure(dpi=150)
plt.plot(radii, P_V_full, label=r'$P_V(r)$', color='darkorange')
plt.plot(radii, 2.5e5*radii**3 + 1, label=r'$r^3$ fit')
plt.title('Average Galaxy Count within Distance $r$', fontsize=16)
plt.xlabel(r'Distance $r$', fontsize=10)
plt.ylabel(r'Average Galaxy Count', fontsize=10)
plt.legend(loc='best')
plt.tight_layout()
# =============================================================================

xi_r = xi_r_calc_alt(n_DD_full, n_DR_full, n_RR_full, Nbins)
xi_r_plot(xi_r, Nbins)
plt.title(f'Plot of $\\xi(r)$ for N: {N}, MC Run: {i+1}', fontsize = 16)
    
# r = np.linspace(0.01, 1, factor)
# plt.plot(r, r**-1.8)
# plt.ylim(-3, 200)
    
