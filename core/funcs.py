# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:54:16 2022

@author: Pawel Janas (Python 3.10.5)

Description: Functions for use for janas-gcluster programs.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def newInput(mode, text):
    """
    Function that validates user input and prevents from letters being typed
    in.

    Parameters
    ----------
    mode : bool
        Defines the mode for deciding whether we want the user input to be a
        float or integer.
    text : str
        The message text that will be provided to the user before their input.

    Returns
    -------
    userInput : float, int, str (depending on mode)
        The user input value they entered within function bounds (ie int or
                                                                  float).

    """
    while (mode == 0):   
        try:
            userInput = float(input(text))
        except ValueError:
            print('\aInvalid!!')
            continue
        else:
            return userInput
    while (mode == 1):
        try:
            userInput = int(input(text))
        except ValueError:
            print('\aInvalid! Make sure to type in an integer!')
            continue
        else:
            return userInput
    while (mode == 2):
        try:
            userInput = str(input(text))
        except ValueError:
            print('This is not a string (letters)! Please try again.')
            continue
        else:
            return userInput

def file_read():
    """ Reads in data file. Data file must be 3 columned x, y, z data. """
    while True:
        try:
            file_name = str(input('\nPlease type in the name of the data file: '))
            data = np.loadtxt(file_name, skiprows=1)
        except OSError:
            print("\aThis file was not found in the directory. Please try again!\
(Make sure you're including .txt in the name)")
            continue
        else:
            return data

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
        
def poisson_dist2D(L, N, x, y): # 2D plot
    """ Create a 2D plot of the Poisson distribution """
    plt.figure(dpi=150, figsize=(6, 6))
    plt.plot(x, y, 'o', color='royalblue')
    plt.title('Random Distribution', fontsize=15)

def poisson_dist3D(L, N, x, y, z): # 3D plot
    """ Create a 3D plot of the Poisson distribution """
    plt.figure(dpi=150, figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'r+')
    ax.set_title('Poisson Distribution (3D)', fontsize=15)

def cluster_dist3D(L, f_data, t_data): # 3D plot
    """ Create a 3D plot of the galaxy distribution both original file and 
    thinned set."""
    fig = plt.figure(dpi=250, figsize=plt.figaspect(0.5))
    # first subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(f_data[:,0], f_data[:,1], f_data[:,2], color='r', edgecolor='k', \
               zorder=10)
    ax.set_title('Full Data')
    # second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(t_data[:,0], t_data[:,1], t_data[:,2], color='turquoise', \
               edgecolor='k', zorder=10)
    ax.set_title('Thinned Data')

def xi_r_plot(xi_r_binned, factor):
    fig, ax = plt.subplots(1, 1, dpi=150)
    # ax[0].plot(np.linspace(0, 1, len(xi_r_full)), xi_r_full, 'g', label=r'Full $\xi(r)$ set')
    # fig.suptitle('Two-point correlation function', \
    #                 color='b', fontsize=16)
    # ax[0].set_ylabel(r'$\xi(r)$')
    ax.set_xlabel(r'Distance $r$', fontsize=13)
    ar = np.linspace(0, 1, factor - 1)
    ax.plot(ar[1:], xi_r_binned[1:], color='deepskyblue', label=r'Binned $\xi(r)$')
    ax.set_ylabel(r'Frequency $\xi$')
    avg1 = pd.Series(xi_r_binned[1:]).rolling(window=10).mean()
    ax.plot(ar[1:], avg1, color='k', label=r'$\langle \xi \rangle$')
    # ax[0].legend(loc='best')
    ax.legend(loc='best')
    fig.tight_layout()
    
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

def Count_in_cell(L, data, counts, n_min, n_max, rmax, steps):
    """
    Create spheres at random locations on graph within boundaries set
    by the max radius (rmax) and counts number of galaxies within them with
    increasing galaxy number n determined by step.

    Parameters
    ----------
    L : int (could change to float in future)
        The length of the cube.
    data : array-like
        3 columned data array depicting x,y,z coordinates of data points.
    counts : int
        The number of spheres to count within for analyses.
    n_min : int
        Minimum or starting number of galaxies to draw 'counts' spheres on.
    n_max : int
        The max number of galaxies to analyse (=N total galaxies).
    rmax : float
        The maximum radius of the spheres that are generated.
    steps : int
        The amount of steps to take to cycle through the galactic catalogue.

    Returns
    -------
    $P_V(N)$ : array
        2D array of the counts-in-cell with the rows increasing with increasing
        N determined by 'steps' and the columns being the counts for each 
        sphere respectively.

    """
    a = np.random.random(counts) * (L - rmax) # set inner cube for sphere
    b = np.random.random(counts) * (L - rmax) # generation
    c = np.random.random(counts) * (L - rmax)
    n = np.linspace(n_min, n_max, steps+1, dtype=(int)) # to increase galaxy num
    
    sp_data = [] # split the data to add galaxies as we go, galaxies up to n_min
    sp_data.append(data[:n_min])                          # are not split.
    sp_data.append(np.array_split(data[n_min:], steps))
    
    P_VN = np.zeros((len(n), counts))
    tdata = np.zeros((len(data), 3)) # use this for analyses in loops
    p = np.zeros((len(data), counts))
    
    tdata[:n_min,:] = sp_data[0]
    
    for i in range(steps):
        for j in range(counts):
            chunk = len(sp_data[1][i]) # chunk size for cycling through catalogue
            tdata[n_min+i*chunk:n_min+(i+1)*chunk,:] = sp_data[1][i]
            p[:n_min+(i+1)*chunk,j] = make_sphere(tdata[:n_min+(i+1)*chunk,:],\
                                                  a[j], b[j], c[j])
    for k in range(len(n)):
        for l in range(counts):
            P_VN[k,l] = countinSphere(p[:n[k],l], n[k], rmax)
    return P_VN

def Count_in_cell_single(data, n_min, n_max, rmax, steps):
    """ Same procedure as for Count_in_cell but this time putting sphere on
    a single galaxy rather than distributing them randomly."""
    number_of_rows = data.shape[0] # find number of rows on axis 0
    rand_indice = np.random.choice(number_of_rows)
    rand_point = data[rand_indice,:] # get random galaxy point
    
    # now compute counts in cell
    n = np.linspace(n_min, n_max, steps+1, dtype=(int))
    
    sp_data = [] # split the data to add galaxies as we go, galaxies up to n_min
    sp_data.append(data[:n_min])                          # are not split.
    sp_data.append(np.array_split(data[n_min:], steps))
    
    tdata = np.zeros((len(data), 3)) # use this for analyses in loops
    P_VN1 = np.zeros(len(n))
    tdata = np.zeros((len(data), 3))
    p = np.zeros(len(data))
    
    for i in range(steps):
        chunk = len(sp_data[1][i])
        tdata[n_min+i*chunk:n_min+(i+1)*chunk,:] = sp_data[1][i]
        p[:n_min+(i+1)*chunk] = make_sphere(tdata[:n_min+(i+1)*chunk,:], \
                                            rand_point[0], rand_point[1], \
                                                rand_point[2])
    for k in range(len(n)):
        P_VN1[k] = countinSphere(p[:n[k]], n[k], rmax)
    return P_VN1
    
def galaxy_count_oneorigin(N, r, p1, p2):
    """ Test function for counting the number of galaxies within a sphere
    of increasing r """
    P_V1 = np.zeros(len(r))
    P_V2 = np.zeros(len(r))
    # finding counts in cell for increasing r with origin set at centre
    for i in range(len(r)):
        P_V1[i] = countinSphere(p1, N, r[i])
        P_V2[i] = countinSphere(p2, N, r[i])
    return P_V1, P_V2

def galaxy_count(N, r, arr1, arr2):
    """ Counts the galaxies within spheres placed at random, each with
    increasing radius r. (Another test function). """
    p3 = np.zeros((N,N))
    p4 = np.zeros((N,N))
    P_V1 = np.zeros((N,len(r)))
    P_V2 = np.zeros((N,len(r)))
    for i in range(len(r)):
        for j in range(N):
            p3[:,j] = make_sphere(arr1, arr1[j,0], \
                                  arr1[j,1], arr1[j,2])
            p4[:,j] = make_sphere(arr2, arr2[j,0], \
                                  arr2[j,1], arr2[j,2])
            P_V1[j,i] = countinSphere(p3[:,j], N, r[i])
            P_V2[j,i] = countinSphere(p4[:,j], N, r[i])
    return P_V1, P_V2

def distances_to_each(N, data1, data2, mode):
    """ Calculates distances between data for n_DD, n_RR etc. N and len(data1)
    must have the same length """
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

def moving_average(data, window_size):
    """ Computes the moving average and stores the values in an array. """
    data = data.flatten()
    moving_avg_ar = pd.Series(data).rolling(window_size).mean().to_numpy()
    final_arr = moving_avg_ar[window_size-1:] # remove all null entries
    return final_arr # return array of moving averages for further analyses

def xi_r_calc(n_RR1, n_RR2):
    r""" Calculates the correlation function estimator using 
    $${\xi}(r) = {n_{DD_1}}/{n_{DD_2}} - 1$$  """
    data1 = n_RR1.flatten()
    data2 = n_RR2.flatten()
    
    r = np.linspace(0, np.amax(data1), len(data1)//10)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
    
    data1.sort()
    data2.sort()
    
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

def xi_r_calc_alt(n_DD, n_DR, n_RR, factor):
    """
    Calculates the correlation function estimator.

    Parameters
    ----------
    n_DD : array
        2D masked array (excluding zero entries) of pairwise distances for 
        data-data catalogue.
    n_DR : array
        2D masked array (excluding zero entries) of pairwise distances for 
        data-random catalogue..
    n_RR : array
        2D masked array (excluding zero entries) of pairwise distances for 
        data-data catalogue..
    factor : int
        Number of bins for histogram generation.

    Returns
    -------
    xi_r_binned : array
        1D array of two-point correlation values. Can now be plotted.

    """
    data1 = n_DD.compressed() # compressed instead of flatten() to remove masked
    data2 = n_DR.compressed()                                      # entries
    data3 = n_RR.compressed()
    
    r = np.linspace(0, 1, factor)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
    binned_data3 = np.histogram(data3, r)
            
    # handle div by 0...
    xi_r_binned = ((binned_data1[0]) * \
                   (binned_data3[0])) / \
        ((binned_data2[0]+(binned_data2[0]==0))**2) - 1

    return xi_r_binned

