# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:19:46 2022

@author: Pawel Janas (Python 3.10.5)

Description: SPUR Project init file (Monte Carlo Models on Galaxy Clustering).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# verifying user input
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
    userInput : float, int (depending on mode)
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

def moving_average(data, window_size):
    """ Computes the moving average and stores the values in an array. """
    data = data.flatten()
    moving_avg_ar = pd.Series(data).rolling(window_size).mean().to_numpy()
    final_arr = moving_avg_ar[window_size-1:] # remove all null entries
    return final_arr # return array of moving averages for further analyses

def xi_r_calc(n_RR1, n_RR2, factor):
    r""" Calculates the correlation function estimator using 
    $${\xi}(r) = {n_{DD_1}}/{n_{DD_2}} - 1$$  """
    data1 = n_RR1.flatten()
    data2 = n_RR2.flatten()
    
    r = np.linspace(0, np.amax(data1), factor)
    binned_data1 = np.histogram(data1, r) # bin the data in r
    binned_data2 = np.histogram(data2, r)
            
    # handle div by 0...
    xi_r_binned = (binned_data1[0]+(binned_data1[0]==0)) / \
        (binned_data2[0]+(binned_data2[0]==0)) - 1

    return xi_r_binned

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

#------------------------------MAIN PROGRAM START---------------------------#
L = 1 # length of box for plot
N = newInput(1, 'Please type in the max number of galaxies to be \
generated:\n') # number of galaxies
print(r'Generating data, plots and $P_V(r)$...')

factor = 200

x1, y1, z1 = get_coords(L, N) # first set of coordinates
x2, y2, z2 = get_coords(L, N) # second " " "
data1 = np.transpose([x1, y1, z1]) # package coords into data arrays
data2 = np.transpose([x2, y2, z2])
#n = N / L**3 # galaxy number density (will change L to small l for local densities)

poisson_dist2D(L, N, x1, y1)
poisson_dist3D(L, N, x1, y1, z1)

p1 = make_sphere(data1, 0.5, 0.5, 0.5)
p2 = make_sphere(data2, 0.5, 0.5, 0.5)

rmin, rmax = 0.1, 1
r = np.linspace(rmin, rmax, 50) # bins

n_RR1 = distances_to_each(N, data1, data2, mode='DD') # find distances between pairs of points
n_RR2 = distances_to_each(N, data1, data2, mode='DR')

# count galaxies with increasing r
P_Vr1, P_Vr2 = galaxy_count_oneorigin(N, r, p1, p2)

# now doing it for individual pairs i.e setting the origin of each sphere as a
# data point and repeating above calculation, this time we have a 2D array:    
P_VV1, P_VV2 = galaxy_count(N, r, data1, data2)

# Now creating spheres at random locations and increasing galaxy number (n)

#-------------------------------INPUTS-------------------------------------#
nmin = newInput(1, 'Type in the starting number of galaxies:\n')
nmax = N
# validate nmin and nmax:
while (nmin > nmax):
    print('\aThese are invalid galaxy numbers! Make sure starting number of \
galaxies is less than max number of galaxies.')
    nmin = newInput(1, 'Type in the starting number of galaxies: ')
    N = newInput(1, 'Please type in the number of max number of galaxies to be \
generated:\n')

steps = newInput(1, 'Type in the number of steps to take to cycle through \
galaxy number (nmin to nmax):\n')
counts = newInput(1, 'Type in the number of spheres to randomly generate \
for counts in cell:\n') # number of spheres to generate
RMAX = newInput(0, 'Type in the radius of spheres to be generated (0 < r <= 1):\n')
while(RMAX <= 0 or RMAX > 1): # validate
    print('\aThis is not a valid radius! Please type in the radius again.')
    RMAX = newInput(0, 'Type in the radius of spheres to be generated \
(0 < r <= 1):\n')
#----------------------------end INPUTS-----------------------------------#

P_VN = Count_in_cell(L, data1, counts, nmin, nmax, RMAX, steps) # P_V(N)
P_VN1 = Count_in_cell_single(data1, nmin, nmax, RMAX, steps) # P_V(N) for one point
t_PVN = P_VN # just for variable explorer
t_PVN1 = P_VN1

xi_r = xi_r_calc(n_RR1, n_RR2, factor)

#------------MEAN AND VARIANCE CHANGES OF xi(R) with increasing N-----------#

iterations = 5
N_big = np.linspace(nmin, nmax, iterations+1, dtype=(int))
mean_xi_r = np.zeros(iterations)
var_xi_r = np.zeros(iterations)
        
avg_chunk = 100 # to bin averaged values of xi_r

split_data1 = [] # split the data to add galaxies as we go, galaxies up to n_min
split_data1.append(data1[:nmin])                         # are not split.
split_data1.append(np.array_split(data1[nmin:], iterations))

split_data2 = [] # split the data to add galaxies as we go, galaxies up to n_min
split_data2.append(data2[:nmin])                         # are not split.
split_data2.append(np.array_split(data2[nmin:], iterations))

test_data1 = np.zeros((len(data1), 3))
test_data2 = np.zeros((len(data2), 3))

test_data1[:nmin] = split_data1[0]
test_data2[:nmin] = split_data2[0]

for i in range(iterations):
    chunk = len(split_data1[1][i])
    
    test_data1[nmin+i*chunk:nmin+(i+1)*chunk,:] = split_data1[1][i]
    test_data2[nmin+i*chunk:nmin+(i+1)*chunk,:] = split_data2[1][i]
    n_RR1_multiN = np.zeros((len(test_data1[:nmin+(i+1)*chunk,:]), \
                             len(test_data1[:nmin+(i+1)*chunk,:])))
    n_RR2_multiN = np.zeros((len(test_data1[:nmin+(i+1)*chunk,:]), \
                             len(test_data1[:nmin+(i+1)*chunk,:])))
    
    # print(N_big[i]) # debugging
    
    n_RR1_multiN = distances_to_each(len(n_RR1_multiN), \
                            test_data1[:nmin+(i+1)*chunk,:], data2, mode='DD')
    n_RR2_multiN = distances_to_each(len(n_RR2_multiN), \
                            test_data2[:nmin+(i+1)*chunk,:], data2, mode='DD')
        
    xi_r_multiN = xi_r_calc(n_RR1_multiN, n_RR2_multiN, factor)
    xi_r_avged = np.zeros(len(n_RR1_multiN)**2 // avg_chunk)
    
    for j in range(len(n_RR1_multiN)**2 // avg_chunk):
        xi_r_avged[j] = np.mean(xi_r_multiN[j*avg_chunk:(j+1)*avg_chunk])
        
    mean_xi_r[i] = np.mean(xi_r_avged)
    var_xi_r[i] = np.var(xi_r_avged)

#------------end MEAN AND VARIANCE CHANGES OF xi(R) with increasing N--------#

# PLOTTING
# plt.figure(dpi=150)
# plt.plot(np.linspace(nmin, nmax, steps+1), P_VN) # plot spheres findings
# plt.title(rf'$P_V(N)$ with $r_{{max}}$= {RMAX}', color='b')
# plt.xlabel('N')
# plt.ylabel(r'$P_{V}$', fontsize=12)
# plt.tight_layout()

# fig, ax = plt.subplots(2, 1, dpi=150, figsize=(7,7))
# ax[0].plot(xi_r_full, 'g', label=r'Full $\xi(r)$ set')
# avg0 = pd.Series(xi_r_full).rolling(window=20).mean()
# ax[0].plot(avg0, color='k', label=r'$\langle \xi \rangle$')
# fig.suptitle('Two-point correlation function for random distributions', \
#                 color='b', fontsize=16)
# ax[0].set_xlabel(r'Index')
# fig.supylabel(r'$\xi(r)$', fontsize=13)
# ar = np.linspace(0, np.amax(n_RR1), nmax**2//10 - 1)
# ax[1].plot(ar[1:], xi_r_binned[1:], color='darkorange', label=r'Binned $\xi(r)$')
# ax[1].set_xlabel(r'Distance $r$')
# avg1 = pd.Series(xi_r_binned[1:]).rolling(window=10).mean()
# ax[1].plot(ar[1:], avg1, color='k', label=r'$\langle \xi \rangle$')
# ax[0].legend(loc='best')
# ax[1].legend(loc='best')
# fig.tight_layout()

xi_r_plot(xi_r, factor)
plt.title(r'Plot of correlation function ($\xi(r)$)', fontsize=16)
#------------------------------END MAIN-------------------------------------#