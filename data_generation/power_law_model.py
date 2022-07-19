# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:00:54 2022

@author: Pawel Janas (Python 3.10.5)

Description: Power law cluster model (r^-1.8) using Poisson cluster model as 
pre-requisit.
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import math

class powerlaw_pdf(st.rv_continuous): # subclassing to create custom pdf
    """ Generate custom probability density function to sample distances from."""
    def _pdf(self, x):
        return pow(x, -1.8)

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

def cluster_dist3D(L, x, y, z): # 3D plot
    """ Create a 3D plot of the galaxy distribution """
    plt.figure(dpi=150, figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, color='r', edgecolor='k', zorder=10)
    ax.set_title('Clustered Distribution', fontsize=15)

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

N_samples = 1000
power_lawPDF = powerlaw_pdf(a=1, b=100, name='power_law') # 
rad = power_lawPDF.rvs(size=N_samples) / 100 # div by 100 for scale factor

plt.figure(dpi=150)
plt.hist(rad, bins=100, color='slateblue') # histogram for radius distribution 
plt.title(f'Distance Distribution with {N_samples} samples', color='blue', \
          fontsize=15)
plt.xlabel(r'$r$', fontsize=12)
plt.ylabel(r'Frequency', fontsize=12)
plt.tight_layout()

#---------------------------MAIN PROGRAM START-----------------------------#
L = 1
N = 100 # number of galaxies to place within each sphere
N_pois = 1000 # number of random points
RMAX = 0.07 # max radius of sphere 0.07

points_to_generate = N_pois//10 # number of points to place spheres on (N_pois//10 def)

data_random = np.transpose(get_coords(L, N_pois))
# data_random = remove_low_pairs(data_random, RMAX) # (seems to have no effect)

N_pois = len(data_random) # update max count if remove_low_pairs() called
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
        r = np.random.choice(rad) # set random radius from radius distribution
        sp_data[1][i][j,0], sp_data[1][i][j,1], \
            sp_data[1][i][j,2] = sample_spherical(1, r, centre)

# recombine the data into single array
data = sp_data[0]
for k in range(points_to_generate):
    ar_to_add = sp_data[1][k]
    data = np.concatenate([data, ar_to_add])
    
rand_data = np.transpose(get_coords(L, len(data)))

cluster_dist3D(L, data[:,0], data[:,1], data[:,2])

# n_DD = distances_to_each(len(data), data, data, 'DD')
# n_DDf = n_DD.flatten()
# n_RR = distances_to_each(len(data), rand_data, rand_data, 'DD')
# n_RRf = n_RR.flatten()
# n_DR = distances_to_each(len(data), data, rand_data, 'DR')
# n_DRf = n_DR.flatten()

# xi_r = xi_r_calc_alt(n_DD, n_DR)

# r = np.linspace(0, 1, len(data)**2//10 - 1)
# plt.figure(dpi=150)
# plt.plot(r[1:], xi_r[1:])

# file_write(data[:,0], data[:,1], data[:,2], 'powerlaw_HIGHN1.txt')

#-----------------------------PLOTTING AND TESTING---------------------------#
# #-------------------Test code for single cluster placed at origin----------#
centre = np.zeros(3)
o = np.zeros(3) # origin
n = 100
num_of_rad = rad.shape[0]
points = np.zeros((n, 3))
for i in range(n):
    rand_indice = np.random.choice(num_of_rad)
    radius = rad[rand_indice]
    points[i,0], points[i,1], points[i,2] = sample_spherical(1, radius, centre)
#----------------------------------------------------------------------------#

# plt.figure(dpi=150)
# plt.hist(n_RRf, bins=int(np.sqrt(len(n_RRf))), label=r'$n_{RR}$')
# plt.hist(n_DDf, bins=int(np.sqrt(len(n_DDf))), label=r'$n_{DD}$')
# plt.title(r'Histograms of n_DD and n_RR')
# plt.legend(loc='best')

# p = make_sphere(points, 0,0,0)
# r = np.linspace(0.01, 0.1, 100)
# rho_r = np.zeros(len(r))
# for i in range(len(r)):
#     count = countinSphere(p, n, r[i])
#     vol = 4/3 * np.pi * r[i]**3
#     rho_r[i] = count / vol

# factor = n*10
# fit = pow(r, -1.8)
# plt.figure(dpi=150)
# plt.plot(r, rho_r, label=r'$\rho(r)$ (counts per volume)')
# plt.plot(r, fit*factor, label=r'$r^{-1.8}$ (scaled)')
# plt.title(r'$\rho(r)$ Comparison (single cluster)')
# plt.xlabel(r'$r$')
# plt.ylabel(r'$\rho$')
# plt.legend()
#---------------------------------------------------------------------------#



