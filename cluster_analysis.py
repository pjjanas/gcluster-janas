# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:57:23 2022

@author: Pawel Janas (Python 3.10.5)

Description: Program to run cluster analysis tools on a specific data set.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# append path to get specific module
sys.path.append("C:\\Users\\pjjan\\OneDrive\\Documents\\Python\\SPUR")

from funcs import file_read, newInput, get_coords, get_thinned, cluster_dist3D, \
    distances_to_each, xi_r_calc_alt, xi_r_plot

#--------------------------MAIN PROGRAM STARTS HERE-------------------------#
# inform the user what's happening
print('\nThis program is used to check and run some analysis routines on a \
desired data set which contains x, y, and z coordinates. It is useful for \
visualising the desired data set with and without thinning, and running desired \
functions such as calculating pairwise distances and the two point correlation \
function.')
 
# create algo for thinning data
data = file_read()
#data = np.transpose(get_coords(1, 10000)) # use this if you want random data
num_of_rows = data.shape[0]
L = 1

N = newInput(1, f'\nPlease type in the number of data points N to analyse \
(must be <= {num_of_rows}):\n') # how many data points you want
while (N > num_of_rows):
    print(f'\aThis number is greater than the size of the data set! Make sure \
N <= {num_of_rows}.')
    N = newInput(1, f'Please type in the number of data points N to analyse \
(must be <= {num_of_rows}):\n')

factor = 500 # number of bins for histogram of xi_r

new_data = get_thinned(data, N, replace=False) # get thinned data without replacement

q = newInput(2, '\nWould you like 3D plots of the data? (y/n): ')
while True:
    if q == 'y':
        cluster_dist3D(L, data, new_data)
        break # break while loop
    elif q == 'n':
        break
    else:    
        q = newInput(2, 'Would you like 3D plots of the data? (y/n): ')

q = newInput(2, '\nRun distances calculations for n_DD and n_DR? (y/n): ')
while True:
    if q == 'y':
        print('Running...This might take a while for large counts.')
        random_data = np.transpose(get_coords(L, N)) # get Poisson data for n_DR calc

        n_DR = distances_to_each(N, new_data, random_data, 'DR')
        n_DD = distances_to_each(N, new_data, random_data, 'DD')
        n_RR = distances_to_each(N, random_data, random_data, 'DD')
        print('Finished n_DD etc calculation...') # debugging
        
        q = newInput(2, '\nCalculate Correlation function $\\xi(r)$? (y/n): ')
        while True:
            if q == 'y':
                xi_r = xi_r_calc_alt(n_DD, n_DR, n_RR, factor)
                break
            elif q == 'n':
                break
            else:
                q = newInput(2, r'Calculate Correlation function $\xi(r)$? (y/n): ')
                
        q = newInput(2, '\nPlot $\\xi(r)$? (y/n): ') # ask to plot xi(r)
        while True:
            if q == 'y':
                ar = np.linspace(0, 1, factor - 1) # debugging
                xi_r_plot(xi_r, factor) # plot
                plt.title(r'Plot of correlation function ($\xi(r)$)', fontsize=16)
                break
            elif q == 'n':
                break
            else:
                q = newInput(2, r'Plot $\xi(r)$? (y/n): ')
        break

    elif q =='n':
        break
    else:
        q = newInput(2, 'Run distances calculations for n_DD and n_DR? (y/n): ')

