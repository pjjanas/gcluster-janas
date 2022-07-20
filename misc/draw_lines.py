# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:11:56 2022

@author: Pawel Janas (Python 3.8)

Student Number: 19318421

Description: 
"""
import numpy as np
import matplotlib.pyplot as plt

# change default font to make graphs presentable
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['agg.path.chunksize'] = 10000 # for large computations for plotting

def file_read(file_name=None):
    if file_name==None:
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
    else:
        while True:
            try:
                data = np.loadtxt(file_name, skiprows=1)
            except OSError:
                print("\aThis file was not found in the directory. Please try again!\
    (Make sure you're including .txt in the name)")
                continue
            else:
                return data

def dist3D(L, x, y, z): # 3D plot
    """ Create a 3D plot of the data """
    plt.figure(dpi=150, figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, color='coral', edgecolor='k', zorder=10)
    ax.set_title('Distance Distribution', fontsize=15)

def connectpoints(x, y, z, p1, p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1, z2 = z[p1], z[p2]
    ax.plot3D()
    
L = 1
N = 300
data = file_read('powerlaw_MAST.txt')
x, y, z = data[:,0], data[:,1], data[:,2]

plt.figure(dpi=300, figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.scatter(x[:N], y[:N], z[:N], color='coral', edgecolor='k', zorder=10)
ax.set_title('Distance Distribution', color='royalblue', fontsize=15)

count = 0
for i in range(0, N):
    ax.plot3D([x[5], x[i]], [y[5], y[i]], [z[5], z[i]], markerfacecolor='coral', \
              linestyle='-', color='k', lw=0.3)
   



