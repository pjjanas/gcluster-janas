# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:06:38 2022

@author: Pawel Janas (Python 3.8)

Student Number: 19318421

Description: File reading for Galaxy Data
"""

import numpy as np

def file_read(file_name):
    in_file = open(file_name, 'r') # open file for reading
    
    data = np.loadtxt(file_name, skiprows=1)
    return data

data = file_read('pairwise_MASTER.txt')
        

