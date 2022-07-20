# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:13:59 2022

@author: Pawel Janas (Python 3.8)

Student Number: 19318421

Description: testing string input for data
"""

import numpy as np

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
        
        
def file_read():
    while True:
        try:
            file_name = str(input('Please type in the name of the data file: '))
            data = np.loadtxt(file_name, skiprows=1)
        except OSError:
            print('\aThis file was not found in the directory. Please try again!')
            continue
        else:
            return data

data = file_read()



