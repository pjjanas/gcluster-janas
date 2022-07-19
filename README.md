# gcluster-janas
Repository for a university based project on galaxy clustering for which the code was developed by Pawel Janas with the help of Professor Coles at Maynooth University, Ireland 2022.

# Introduction
Welcome to `gcluster-janas`! The programs found in this repositary simulate galaxy distributions in a simple cubic geometry and run specific analysis algorithms on them such as the two-point correlation function. I created this repository as a placeholder for my project based code/files for a project I completed with Maynooth University as part of their SPUR program. In this README document, I will outline the basics of using my code, specific function/variable names and ways to improve my code for future use.

## Disclaimer
This code is not a finished product and should not be treated as such. Bugs and glitches may occur and minor mistakes may exist in the algorithms. Misuse of the program (e.g. allowing it to run for 50,000 Monte Carlo runs) may crash your machine.

## Installation
This program runs on Python 3.8 and above (not tested with earlier versions). It requires the following packages:
```
numpy 
matplotlib 
math 
pandas
scipy
```
The programs were developed using the [Spyder](https://github.com/spyder-ide/spyder) IDE.
To 'install' this repository, simply download the repository or just the program files and place them in your current working directory (mine was a specific folder named `Programs` from which all Python programs ran). If you require an IDE or a custom working environment. I highly recomment installing through [Anaconda](https://www.anaconda.com/). In the future I might add a `setup.py` file to easily install with `pip`, but for now please use the crude installation as listed.

# Programs and their Functions
## `funcs.py` 
This program is simply a placeholder for all main functions used for analysis of cluster patterns. Each main function has its own documentation string. 
### Note on `xi_r`:
In the code, `xi_r_calc` has two definitions, one basic and one 'alt'. The basic one uses the Peebles & Hauser method of calculating the correlation function (n_DD/n_RR - 1). The 'alt' method uses the bit more complex Hamilton estimator (n_DD * n_RR / n_DR^2). Where n_DD, n_RR etc are the binned frequency values of pairwise distance catalogues. For example, n_DD is the 'data-data' set where the pairwise distances are calculated within the data set and n_DR is the 'data-random' set of cross correlated distances.

## `cluster_analysis.py`
Recommended first program to use as it gives the user a good overview of what the key programs do. User decides what functions to run and data files to examine. **Note:** The `factor` variable in the code is hard-coded and needs to be changed before running the program if desired.
