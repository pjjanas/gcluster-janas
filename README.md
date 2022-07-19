# janas-gcluster
Repository for a university based project on galaxy clustering for which the code was developed by Pawel Janas with the help of Prof. Peter Coles at Maynooth University, Ireland 2022.

# Introduction
Welcome to `janas-gcluster`! The programs found in this repositary simulate galaxy distributions in a simple cubic geometry and run specific analysis algorithms on them such as the two-point correlation function. I created this repository as a placeholder for my project based code/files for a project I completed with Maynooth University as part of their SPUR program. In this README document, I will outline the basics of using my code, specific function/variable names and ways to improve my code for future use.

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

# Data
Feel free to use the simulated data created using `power_law_model.py` `pairwise_cluster_model` etc or use your own. **NB:** Only 3 columned x, y and z data is compatible with the programs and each `.txt` data file follows the following format:
```
x        y         z (headings to be read in separately)
(data lies here)
```

# Programs and their Functions
## `funcs.py` 
This program is simply a placeholder for all main functions used for analysis of cluster patterns. Each main function has its own documentation string. 
### Note on `xi_r`:
In the code, `xi_r_calc` has two definitions, one basic and one 'alt'. The basic one uses the Peebles & Hauser method of calculating the correlation function (xi(r) = n_DD/n_RR - 1). The 'alt' method uses the bit more complex Hamilton estimator (xi(r) = n_DD * n_RR / n_DR^2 - 1). Where n_DD, n_RR etc are the binned frequency values of pairwise distance catalogues. For example, n_DD is the 'data-data' set where the pairwise distances are calculated within the data set and n_DR is the 'data-random' set of cross correlated distances.

## `cluster_analysis.py`
Recommended first program to use as it gives the user a good overview of what the key programs do. User decides what functions to run and data files to examine. The random catalogue for n_RR, n_DR calculation is simulated at each runtime (this is not the case for `monte_carlo_sim.py`. **Note:** The `factor` variable in the code is hard-coded and needs to be changed before running the program if desired.

## `monte_carlo_sim.py`
This is the heart of `janas-gcluster` API. This program does similar analysis methods to `cluster_analysis.py` but implements a Monte Carlo approach to calculate xi(r). This is done to see if any trend arises in xi(r). See below pictures of how the two-point correation function (xi(r)) changes with increasing MC run count for two random catalogues.

<img src="https://user-images.githubusercontent.com/81090178/179813409-08d36eea-49a7-4f57-890e-37d166a8dc29.png" width="400">
<img src="https://user-images.githubusercontent.com/81090178/179813410-012e6dab-1c2e-433c-9c75-19e75fb142e2.png" width="400">
<img src="https://user-images.githubusercontent.com/81090178/179813399-afb76a35-f59a-4e2c-90b1-23250fbeeb42.png" width="400">
<img src="https://user-images.githubusercontent.com/81090178/179813406-1ab74776-1b49-4a74-aeaf-7f0dc09ee798.png" width="400">

The variables in this program need to be changed via editor such as `MCRUNS` `file_name` and `Nbins` so please make sure to check them. You can plot xi(r) at powers of 10 but I don't recommend this for higher run counts as it eats up memory so comment that block out if necessary.

## `SPUR_clustering_v4.py`
This program is mainly deprecated which was used for early testing but it has one capability that other programs have not; you can define a minimum and maximum bound for the total galaxy number **N** and run analysis tools similar to what is in `cluster_analysis.py`. The user defines these bounds and also chooses how many steps to take from `nmin` to `nmax`. Also calculates the mean and variance of xi(r) for increasing N. If using Spyder as IDE and other IDEs, variables will be stored for checking. `galaxy_count` needs to be edited as the arrays generated are too big. **Note:** This program contains a lot of functions that are not in `funcs.py` and there may be some duplicate-like appearances so be careful.

# Data Generation
The following programs were used to simulate 3D data where each point represents a galaxy. Please uncomment the `file_write` lines to save a custom data file (make sure to choose a name for the file).

## `power_law_model.py`
This simulates clusters around 10% of the initial random catalogue with a density profile proportional to r^-1.8. This was done by creating a new probability density function (pdf) and then sampling r values from it for the cluster generation algorithm. This was achieved by subclassing `scipy` for a custom pdf. The direction at which to place each galaxy is also randomly chosen from 3 normal distributions (`sample_spherical`) to avoid corner densities at pole of each cluster.

## `pairwise_cluster_model.py`
Gives 10% of the total catalogue a 'pair' at a fixed distance r. This data-set takes many Monte Carlo runs to achieve some steady state of the correlation function xi(r). Would recommend changing variable `N` to place more points (galaxies) at fixed distance r for each cluster.

## `poisson_cluster_model.py`
Same as `power_law_model` except r values are chosen at random instead of from a custom density function like above.
