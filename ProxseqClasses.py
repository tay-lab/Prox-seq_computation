# -*- coding: utf-8 -*-
"""
Author: Hoang Van temp_change
Address: Pritzker School of Molecular Engineering
         The University of Chicago
         Chicago, IL 60637, USA

This file contains the functions used to simulate Prox-seq data, and the predictive
and linear regression methods for prediction of protein complex from Prox-seq
"""

# Import packages
import numpy as np
import math
import random # for sampling from a set without replacement
import pandas as pd

import scipy.spatial as spatial
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

import matplotlib.pyplot as plt

import copy
import datetime

# =============================================================================
# # Random point generators
# # Return an n-by-3 array, where the columns are x, y and z coordinates
# # Do NOT put seed number into the point generators!!!
# =============================================================================
def randomPointGen2D(n):
    '''
    Generate points on a sphere surface with unit radius

    Parameters
    ----------
    n : int
        Number of points

    Returns
    -------
    numpy array
        Return an n-by-3 array, where the columns are x, y and z coordinates

    '''
    # Generate n random points on the surface of a unit sphere
    # Ref: http://mathworld.wolfram.com/SpherePointPicking.html
    theta = np.random.uniform(0, 2*math.pi, size=(n,))
    z = np.random.uniform(-1,1, size=(n,))
    x = np.sqrt(1-z**2)*np.cos(theta)
    y = np.sqrt(1-z**2)*np.sin(theta)

    return np.array([x,y,z]).T

def randomPointGen3D(n):
    '''
    Generate points inside a sphere with unit radius

    Parameters
    ----------
    n : int
        Number of points

    Returns
    -------
    numpy array
        Return an n-by-3 array, where the columns are x, y and z coordinates

    '''
    # Generate n random points inside a unit sphere
    # Ref: https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238
    u = np.random.uniform(0,1, size=(n,))
    x = np.random.normal(size=(n,))
    y = np.random.normal(size=(n,))
    z = np.random.normal(size=(n,))

    rescale = np.power(u,1/3) / np.sqrt(x*x+y*y+z*z)

    return (np.array([x,y,z]) * rescale).T

# =============================================================================
# # Simulate single cell PLA data
# # Assume there are N protein targets, and some of the proteins form pairwise complexes
# # Assume non-complex proteins and protein complexes are randmoly distributed on a sphere
# # Assume saturated antibody binding: all proteins and protein complexes are bound by PLA probes
# # If a pair of PLA probes A and B are within a certain distance (ie, the ligation distance), they are ligated
# # If more than one pair of probes are within the ligation distance, there are 2 options: all of them are ligated, or only one pair is
#
# # Cell variance: gamma
# =============================================================================
class simulatePLA:
    '''
    A class to simulate PLA product counts in single cells.

    Parameters
    ----------
    cell_d : float
        The cell diameter in nanometer.
        Default is 10,000.

    PLA_dist : float
        The ligation distance in nanometer.
        Default is 50.

    n_cells : int
        The number of cells to simulate.

    mode : string
        '2D' (Prox-seq probes target cell surface proteins) or
        '3D' (Prox-seq probes target intracellular proteins).
        Default is '2D'.

    ligate_all : boolean, optional
        Whether only 1 PLA pair or all pairs are allowed to be ligated.
        Default is False.

    protein_variance : boolean
        Whether to simulate variance of protein/complex expression using a
        negative binomial distribution.
        Default is False.

    n_nbinom : float, optional
        n is the number of successes for negative binomial distribution for protein
        variance. p, the probability of success, is set such that the mean of the
        negative binomial distribution is equal to the desired mean.
        Increase this n parameter in order to decrease the protein variance.
        Default is 1.5.

    seed_num : float, optional
        The seed number for RNG.
        Default is 1.

    sep: string, optional
        The separator format for PLA product.
        Default is ':'.

    Attributes
    ----------
    pla_count : pandas data frame
        The simulated PLA count data.

    complex_count : pandas data frame
        The true complex abundance.

    probe_count : pandas data frame
        The true count of non-interacting probes.

    non_proximal_count : pandas data frame
        The count of  probes. In a Prox-seq experiments, these probes
        are not measured, because they are not ligated with any other probes.

    '''

    def __init__(self, cell_d=10000, PLA_dist=50, n_cells=100,
                 mode='2D', ligate_all=False,
                 protein_variance=False,
                 n_nbinom=1.5, seed_num=1, sep=':'):
        self.cell_d = cell_d
        self.PLA_dist = PLA_dist
        self.n_cells = n_cells
        self.protein_variance = protein_variance
        self.mode = mode
        self.ligate_all = ligate_all
        self.n_nbinom = n_nbinom
        self.seed_num = seed_num
        self.sep = sep

    def simulate(self, num_complex, probeA, probeB, verbose=True):
        '''
        Simulate PLA product count data

        Parameters
        ----------
        num_complex : list or numpy array
            An NA-by-NB array containing the number of complexes on each cell
            (NA and NB is the number of targets of probe A and B). Element [i,j] is
            the abundance of complex i:j.

        probeA : list or numpy array
            An NA-by-1 array containing the counts of non-interacting proteins
            bound by probe A. Element [i] is the abundance of non-complex forming
            protein i, bound by Prox-seq probe A.

        probeB : list or numpy array
            An NB-by-1 array containing the counts of non-interacting proteins
            bound by probe B. Element [j] is the abundance of non-complex forming
            protein j, bound by Prox-seq probe B.

        verbose : bool, optional
            Whether to print out the simulation progress.
            Default is True.

        '''

        # Convert to numpy arrays
        num_complex = np.array(num_complex)
        probeA = np.array(probeA)
        probeB = np.array(probeB)

        # Check for the length of input probe and complex arrays
        if len(probeA) != len(probeB):
            raise ValueError("probeA_ns and probeB_ns must have equal length!")
        if num_complex.shape[0] != num_complex.shape[1]:
            raise ValueError("num_complex must be a square 2D array!")
        if num_complex.shape[0] != len(probeA):
            raise ValueError("num_complex and probeA, probeB must have the same length!")

        # Simulation parameters
        print(f"cell_d={self.cell_d}. PLA_dist={self.PLA_dist}.")
        if self.protein_variance:
            print(f"protein_variance={self.protein_variance}. Negative binomial: n={self.n_nbinom}. seed_num={self.seed_num}.")
        else:
            print(f"protein_variance={self.protein_variance}.")

        self.num_complex = num_complex.copy()
        self.probeA = probeA.copy()
        self.probeB = probeB.copy()

        # Seed number
        np.random.seed(self.seed_num)
        random.seed(self.seed_num)

        # Initialize dge dictionary, each key is a single cell, each value is the cell's PLA count
        dge = {}

        # Dictionary to store actual complex abundance of each single cell
        dge_complex_true = {}

        # Dictionary to store actual non-complexing forming probe abundance of each single cell
        dge_probe_true = {}

        # Dictionary to store  probes of each single cell
        dge_non_proximal = {}


        # Protein variance: negative binomial distribution
        # np.random.negative_binomial(n, p, size)
        # n: number of successes
        # p: probability of success
        # mean = (1-p)*n/p ; var = (1-p)*n/p^2
        variance_probeA_ns = []
        variance_probeB_ns = []
        variance_complex = []
        if self.protein_variance:
            for i in range(len(self.probeA)):
                temp_scale = np.random.negative_binomial(n=self.n_nbinom, p=1/(1+self.probeA[i]/self.n_nbinom), size=self.n_cells)
                variance_probeA_ns.append(temp_scale)
                variance_probeB_ns.append(temp_scale*self.probeB[i]/self.probeA[i])

            for i in range(self.num_complex.shape[0]):
                variance_complex.append([])
                for j in range(self.num_complex.shape[1]):
                    if num_complex[i,j] == 0:
                        variance_complex[i].append(np.zeros(self.n_cells))
                    else:
                        if i <= j:
                            variance_complex[i].append(np.random.negative_binomial(n=self.n_nbinom, p=1/(1+self.num_complex[i,j]/self.n_nbinom), size=self.n_cells))
                        else: # ensure cij is correlated with cji
                            variance_complex[i].append(variance_complex[j][i]*num_complex[i,j]/num_complex[j,i])
            variance_probeA_ns = np.array(variance_probeA_ns)
            variance_probeB_ns = np.array(variance_probeB_ns)
            variance_complex = np.reshape(np.array(variance_complex),
                                          (self.num_complex.shape[0],self.num_complex.shape[1],self.n_cells))

        # Start simulation
        # Iterate through each single cell
        print(f'{datetime.datetime.now().replace(microsecond=0)}     Start simulation')
        for cell_i in range(self.n_cells):

            # Initialize the count dictionary of each single cell
            dge[cell_i] = {f'{i+1}{self.sep}{j+1}':0 for i in range(len(self.probeA)) for j in range(len(self.probeB))}

            # Add protein variance
            if self.protein_variance:
                probeA_i = copy.deepcopy(variance_probeA_ns[:,cell_i]).astype(int)
                probeB_i = copy.deepcopy(variance_probeB_ns[:,cell_i]).astype(int)
                num_complex_i = copy.deepcopy(variance_complex[:,:,cell_i]).astype(int)
            else:
                probeA_i = copy.deepcopy(probeA).astype(int)
                probeB_i = copy.deepcopy(probeB).astype(int)
                num_complex_i = copy.deepcopy(num_complex).astype(int)

            # Save the true complex abundance
            dge_complex_true[cell_i] = num_complex_i.reshape(-1,)

            # Save the true non-complex forming abundance
            dge_probe_true[cell_i] = np.hstack((probeA_i, probeB_i))

            # Generate probes A and B (ie, non-complex-forming probes)
            if probeA.sum() > 0:
                if self.mode == '2D':
                    temp_probeA_i = self.cell_d/2*randomPointGen2D(probeA_i.sum())
                elif self.mode == '3D':
                    temp_probeA_i = self.cell_d/2*randomPointGen3D(probeA_i.sum())
            # Target id of each probe A molecule
            temp_probeA_i_targets = np.repeat([f"{s+1}" for s in range(len(probeA_i))], probeA_i)
            if probeB.sum() > 0:
                if self.mode == '2D':
                    temp_probeB_i = self.cell_d/2*randomPointGen2D(probeB_i.sum())
                elif self.mode == '3D':
                    temp_probeB_i = self.cell_d/2*randomPointGen3D(probeB_i.sum())
            # Target id of each probe B molecule
            temp_probeB_i_targets = np.repeat([f"{s+1}" for s in range(len(probeB_i))], probeB_i)

            # Generate the protein complexes
            if self.mode == '2D':
                temp_complex_i = self.cell_d/2*randomPointGen2D(num_complex_i.sum())
            elif self.mode == '3D':
                temp_complex_i = self.cell_d/2*randomPointGen3D(num_complex_i.sum())

            # Target names of probes A and B that bind to the protein complexes
            complex_probeA_targets = [[f"{s+1}" for _ in range(num_complex_i.shape[1])] for s in range(num_complex_i.shape[0])]
            complex_probeA_targets = np.repeat(complex_probeA_targets, num_complex_i.flatten())
            complex_probeB_targets = [[f"{s+1}" for s in range(num_complex_i.shape[1])] for _ in range(num_complex_i.shape[0])]
            complex_probeB_targets = np.repeat(complex_probeB_targets, num_complex_i.flatten())

            # Combine non-complex and complex probes
            # x,y,z coordinates
            probeA_i = np.vstack((temp_probeA_i, copy.deepcopy(temp_complex_i)))
            probeB_i = np.vstack((temp_probeB_i, copy.deepcopy(temp_complex_i)))
            # Probe target name
            probeA_targets = np.concatenate((temp_probeA_i_targets, complex_probeA_targets))
            probeB_targets = np.concatenate((temp_probeB_i_targets, complex_probeB_targets))

            # Calculate pairwise euclidean distance
            pairwise_dist = spatial.distance.cdist(probeA_i, probeB_i, metric="euclidean")
            # pairwise_dist[i,j] = distance between probeA_i[i] and probeB_i[j]

            # Ligation =========
            # Go through each probe A, then see if it can ligate with any probe B
            probeB_blacklist = set([]) # index of blacklisted probes B, which are excluded from future ligation in ligate_all=False
            probeA_blacklist = set([]) # index of ligated probes A, used for non_proximal_count

            # Find pairs within ligation distance
            valid_pairs = np.argwhere(pairwise_dist <= self.PLA_dist)
            # Shuffle the pairs
            np.random.shuffle(valid_pairs)

            # Iterate
            for i in valid_pairs:
                if self.ligate_all:
                    dge[cell_i][f"{probeA_targets[i[0]]}{self.sep}{probeB_targets[i[1]]}"] += 1

                else:
                    if (i[0] in probeA_blacklist) or (i[1] in probeB_blacklist):
                        continue
                    else:
                        dge[cell_i][f"{probeA_targets[i[0]]}{self.sep}{probeB_targets[i[1]]}"] += 1

                probeA_blacklist.add(i[0])
                probeB_blacklist.add(i[1])

            # Tally the  probe count
            dge_non_proximal[cell_i] = {}
            probeA_non_proximal = np.ones(probeA_targets.shape, dtype=bool)
            probeA_non_proximal[list(probeA_blacklist)] = False
            probeA_non_proximal = np.unique(probeA_targets[probeA_non_proximal], return_counts=True)
            for i, j in zip(*probeA_non_proximal):
                dge_non_proximal[cell_i][f"{i}_A"] = j
            probeB_non_proximal = np.ones(probeB_targets.shape, dtype=bool)
            probeB_non_proximal[list(probeB_blacklist)] = False
            probeB_non_proximal = np.unique(probeB_targets[probeB_non_proximal], return_counts=True)
            for i, j in zip(*probeB_non_proximal):
                dge_non_proximal[cell_i][f"{i}_B"] = j

            # Keep track of time
            if verbose:
                if (cell_i+1) % 10 == 0:
                    print(f'{datetime.datetime.now().replace(microsecond=0)}     Processed {cell_i+1:>6} cells')

        # Convert dictionary to pandas data frame
        self.pla_count = pd.DataFrame(dge)

        self.complex_count = pd.DataFrame(dge_complex_true,
                                          index=[f'{i+1}{self.sep}{j+1}'
                                                 for i in range(num_complex.shape[0]) for j in range(num_complex.shape[1])])

        self.probe_count = pd.DataFrame(dge_probe_true, index=[f"{i+1}_A"for i in range(num_complex.shape[0])] +
                                        [f"{i+1}_B" for i in range(num_complex.shape[1])])

        self.non_proximal_count = pd.DataFrame(dge_non_proximal)
        self.non_proximal_count.fillna(value=0, inplace=True) # replace nan with 0



# =============================================================================
# # Class to import PLA count data
# # Can be used to calculate protein abundance and expected count, and predict
# # complex count
# =============================================================================
class plaObject:
    '''
    Import PLA product count data. Rows are PLA products, columns are single cells.
    Can be used to calculate protein abundance, expected count, and predict
    protein complex count

    Parameters
    ----------
    data : pandas data frame
        Data frame of PLA count. Columns are cell barcodes, rows are PLA products.

    non_proximal_marker : string, option
        The name for marker of probeB_non_proximal PLA products. If a string, the marker
        will be used to extract probeB_non_proximal count of each probe. For example, if
        non_proximal_marker="oligo", then PLA product "CD3:oligo" is understood as
        probeB_non_proximal count of CD3 probe A, ie "CD3_A".
        If None, then non-proximal count is not extracted.
        Default is None.

    sep : string, optional
        The separator convention in the names of PLA complexes.
        Default is ':'.

    Attributes
    ----------
    pla_count: pandas data frame
        Imported PLA product count data.

    protein_count : pandas data frame
        Calculate protein count from PLA count data. Output is comparable to
        CITE-seq and REAP-seq.

    proteins : list
        List of detected proteins.

    pla_expected : pandas data frame
        Expected PLA count if no protein complexes exist.

    complex_count : pandas data frame
        Predicted protein complex count. May have a suffix depending on the
        settings of predictComplex().

    pla_probe_count : pandas data frame
        The count of each Prox-seq probe A and B, calculted from the detected
        PLA products.

    tol_ : numpy array
        Array of tolerance values for each iteration for predictComplex 'iterative'
        method. This is used as the convergence criterion.

    shape : tuple
        Shape of PLA product count matrix.

    sep : string
        Delimiter of PLA products. Example: for CD3:CD3, sep is ':'

    lr_params : pandas data frame
        Contains the parameters of the weighted least squares models, obtained
        from the linear regression (LR) method
        Columns: intercept , slope , SE_intercept , SE_slope ,
        intercept P value , slope P value , df

    complex_fisher : pandas data frame
        P-value of protein complex expression, calculated using one-sided
        Fisher's exact test


    '''
    def __init__(self, data, non_proximal_marker=None, sep=':'):

        if non_proximal_marker is None:
            self.pla_count = data.copy()
            self.non_proximal_count = None

        else:
            probeA = np.array([s.split(sep)[0] for s in data.index])
            probeB = np.array([s.split(sep)[1] for s in data.index])
            self.pla_count = data.loc[(probeA!=non_proximal_marker) & (probeB!=non_proximal_marker),:].copy()

            # Extract non_proximal_count data
            self.non_proximal_count = data.loc[(probeA!=non_proximal_marker) ^ (probeB!=non_proximal_marker),:].copy()
            new_index = []
            for i in self.non_proximal_count.index:
                k,j = i.split(sep)
                if k == non_proximal_marker:
                    new_index.append(f"{j}_B")
                elif j == non_proximal_marker:
                    new_index.append(f"{k}_A")
            self.non_proximal_count.index = new_index

        self.sep = sep
        self.shape = data.shape


    def calculateProteinCount(self):
        '''
        Calculate protein count
        '''
        # Get AB1 and AB2 of each row of data
        AB1 = np.array([s.split(self.sep)[0] for s in self.pla_count.index])
        AB2 = np.array([s.split(self.sep)[1] for s in self.pla_count.index])

        # Get the unique antibody targets
        AB_unique = np.unique(np.concatenate((AB1,AB2)))
        AB_unique.sort()

        # Save list of proteins
        self.proteins = list(AB_unique)

        # Initialize output self.pla_countframes
        self.protein_count = pd.DataFrame(0, index=AB_unique, columns=self.pla_count.columns)

        for i in self.protein_count.index:
            self.protein_count.loc[i,:] = (self.pla_count.loc[AB1==i,:]).sum(axis=0) + (self.pla_count.loc[AB2==i,:]).sum(axis=0)


    def calculateExpected(self):
        '''
        Calculate the expected random count of all PLA products.

        '''

        # Initialize output
        self.pla_expected = pd.DataFrame(columns=self.pla_count.columns, index=self.pla_count.index)

        # Get AB1 and AB2 of each row of data
        AB1 = np.array([s.split(self.sep)[0] for s in self.pla_count.index])
        AB2 = np.array([s.split(self.sep)[1] for s in self.pla_count.index])
        for i in self.pla_count.index:
            self.pla_expected.loc[i,:] = self.pla_count.loc[AB1==i.split(self.sep)[0],:].sum(axis=0)*self.pla_count.loc[AB2==i.split(self.sep)[1],:].sum(axis=0)/self.pla_count.sum(axis=0)

        # Replace 0 divided by 0 with 0
        self.pla_expected.fillna(0, inplace=True)

    def calculateProbeCount(self):
        '''
        Calculate the counts of probes A and B of each target from the counts
        of PLA products.

        '''

        # Get AB1 and AB2 of each row of data
        AB1 = np.array([s.split(self.sep)[0] for s in self.pla_count.index])
        AB2 = np.array([s.split(self.sep)[1] for s in self.pla_count.index])

        # Get the unique AB1 and AB2 probe targets
        AB1_unique = list(set(AB1))
        AB2_unique = list(set(AB2))
        AB1_unique.sort()
        AB2_unique.sort()

        # Initialize temporary self.pla_count frames
        output1 = pd.DataFrame(0, index=AB1_unique, columns=self.pla_count.columns) # store abundance of all probe A
        output2 = pd.DataFrame(0, index=AB2_unique, columns=self.pla_count.columns) # store abundance of all probe B

        for i in output1.index:
            output1.loc[i,:] = self.pla_count.loc[AB1==i,:].sum(axis=0)
        for i in output2.index:
            output2.loc[i,:] = self.pla_count.loc[AB2==i,:].sum(axis=0)
        output1.index = [f"{i}_A" for i in output1.index]
        output2.index = [f"{i}_B" for i in output2.index]

        self.pla_probe_count = pd.concat([output1,output2])

    def predictComplex(self, method='iterative', non_proximal_count=None, scale=1,
                       intercept_cutoff=1, slope_cutoff=0,
                       mean_cutoff=1, p_cutoff=0.05,
                       non_interacting=None,
                       p_adjust=True, sym_weight=0.25, df_guess=None,
                       nIter=200, tol=1, suffix=''):
        '''
        Predict complex count with two methods, 'iterative' and 'lr'.

        Parameters
        ----------
        method : string, optional
            Whether to use the 'iterative' or 'lr' method for predicting complex count.
            lr method: use  count and weighted least squares regression.
            iterative method: iteratively solve a system of quadratic equations.
            Default is 'iterative'.

        ========== 'lr' method ==========
        non_proximal_count : pandas data frame
            Count of  probes. The row names of the data frame have the
            label of A and B to indicate probe A and B. For example: CD3_A and
            CD3_B.
            If None, use the non_proximal_count currently stored.
            Default is None.

        scale : float, option
            A positive scaling factor for linear regression. The product of
            non-proximal probe counts is divided by this factor before fitting,
            in order to make the linear regression more stable.
            Default is 1.

        intercept_cutoff : float, optional
            The value for the intercept under the null hypothesis.
            Default is 1.

        slope_cutoff : float, optional
            The value for the intercept under the null hypothesis.
            Default is 0.

        p_cutoff : boolean, optional
            The P-value to reject the null hypothesis.
            Default is 0.01.


        ========== 'iterative' method ==========
        non_interacting : list, optional
            List of PLA products or proteins that do no form protein complexes.
            Example: [X:Y] means X:Y does not form a complex, while [X] means X does
            not form complexes with any other proteins.
            If None, use an empty list.
            Default is None.

        mean_cutoff : float, optional
            PLA products whose estimated complex abundance at each iteration
            fails the 1-sided t-test sample mean>mean_cutoff is kept as 0.
            Only one of mean_cutoff or mean_frac_cutoff can be specified,
            the other has to be None.
            Default is 1.

        p_cutoff : boolean, optional
            The P-value to reject the null hypothesis.
            Default is 0.05.

        p_adjust : boolean
            Whether to perform FDR correction for the one-sided t-test.
            Default is True.

        sym_weight : float (0 <= sym_weight <= 1), optional
            The weight factor used to enforce symmetry condition.

        df_guess : pandas data frame, optional
            First guesses of true complex abundance (must be the same shape as data).
            If None (the default), use 0 as the first guess.

        nIter : int, optional
            Max number of iterations to perform.
            Default is 200.

        tol : float, optional
            The tolerance threshold for convergence. The prediction converges
            if the max absolute difference between new and old predicted
            complex count is sufficiently low.
            Default is 1.

        ========== Both methods ==========
        suffix : string, optional
            Add a suffix to the attribute complex_count and tol_ to distinguish
            from different settings of the predictComplex() method.
            Default is ''.

        '''


        if method == 'lr':
            # Check if non_proximal_count data is available
            if non_proximal_count is None:
                if self.non_proximal_count is None:
                    raise TypeError("Missing non_proximal_count data for LR method.")
            else:
                self.non_proximal_count = non_proximal_count.copy()

            # Check if the scaling factor is positive
            if scale <= 0:
                raise ValueError("scale must be a positive number.")

            # Initialize the complex_count data frame
            complex_count = pd.DataFrame(0, index=self.pla_count.index, columns=self.pla_count.columns)


            # Store WLS parameters
            LS_out = pd.DataFrame(np.nan, index=self.pla_count.index,
                                   columns=["intercept","pval_intercept","slope","pval_slope"])

            # Iterate through each PLA product and perform linear regression
            for i in self.pla_count.index:
                # Get targets of probe A and B
                probeA, probeB = i.split(self.sep)

                # Set up variables

                X = (self.non_proximal_count.loc[f"{probeA}_A",:]*self.non_proximal_count.loc[f"{probeB}_B",:]/scale).to_numpy() # scale the product by the scale factor
                # Multiple OLS
                # X = np.vstack((self.non_proximal_count.loc[f"{probeA}_A",:],
                #                self.non_proximal_count.loc[f"{probeB}_B",:],
                #                self.non_proximal_count.loc[f"{probeA}_A",:]*self.non_proximal_count.loc[f"{probeB}_B",:]/scale)).T

                X = sm.add_constant(X, prepend=True)
                y = self.pla_count.loc[i,:].to_numpy()

                # Ordinary least squares
                # results = sm.OLS(y, X).fit()

                # Weighted least squares
                mask = X[:,1] > 0
                # Only look at PLA products with random ligation noise in at least 3 cells
                if sum(mask) < 3:
                    continue
                results = sm.WLS(y[mask], X[mask,:], weights=1/X[mask,1]).fit()

                # One-sided t-test on both intercepts and slope
                # Intercept: one-sided t-test (mean above mean_cutoff)
                t_intercept = (results.params[0] - intercept_cutoff)/results.bse[0]
                # Slope: one-sided t-test (mean above 0)
                t_slope = (results.params[1] - slope_cutoff)/results.bse[1]
                # t_slopeB = (results.params[2]-0)/results.bse[2]
                # t_slopeAB = (results.params[3]-0)/results.bse[3]

                # Store the LS parameters
                LS_out.loc[i,:] = [results.params[0],
                                    1 - stats.t.cdf(t_intercept, df=results.df_resid),
                                    results.params[1],
                                    1 - stats.t.cdf(t_slope, df=results.df_resid)]

                # Store the predicted complex count
                complex_count.loc[i,:] = self.pla_count.loc[i,:] - X[:,1]*results.params[1]

            # FDR correction
            LS_out.loc[:,"fdr_intercept"] = np.nan
            mask = ~LS_out.loc[:,"pval_intercept"].isna()
            LS_out.loc[mask,"fdr_intercept"] = multipletests(LS_out.loc[mask,"pval_intercept"], method='fdr_bh')[1]
            LS_out.loc[:,"fdr_slope"] = np.nan
            mask = ~LS_out.loc[:,"pval_slope"].isna()
            LS_out.loc[mask,"fdr_slope"] = multipletests(LS_out.loc[mask,"pval_slope"], method='fdr_bh')[1]

            # Calculate complex count
            for i in complex_count.index:
                # Get targets of probe A and B
                probeA, probeB = i.split(self.sep)

                # Filter out complexes that fail the intercept test
                if (LS_out.at[i,"fdr_intercept"] > p_cutoff):
                    complex_count.loc[i,:] = 0
                    # if (LS_out.at[i,"fdr_slope"] <= p_cutoff):
                    #     complex_count.loc[i,:] = self.pla_count.loc[i,:] - LS_out.at[i,"slope"]*self.non_proximal_count.loc[f"{probeA}_A",:]*self.non_proximal_count.loc[f"{probeB}_B",:]/scale
                    # else: # slope is 0
                    #     complex_count.loc[i,:] = self.pla_count.loc[i,:]


            # Set minimum count to 0
            complex_count[complex_count<0] = 0

            setattr(self, f"complex_count{suffix}",
                    pd.DataFrame(data=complex_count.round().astype(np.int64), index=self.pla_count.index, columns=self.pla_count.columns))

            # return LS_out
            setattr(self, f"lr_params{suffix}", LS_out)

        elif method == 'iterative':

            # Store tolerane values
            tol_ = []

            # Check sym_weight
            if not (0 <= sym_weight <= 1):
                raise ValueError("sym_weight has to be within 0 to 1.")

            # Convert input data frame into numpy array
            pla = self.pla_count.to_numpy(copy=True)

            # Convert list to set
            if non_interacting is None:
                non_interacting = []
            non_interacting = set(non_interacting)

            # Set of PLA products
            pla_product_set = set(self.pla_count.index)

            # Get a list of probe A and B targets
            probeA = np.array([s.split(self.sep)[0] for s in self.pla_count.index])
            probeB = np.array([s.split(self.sep)[1] for s in self.pla_count.index])

            # Initialize a numpy array to store estimated complex amount
            if df_guess is None:
                complex_out = np.zeros(pla.shape)
            else:
                complex_out = df_guess.values

            # Iteration
            loop_num = 0
            max_change = tol + 1
            while (loop_num < nIter) and (max_change > tol):

                # Dict to store the one-sided t-test p-values
                tp_all = {}

                # PLA product count minus previous iteration's complex count
                temp_pla = pla - complex_out

                # Calculate the sum of probe A and B
                temp_pla_probeA = {}
                for i in set(probeA):
                    temp_pla_probeA[i] = temp_pla[probeA==i,:].sum(axis=0)
                temp_pla_probeB = {}
                for i in set(probeB):
                    temp_pla_probeB[i] = temp_pla[probeB==i,:].sum(axis=0)
                temp_pla_sum = temp_pla.sum(axis=0)

                # First pass: get all the p-values
                for i in range(self.pla_count.shape[0]):

                    # if this PLA product is not detected in any cells, skip
                    if np.sum(pla[i,:]) == 0:
                        continue

                    temp_complex = self.pla_count.index[i]
                    temp_probeA, temp_probeB = temp_complex.split(self.sep) # target of probe A and B

                    # Apply the constraints
                    if (temp_complex in non_interacting) or (temp_probeA in non_interacting) or (temp_probeB in non_interacting):
                        continue

                    # Expected PLA product count
                    temp_expected = temp_pla_probeA[temp_probeA]*temp_pla_probeB[temp_probeB]/temp_pla_sum

                    # Updated complex count
                    temp_diff = pla[i,:] - temp_expected

                    # Check to see if the estimated abundance passes the mean_cutoff (old stats version doesn't have 'alternative' option)
                    # Ha: sample mean > mean_cutoff
                    tval, tp = stats.ttest_1samp(temp_diff, mean_cutoff)
                    if (tval > 0):
                        tp_all[self.pla_count.index[i]] = tp/2
                    else:
                        tp_all[self.pla_count.index[i]] = 1-tp/2

                # Convert p-values dictionary to series
                tp_all = pd.Series(tp_all)
                # Multiple comparison correction
                if p_adjust:
                    _, tp_adj, _,_ = multipletests(tp_all.to_numpy(), alpha=p_cutoff, method='fdr_bh')
                    tp_adj = pd.Series(tp_adj, index=tp_all.index)
                else:
                    tp_adj = tp_all

                # Array to store the change in the complex estimates
                temp_change = np.zeros(pla.shape) + tol + 1
                # Second pass: calculate protein complex
                for i in range(self.pla_count.shape[0]):

                    # if this PLA product is not detected in any cell, skip
                    if np.sum(pla[i,:]) == 0:
                        temp_change[i,:] = 0
                        continue

                    # target of probe A and B
                    temp_complex = self.pla_count.index[i]
                    temp_probeA, temp_probeB = temp_complex.split(self.sep)

                    # Apply the constraints
                    if (temp_complex in non_interacting) or (temp_probeA in non_interacting) or (temp_probeB in non_interacting):
                        temp_change[i,:] = 0
                        continue

                    # Check to see if the estimated abundance passes the mean_cutoff
                    # Ha: sample mean > mean_cutoff
                    if (tp_adj[self.pla_count.index[i]] <= p_cutoff):
                        temp_expected = temp_pla_probeA[temp_probeA]*temp_pla_probeB[temp_probeB]/temp_pla_sum
                        temp_diff = pla[i,:] - temp_expected

                    elif (f"{temp_probeB}{self.sep}{temp_probeA}" in pla_product_set):
                        # check for symmetry
                        temp_symmetry = complex_out[self.pla_count.index==f"{temp_probeB}{self.sep}{temp_probeA}",:]
                        if np.mean(temp_symmetry) > mean_cutoff:
                            temp_diff = sym_weight*temp_symmetry
                        else:
                            temp_change[i,:] = 0
                            continue
                    else:
                        temp_change[i,:] = 0
                        continue

                    # Force negative values to be zero <---- should be done after t-test
                    temp_diff[temp_diff < 0] = 0

                    # Check if observed is 0 but estimated is non 0, then force the estimated to be 0
                    # This should only be done after t-test
                    temp_diff[(temp_diff > 0) & (pla[i,:] == 0)] = 0

                    # Store convergence information
                    temp_change[i,:] = temp_diff - complex_out[i,:]

                    # Store the new solutions/estimates
                    complex_out[i,:] = temp_diff

                # Round the adjustment amount
                complex_out = np.round(complex_out)
                # Save the maximum change in the solution for convergence check
                max_change = np.abs(temp_change).max()

                tol_.append(max_change)

                loop_num += 1

            print(f"predictComplex done: Loop number {loop_num}, tolerance {max_change:.2f}")

            setattr(self, f"complex_count{suffix}", pd.DataFrame(data=complex_out, index=self.pla_count.index, columns=self.pla_count.columns))
            setattr(self, f"tol{suffix}_", np.array(tol_))

        elif method == "test": # calculate random ligation from PLA products
            # random ligation of PLA product i:j = abundance of probe Ai * abundance of probe Bj
            # abundance of probe Ai = sum of Xi,k over k != j
            # abundance of probe Bj = sum of Xk,j over k != i

            # Set up variables
            # Initialize the complex_count data frame
            complex_count = pd.DataFrame(0, index=self.pla_count.index, columns=self.pla_count.columns)


            # Store WLS parameters
            LS_out = pd.DataFrame(np.nan, index=self.pla_count.index,
                                   columns=["intercept","pval_intercept","slope","pval_slope"])

            # Iterate through each PLA product and perform linear regression
            for i in self.pla_count.index:
                # Get targets of probe A and B
                probeA, probeB = i.split(self.sep)

                # Set up the variables for regression
                X = ((self.pla_probe_count.loc[f"{probeA}_A",:]-self.pla_count.loc[i,:])*
                     (self.pla_probe_count.loc[f"{probeB}_B",:]-self.pla_count.loc[i,:])).to_numpy()
                # Multiple OLS
                # X = np.vstack((self.non_proximal_count.loc[f"{probeA}_A",:],
                #                self.non_proximal_count.loc[f"{probeB}_B",:],
                #                self.non_proximal_count.loc[f"{probeA}_A",:]*self.non_proximal_count.loc[f"{probeB}_B",:]/scale)).T

                X = sm.add_constant(X, prepend=True)
                y = self.pla_count.loc[i,:].to_numpy()

                # Ordinary least squares
                results = sm.OLS(y, X).fit()

                # # Weighted least squares
                # mask = X[:,1] > 0
                # # Only look at PLA products with random ligation noise in at least 3 cells
                # if sum(mask) < 3:
                #     continue
                # results = sm.WLS(y[mask], X[mask,:], weights=1/X[mask,1]).fit()

                # One-sided t-test on both intercepts and slope
                # Intercept: one-sided t-test (mean above mean_cutoff)
                t_intercept = (results.params[0]-mean_cutoff)/results.bse[0]
                # Slope: one-sided t-test (mean above 0)
                t_slope = (results.params[1]-0)/results.bse[1]
                # t_slopeB = (results.params[2]-0)/results.bse[2]
                # t_slopeAB = (results.params[3]-0)/results.bse[3]

                # Store the LS parameters
                LS_out.loc[i,:] = [results.params[0],
                                    1 - stats.t.cdf(t_intercept, df=results.df_resid),
                                    results.params[1],
                                    1 - stats.t.cdf(t_slope, df=results.df_resid)]

                # Store the predicted complex count
                complex_count.loc[i,:] = self.pla_count.loc[i,:] - X[:,1]*results.params[1]

            # FDR correction
            LS_out.loc[:,"fdr_intercept"] = np.nan
            mask = ~LS_out.loc[:,"pval_intercept"].isna()
            LS_out.loc[mask,"fdr_intercept"] = multipletests(LS_out.loc[mask,"pval_intercept"], method='fdr_bh')[1]
            LS_out.loc[:,"fdr_slope"] = np.nan
            mask = ~LS_out.loc[:,"pval_slope"].isna()
            LS_out.loc[mask,"fdr_slope"] = multipletests(LS_out.loc[mask,"pval_slope"], method='fdr_bh')[1]

            # Calculate complex count
            for i in complex_count.index:
                # Get targets of probe A and B
                probeA, probeB = i.split(self.sep)

                if not (LS_out.at[i,"fdr_intercept"] <= p_cutoff):
                    complex_count.loc[i,:] = 0
                    # if (LS_out.at[i,"fdr_slope"] <= p_cutoff):
                    #     complex_count.loc[i,:] = self.pla_count.loc[i,:] - LS_out.at[i,"slope"]*self.non_proximal_count.loc[f"{probeA}_A",:]*self.non_proximal_count.loc[f"{probeB}_B",:]/scale
                    # else: # slope is 0
                    #     complex_count.loc[i,:] = self.pla_count.loc[i,:]


            # Set minimum count to 0
            complex_count[complex_count<0] = 0

            setattr(self, f"complex_count{suffix}",
                    pd.DataFrame(data=complex_count.round().astype(np.int64), index=self.pla_count.index, columns=self.pla_count.columns))

            # return LS_out
            setattr(self, f"lr_params{suffix}", LS_out)

    def predictComplexFisher(self):
        '''
        Predict complex expression in single-cells using one-sided Fisher's
        exact test, and return a BH-corrected P-value.

        '''
        fisher_p = pd.DataFrame(np.nan, index=self.pla_count.index, columns=self.pla_count.columns)
        probeA = np.array([s.split(':')[0] for s in fisher_p.index])
        probeB = np.array([s.split(':')[1] for s in fisher_p.index])
        for i in fisher_p.index:
            tempA, tempB = i.split(':')
            x00 = (probeA == tempA) & (probeB == tempB)
            x01 = (probeA == tempA) & (probeB != tempB)
            x10 = (probeA != tempA) & (probeB == tempB)
            x11 = (probeA != tempA) & (probeB != tempB)
            temp01 = self.pla_count.loc[x01,:].sum(axis=0)
            temp10 = self.pla_count.loc[x10,:].sum(axis=0)
            temp11 = self.pla_count.loc[x11,:].sum(axis=0)
            for j in fisher_p.columns:
                fisher_p.at[i,j] = stats.fisher_exact([[self.pla_count.loc[x00,j], temp01[j]],
                                                            [temp10[j], temp11[j]]], alternative='greater')[1]

        # FDR correction: single cell by single cell
        self.complex_fisher = pd.DataFrame(np.nan, index=self.pla_count.index, columns=self.pla_count.columns)
        for i in self.complex_fisher.columns:
            self.complex_fisher.loc[:,i] = multipletests(fisher_p.loc[:,i], method='fdr_bh')[1]
