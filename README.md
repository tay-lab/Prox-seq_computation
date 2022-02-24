# Prox-seq_computation
A computational framework for simulating and analyzing Prox-seq data

For introduction to Prox-seq, including how to align Prox-seq sequencing data, visit https://github.com/tay-lab/Prox-seq

All relevant functions and classes are included in [ProxseqClasses.py](https://github.com/tay-lab/Prox-seq_computation/blob/main/ProxseqClasses.py). To use them, simply download the py files to your working directory and import it. Details of the functions and classes are available in the next section.

For a tutorial on how to use the framework, please refer to [Prox-seq_simulation_tutorial.ipynb](https://github.com/tay-lab/Prox-seq_computation/blob/main/Prox-seq_simulation_tutorial.ipynb) and the tutorial [dataset](https://github.com/tay-lab/Prox-seq_computation/blob/main/tutorial_count_matrix.txt.gz).

## Details of ProxseqClasses.py

### Simulation of Prox-seq data
**Class name: simulatePLA**
    '''
    A class to simulate PLA counts of a cocktail of N targets.

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
        '2D' (Prox-seq probes target cell surface proteins) or '3D' (Prox-seq probes target intracellular proteins).
        Default is '2D'.

    ligate_all : boolean, optional
        Whether only 1 PLA pair or all pairs are allowed to be ligated.
        Default is False.

    protein_variance : boolean
        Whether to simulate variance of protein/complex expression using a negative binomial distribution.
        Default is False.

    n_nbinom : float, optional
        n is the number of successes for negative binomial distribution for protein variance. p, the probability of success, is set such that the mean of the
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
        The count of  probes. In a Prox-seq experiments, these probes are not measured, because they are not ligated with any other probes.

#### Attributes of class simulatePLA
**simulate(self, num_complex, probeA, probeB, verbose=True)**
    Simulate PLA product count data

    Parameters
    ----------
    num_complex : list or numpy array
        An NA-by-NB array containing the number of complexes on each cell (NA and NB is the number of targets of probe A and B). Element [i,j] is the abundance of complex i:j.

    probeA : list or numpy array
        An NA-by-1 array containing the counts of non-interacting proteins bound by probe A. Element [i] is the abundance of non-complex forming protein i, bound by Prox-seq probe A.

    probeB : list or numpy array
        An NB-by-1 array containing the counts of non-interacting proteins
        bound by probe B. Entry [j] is the abundance of non-complex forming protein j, bound by Prox-seq probe B.

    verbose : bool, optional
        Whether to print out the simulation progress.
        Default is True.

### Analysis of PLA product count data
**Class name: plaObject**
Import PLA product count data. Rows are PLA products, columns are single cells.
Can be used to calculate protein abundance, expected count, and predict
protein complex count

  Parameters
  ----------
  data : pandas data frame
      Data frame of PLA count. Columns are cell barcodes, rows are PLA products.

  non_proximal_marker : string, option
      The name for marker of probeB_non_proximal PLA products. If a string, the marker will be used to extract probeB_non_proximal count of each probe. For example, if non_proximal_marker="oligo", then PLA product "CD3:oligo" is understood as probeB_non_proximal count of CD3 probe A, ie "CD3_A".
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
      Calculate protein count from PLA count data. Output is comparable to CITE-seq and REAP-seq.

  proteins : list
      List of detected proteins.

  pla_expected : pandas data frame
      Expected PLA count if no protein complexes exist.

  complex_count : pandas data frame
      Predicted protein complex count. May have a suffix depending on the setting of predictComplex().

  pla_probe_count : pandas data frame
      The count of each Prox-seq probe A and B, calculted from the detected PLA products.

  tol_ : numpy array
      Array of tolerance values for each iteration for predictComplex 'iterative' method. This is used as the convergence criterion.

  shape : tuple
      Shape of PLA product count matrix.

  sep : string
      Delimiter of PLA products. Example: for CD3:CD3, sep is ':'

  lr_params : pandas data frame
      Contains the parameters of the weighted least squares models, obtained from the linear regression (LR) method
      Columns: intercept , slope , SE_intercept , SE_slope ,
      intercept P value , slope P value , df

  complex_fisher : pandas data frame
      P-value of protein complex expression, calculated using one-sided Fisher's exact test

#### Attributes
**calculateProteinCount()**
Calculate protein count

**calculateExpected()**
Calculate the expected random count of all PLA products.

**calculateProbeCount()**
Calculate the counts of probes A and B of each target from the counts of PLA products.

**predictComplex(self, method='iterative', non_proximal_count=None, scale=1, intercept_cutoff=1, slope_cutoff=0, mean_cutoff=1, p_cutoff=0.05, non_interacting=None, p_adjust=True, sym_weight=0.25, df_guess=None, nIter=200, tol=1, suffix='')**
Predict complex count with two methods, 'iterative' or 'lr'.

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

**predictComplexFisher()**
Predict complex expression in single-cells using one-sided Fisher's exact test, and return a BH-corrected P-value.

### General functions
**randomPointGen2D(n)**
For generating random points on a sphere surface with unit radius (i.e., extracellular protein targets)

    Parameters
    ----------
    n : int
        Number of points

    Returns
    -------
    numpy array
        Return an n-by-3 array, where the columns are x, y and z coordinates

**def randomPointGen3D(n)**
For generating random points on a sphere surface with unit radius (i.e., intracellular protein targets)

    Parameters
    ----------
    n : int
        Number of points

    Returns
    -------
    numpy array
        Return an n-by-3 array, where the columns are x, y and z coordinates
