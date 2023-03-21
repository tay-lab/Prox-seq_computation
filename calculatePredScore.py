"""

Author: Junjie Xia
Address: Tay Lab

"""
# Import packages
import numpy as np
import pandas as pd
import scipy.stats as stats

def calculatePredScore(true_complex,pred_complex,w1=0.5,w2=0.4,w3=0.1):
    """
    Function used to evaluate performance of methods of predicting real protein complexes from observed PLA count matrix
    
    Combined with Prox-seq simulation model #ProxseqClasses#
    
    Parameters
    -------------
    true_complex: pandas dataframe
    
    pred_complex: pandas dataframe
    
    w1: float
        weight assigned to sum of pearson correlation coefficient between true_complex and pred_complex
    w2: float
        weight assigned to sum of deviation between true_complex and pred_complex
    w3: float
        weight assigned to sum of false positive rate of pred_complex
    w1 + w2 + w3 = 1
    
    Returns
    -------------
    score: float
           score = w1*Pearson - w2*deviation - w3*FP_rate
    """
    pearsonr = pd.DataFrame(np.nan, index=true_complex.index, columns=['pearson'])
    for i in pearsonr.index:
        pearsonr.at[i,'pearson'] = stats.pearsonr(true_complex.loc[i,:], pred_complex.loc[i,:])[0]
    mean_deviation = pd.DataFrame({"True":(true_complex).mean(axis=1),
                                   "Prediction":(pred_complex).mean(axis=1)})
    scoredf = pd.concat([mean_deviation, pearsonr], axis=1)
    scoredf = scoredf[scoredf['True']>0]
    scoredf['deviation'] = abs(scoredf['Prediction'] - scoredf['True'])/scoredf['True']
    false_rate = pd.DataFrame({'true':(true_complex>0).sum(axis=1),
                               'prediction':(pred_complex>0).sum(axis=1)})
    false_rate = false_rate[false_rate['true']==0]
    
    score = w1*scoredf['pearson'].sum(axis=0) - w2*scoredf['deviation'].sum(axis=0) - w3*(false_rate['prediction']/dge.shape[1]).sum(axis=0)
    
    return score