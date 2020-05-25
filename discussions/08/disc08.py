from glob import glob
import pandas as pd
import numpy as np

from scipy.stats import linregress


def rmse(datasets):
    """
    Return the RMSE of each of the datasets.

    >>> datasets = {k:pd.read_csv('data/dataset_%d.csv' % k) for k in range(7)}    
    >>> out = rmse(datasets)
    >>> len(out) == 7
    True
    >>> isinstance(out, pd.Series)
    True
    """
    rmse_lis = []
    index_lis = []
    for k,v in datasets.items():
        r = linregress(v)
        def predict(a,r):
            return r.slope*a+r.intercept
        rmse = np.sqrt(np.mean((predict(v.X,r) - v.Y)**2))
        rmse_lis.append(rmse)
        index_lis.append(k)
    result = pd.Series(rmse_lis,index = index_lis)
    return result


def heteroskedasticity(datasets):
    """
    Return a boolean series giving whether a dataset is
    likely heteroskedastic.

    >>> datasets = {k:pd.read_csv('data/dataset_%d.csv' % k) for k in range(7)}    
    >>> out = heteroskedasticity(datasets)
    >>> len(out) == 7
    True
    >>> isinstance(out, pd.Series)
    True
    """
    result = []
    index = []
    for k, v in datasets.items():
        lm = linregress(v['X'], v['Y'])
        def predict(a,r):
            return r.slope*a+r.intercept
        pre_ser = predict(v['X'],lm)-v['Y']
        pre_ser_sq = pre_ser*pre_ser
        linear= linregress(v['X'],pre_ser_sq)
        print()
        result.append(linear.pvalue)
        index.append(k)
    correlation = pd.Series(result,index = index)
    heteroskedasticity = correlation<0.05
    return heteroskedasticity
