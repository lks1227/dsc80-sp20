
import numpy as np
import os

def data2array(filepath):
    """
    data2array takes in the filepath of a 
    data file like `restaurant.csv` in 
    data directory, and returns a 1d array
    of data.

    :Example:
    >>> fp = os.path.join('data', 'restaurant.csv')
    >>> arr = data2array(fp)
    >>> isinstance(arr, np.ndarray)
    True
    >>> arr.dtype == np.dtype('float64')
    True
    >>> arr.shape[0]
    100000
    """
    arr1 = open(filepath,"r")
    arr2 =arr1.readlines()[1:]
    arr3 = np.arange(len(arr2),dtype=np.float)
    for i in range(len(arr2)):
        arr3[i] = float(arr2[i].strip())
    return arr3


def ends_in_9(arr):
    """
    ends_in_9 takes in an array of dollar amounts 
    and returns the proprtion of values that end 
    in 9 in the hundredths place.

    :Example:
    >>> arr = np.array([23.04, 45.00, 0.50, 0.09])
    >>> out = ends_in_9(arr)
    >>> 0 <= out <= 1
    True
    """
    count = 0
    if(len(arr)==0):
        return 0.0
    for i in arr:
        if int(i*100)%10 == 9:
            count = count+1
    proportion = count/len(arr)
    return proportion
