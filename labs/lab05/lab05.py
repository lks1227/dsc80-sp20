import os
import pandas as pd
import numpy as np
import datetime
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def first_round():
    """
    :return: list with two values
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    """
    return [0.164,'NR']


def second_round():
    """
    :return: list with three values
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    >>> out[2] is "ND" or out[2] is "D"
    True
    """
    return [0.008,'R','D']



# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def verify_child(heights):
    """
    Returns a series of p-values assessing the missingness
    of child-height columns on father height.

    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    def per(heights,col):
        check = 'father'
        n_repetitions = 100
        copy = heights.copy()
        kslist = []
        observed_ks, _ = ks_2samp(
            copy.loc[heights[col].isnull(), check],
            copy.loc[~heights[col].isnull(), check]
        )
        for _ in range(n_repetitions):

            # shuffle the ages
            shuffled_table = (
                copy[check]
                .sample(replace=False, frac=1)
                .reset_index(drop=True)
            )

            # 
            shuffled = (
                copy
                .assign(**{'Shuffled': shuffled_table})
            )
            ks, _ = ks_2samp(
                shuffled.loc[shuffled[col].isnull(), 'Shuffled'],
                shuffled.loc[~shuffled[col].isnull(), 'Shuffled']
            )

            # add it to the list of results
            kslist.append(ks)
        arr = np.array(kslist)
        return np.count_nonzero(arr >= observed_ks) / len(kslist)
    lis =[per(heights,'child_95'),per(heights,'child_90'),per(heights,'child_75'),per(heights,'child_50'),per(heights,'child_25'),per(heights,'child_10'),per(heights,'child_5')]
    return pd.Series(lis,index=['child_95','child_90','child_75','child_50','child_25','child_10','child_5'])


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.

    :Example:
    >>> set(missing_data_amounts()) <= set(range(1,6))
    True
    """

    return [2,3]


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a dataframe with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> df['child'] = df['child_50']
    >>> out = cond_single_imputation(df)
    >>> out.isnull().sum() == 0
    True
    >>> (df.child.std() - out.std()) > 0.5
    True
    """
    ser = new_heights.father.quantile([0.25,0.5,0.75])
    first_quar = new_heights[new_heights['father']<ser.iloc[0]].child.mean()
    def helper(ser,Min,Max):
        return ser.apply(check, args=(Min,Max,))
    def check(num,Min,Max):
        if (num>Min) and (num<Max):
            return True
        else:
            return False
    
    second_quar = new_heights[helper(new_heights['father'],ser.iloc[0],ser.iloc[1])].child.mean()
    third_quar = new_heights[helper(new_heights['father'],ser.iloc[1],ser.iloc[2])].child.mean()
    forth_quar = new_heights[new_heights['father']>ser.iloc[2]].child.mean()
    def helper(row):
        if row[1]!=row[1]:
            if row[0]<ser.iloc[0]:
                return first_quar
            elif (row[0]<ser.iloc[1]):
                return second_quar
            elif (row[0]<ser.iloc[2]):
                return third_quar
            else:
                return forth_quar
        else:
            return row[1]
    df = new_heights[new_heights['child'].isnull()]
    df.iloc[0]['child']=1
    return new_heights.apply(helper,axis=1)

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    """

    child = child.dropna()
    count, division = np.histogram(child,bins = 10)
    density = count/count.sum()
    result = []
    for _ in range(N):
        choice = np.random.choice(range(10),p=density)
        rang = [division[choice],division[choice+1]]
        num = np.random.uniform(division[choice],division[choice+1])
        result.append(num)
    return np.array(result)

def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isnull().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    """

    def helper(ele):
        if ele!=ele:
            return quantitative_distribution(child, 1)[0]
        else:
            return ele
    return child.apply(helper)


# ---------------------------------------------------------------------
# Question # X
# ---------------------------------------------------------------------

def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> list1, list2 = answers()
    >>> len(list1)
    4
    >>> len(list2)
    6
    """
    return [[1,2,2,1],['https://world.taobao.com/','https://moz.com/','https://twitter.com/',
                       '*https://www.gradescope.com','*https://www.baidu.com','*https://www.bilibili.com/']]




# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['first_round', 'second_round'],
    'q02': ['verify_child', 'missing_data_amounts'],
    'q03': ['cond_single_imputation'],
    'q04': ['quantitative_distribution', 'impute_height_quant'],
    'q05': ['answers']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
