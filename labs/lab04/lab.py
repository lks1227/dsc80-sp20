
import os

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def latest_login(login):
    """Calculates the latest login time for each user
    :param login: a dataframe with login information
    :return: a dataframe with latest login time for
    each user indexed by "Login Id"
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> result = latest_login(login)
    >>> len(result)
    433
    >>> result.loc[381, "Time"].hour > 12
    True
    """

    login['Time'] = login['Time'].apply(lambda x:pd.Timestamp(x))
    result = login.groupby('Login Id').max()
    return result

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------


def smallest_ellapsed(login):
    """
    Calculates the the smallest time elapsed for each user.
    :param login: a dataframe with login information but without unique IDs
    :return: a dataframe, indexed by Login ID, containing
    the smallest time elapsed for each user.
    >>> fp = os.path.join('data', 'login_table.csv')
    >>> login = pd.read_csv(fp)
    >>> result = smallest_ellapsed(login)
    >>> len(result)
    238
    >>> 18 < result.loc[1233, "Time"].days < 23
    True
    """

    login['Time'] = login['Time'].apply(lambda x:pd.Timestamp(x))
    count = login['Login Id'].value_counts()
    count_i = count[count != 1].index.tolist()
    res = login[login['Login Id'].isin(count_i)]
    df = res.groupby('Login Id').apply(lambda df: np.min(df.diff())).drop('Login Id',axis=1)
    return df


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def total_seller(df):
    """
    Total for each seller
    :param df: like sales
    :return: pivot table
    >>> fp = os.path.join('data', 'sales.csv')
    >>> df = pd.read_csv(fp)
    >>> out = total_seller(df)
    >>> out.index.dtype
    dtype('O')
    >>> out["Total"].sum() < 15000
    True

    """

    result = pd.pivot_table(df,index = 'Name',values = 'Total',aggfunc='sum')
    return result


def product_name(df):
    """
    :param df: like sales
    :return: pivot table
    >>> fp = os.path.join('data', 'sales.csv')
    >>> df = pd.read_csv(fp)
    >>> out = product_name(df)
    >>> out.size
    15
    >>> out.loc["pen"].isnull().sum()
    0
    """

    result = pd.pivot_table(df,index = 'Product',columns = 'Name',aggfunc='sum')
    return result

def count_product(df):
    """
    :param df: like sales
    :return: pivot table
    >>> fp = os.path.join('data', 'sales.csv')
    >>> df = pd.read_csv(fp)
    >>> out = count_product(df)
    >>> out.loc["boat"].loc["Trump"].value_counts()[0]
    6
    >>> out.size
    70
    """

    result = pd.pivot_table(df,columns='Date',index = ['Product','Name'],aggfunc='sum').fillna(0)
    return result.applymap(lambda x:int(x))


def total_by_month(df):
    """
    :param df: like sales
    :return: pivot table
    >>> fp = os.path.join('data', 'sales.csv')
    >>> df = pd.read_csv(fp)
    >>> out = total_by_month(df)
    >>> out["Total"]["May"].idxmax()
    ('Smith', 'book')
    >>> out.shape[1]
    5
    """

    def get_month(date):
        month_list =['January','February','March','April','May','June','July',
                     'August','September','October','November','December']
        return month_list[int(date[:2])-1]
    df['Month'] = df['Date'].apply(get_month)

    return pd.pivot_table(df,columns='Month',index = ['Name','Product'],aggfunc='sum').fillna(0).applymap(lambda x:int(x))

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    """
    diff_of_means takes in a dataframe of counts
    of skittles (like skittles) and their origin
    and returns the absolute difference of means
    between the number of oranges per bag from Yorkville and Waco.

    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> out = diff_of_means(skittles)
    >>> 0 <= out
    True
    """

    return abs(np.mean(data[data['Factory']=='Waco'][col].values)-np.mean(data[data['Factory']=='Yorkville'][col].values))


def simulate_null(data, col='orange'):
    """
    simulate_null takes in a dataframe of counts of
    skittles (like skittles) and their origin, and
    generates one instance of the test-statistic
    under the null hypothesis

    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> out = simulate_null(skittles)
    >>> isinstance(out, float)
    True
    >>> 0 <= out <= 1.0
    True
    """

    table = data[['Factory',col]].copy()
    table['Factory'] = table['Factory'].sample(len(table)).values
    return diff_of_means(table,col)


def pval_orange(data, col='orange'):
    """
    pval_orange takes in a dataframe of counts of
    skittles (like skittles) and their origin, and
    calculates the p-value for the permutation test
    using 1000 trials.

    :Example:
    >>> skittles_fp = os.path.join('data', 'skittles.tsv')
    >>> skittles = pd.read_csv(skittles_fp, sep='\\t')
    >>> pval = pval_orange(skittles)
    >>> isinstance(pval, float)
    True
    >>> 0 <= pval <= 0.1
    True
    """

    lst = []
    for i in range(1000):
        lst.append(simulate_null(data,col))
    standard = diff_of_means(data,col)
    t= np.array(lst)>=standard
    count = np.count_nonzero(t)/len(t)
    return count


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def ordered_colors():
    """
    ordered_colors returns your answer as an ordered
    list from "most different" to "least different"
    between the two locations. You list should be a
    hard-coded list, where each element has the
    form (color, p-value).

    :Example:
    >>> out = ordered_colors()
    >>> len(out) == 5
    True
    >>> colors = {'green', 'orange', 'purple', 'red', 'yellow'}
    >>> set([x[0] for x in out]) == colors
    True
    >>> all([isinstance(x[1], float) for x in out])
    True
    """

    return [('yellow', 0.0),('orange', 0.037),('red', 0.244),('green', 0.479),('purple', 0.99)]


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def same_color_distribution():
    """
    same_color_distribution outputs a hard-coded tuple
    with the p-value and whether you 'Fail to Reject' or 'Reject'
    the null hypothesis.

    >>> out = same_color_distribution()
    >>> isinstance(out[0], float)
    True
    >>> out[1] in ['Fail to Reject', 'Reject']
    True
    """

    return (0.009,'Reject')

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def perm_vs_hyp():
    """
    Multiple choice response for question 8

    >>> out = perm_vs_hyp()
    >>> ans = ['P', 'H']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """

    return ['P','P','H','H','P']


# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def after_purchase():
    """
    Multiple choice response for question 8

    >>> out = after_purchase()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NI']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    """

    return ['MCAR','MD','MAR','MCAR','MAR']

# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def multiple_choice():
    """
    Multiple choice response for question 9

    >>> out = multiple_choice()
    >>> ans = ['MD', 'MCAR', 'MAR', 'NI']
    >>> len(out) == 5
    True
    >>> set(out) <= set(ans)
    True
    >>> out[1] in ans
    True
    """

    return ['MAR', 'MD', 'NI', 'MAR', 'MCAR']

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['latest_login'],
    'q02': ['smallest_ellapsed'],
    'q03': ['total_seller', 'product_name', 'count_product', 'total_by_month'],
    'q04': ['diff_of_means', 'simulate_null', 'pval_orange'],
    'q05': ['ordered_colors'],
    'q06': ['same_color_distribution'],
    'q07': ['perm_vs_hyp'],
    'q08': ['after_purchase'],
    'q09': ['multiple_choice']
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
