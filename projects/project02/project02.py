import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_san(infp, outfp):
    """
    get_san takes in a filepath containing all flights and a filepath where
    filtered dataset #1 is written (that is, all flights arriving or departing
    from San Diego International Airport in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'santest.tmp')
    >>> get_san(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (53, 31)
    >>> os.remove(outfp)
    """

    return ...


def get_sw_jb(infp, outfp):
    """
    get_sw_jb takes in a filepath containing all flights and a filepath where
    filtered dataset #2 is written (that is, all flights flown by either
    JetBlue or Southwest Airline in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'jbswtest.tmp')
    >>> get_sw_jb(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (73, 31)
    >>> os.remove(outfp)
    """

    return ...


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def data_kinds():
    """
    data_kinds outputs a (hard-coded) dictionary of data kinds, keyed by column
    name, with values Q, O, N (for 'Quantitative', 'Ordinal', or 'Nominal').

    :Example:
    >>> out = data_kinds()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'O', 'N', 'Q'}
    True
    """

    return ...


def data_types():
    """
    data_types outputs a (hard-coded) dictionary of data types, keyed by column
    name, with values str, int, float.

    :Example:
    >>> out = data_types()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'int', 'str', 'float', 'bool'}
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------

def basic_stats(flights):
    """
    basic_stats takes flights and outputs a dataframe that contains statistics
    for flights arriving/departing for SAN.
    That is, the output should have have two rows, indexed by ARRIVING and
    DEPARTING, and have the following columns:

    * number of arriving/departing flights to/from SAN (count).
    * mean flight (arrival) delay of arriving/departing flights to/from SAN
      (mean_delay).
    * median flight (arrival) delay of arriving/departing flights to/from SAN
      (median_delay).
    * the airline code of the airline with the longest flight (arrival) delay
      among all flights arriving/departing to/from SAN (airline).
    * a list of the three months with the greatest number of arriving/departing
      flights to/from SAN, sorted from greatest to least (top_months).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = basic_stats(flights)
    >>> out.index.tolist() == ['ARRIVING', 'DEPARTING']
    True
    >>> cols = ['count', 'mean_delay', 'median_delay', 'airline', 'top_months']
    >>> out.columns.tolist() == cols
    True
    """
    result = pd.DataFrame(index = ['ARRIVING','DEPARTING'])
    From_SAN = len(flights[flights['ORIGIN_AIRPORT']=='SAN'])
    To_SAN = len(flights[flights['DESTINATION_AIRPORT']=='SAN'])
    result['count'] = [To_SAN,From_SAN]
    From_san_delay = flights[flights['ORIGIN_AIRPORT']=='SAN']['ARRIVAL_DELAY'].mean()
    To_san_delay = flights[flights['DESTINATION_AIRPORT']=='SAN']['ARRIVAL_DELAY'].mean()
    result['mean_delay']=[To_san_delay,From_san_delay]
    From_san_delays = flights[flights['ORIGIN_AIRPORT']=='SAN']['ARRIVAL_DELAY'].median()
    To_san_delays = flights[flights['DESTINATION_AIRPORT']=='SAN']['ARRIVAL_DELAY'].median()
    result['median_delay']=[To_san_delays,From_san_delays]
    From_max = flights[flights['ORIGIN_AIRPORT']=='SAN']['ARRIVAL_DELAY'].max()
    airline_from = flights[flights['ARRIVAL_DELAY']==From_max]['AIRLINE'].values[0]
    To_max = flights[flights['DESTINATION_AIRPORT']=='SAN']['ARRIVAL_DELAY'].max()
    airline_to = flights[flights['ARRIVAL_DELAY']==From_max]['AIRLINE'].values[0]
    result['airline']=[airline_to,airline_from]
    lis_to = list(flights[flights['DESTINATION_AIRPORT']=='SAN']['MONTH'].value_counts()[:3].index.values)
    lis_from = list(flights[flights['ORIGIN_AIRPORT']=='SAN']['MONTH'].value_counts()[:3].index.values)
    result['top_months']=[lis_to,lis_from]
    return result


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def depart_arrive_stats(flights):
    """
    depart_arrive_stats takes in a dataframe like flights and calculates the
    following quantities in a series (with the index in parentheses):
    - The proportion of flights from/to SAN that
      leave late, but arrive early or on-time (late1).
    - The proportion of flights from/to SAN that
      leaves early, or on-time, but arrives late (late2).
    - The proportion of flights from/to SAN that
      both left late and arrived late (late3).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats(flights)
    >>> out.index.tolist() == ['late1', 'late2', 'late3']
    True
    >>> isinstance(out, pd.Series)
    True
    >>> out.max() < 0.30
    True
    """
    temp1 = np.array(flights[flights['ORIGIN_AIRPORT']=='SAN'].index.to_list())
    temp2 = np.array(flights[flights['DESTINATION_AIRPORT']=='SAN'].index.to_list())
    indexs = np.union1d(temp1,temp2)
    flights_san = flights.loc[ indexs , : ]
    temp1 = np.array(flights[flights['DEPARTURE_DELAY']>0].index.to_list())
    temp2 = np.array(flights[flights['ARRIVAL_DELAY']<=0].index.to_list())
    indexs = np.intersect1d(temp1,temp2)
    flights_late1 = flights.loc[ indexs , : ]
    late1 = len(flights_late1)/len(flights_san)
    temp1 = np.array(flights[flights['DEPARTURE_DELAY']<=0].index.to_list())
    temp2 = np.array(flights[flights['ARRIVAL_DELAY']>0].index.to_list())
    indexs = np.intersect1d(temp1,temp2)
    flights_late2 = flights.loc[ indexs , : ]
    late2 = len(flights_late2)/len(flights_san)
    temp1 = np.array(flights[flights['DEPARTURE_DELAY']>0].index.to_list())
    temp2 = np.array(flights[flights['ARRIVAL_DELAY']>0].index.to_list())
    indexs = np.intersect1d(temp1,temp2)
    flights_late3 = flights.loc[ indexs , : ]
    late3 = len(flights_late3)/len(flights_san)
    result = pd.Series([late1,late2,late3],index = ['late1','late2','late3'] )
    return result


def depart_arrive_stats_by_month(flights):
    """
    depart_arrive_stats_by_month takes in a dataframe like flights and
    calculates the quantities in depart_arrive_stats, broken down by month

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats_by_month(flights)
    >>> out.columns.tolist() == ['late1', 'late2', 'late3']
    True
    >>> set(out.index) <= set(range(1, 13))
    True
    """
    flights = flights.fillna(0)
    flights['dep_late']=flights['DEPARTURE_DELAY']>0
    flights['arr_late']=flights['ARRIVAL_DELAY']>0
    month = flights['MONTH'].unique()
    lis1=flights['dep_late'].to_list()
    lis2=flights['arr_late'].to_list()
    flights['late1'] = pd.Series(np.array(lis1) & ~np.array(lis2))
    flights['late2']=pd.Series(~np.array(lis1) & np.array(lis2)) 
    flights['late3']=pd.Series(np.array(lis2) & np.array(lis1) )
    data = flights.groupby(['MONTH','late1'],as_index=False).count()
    total = flights.groupby(['MONTH']).count()['YEAR']
    la1 = data[data['late1']==True]['YEAR'].to_list()
    lat1 = pd.Series(la1,index=month)
    late1 = lat1/total
    data = flights.groupby(['MONTH','late2'],as_index=False).count()
    la2 = data[data['late2']==True]['YEAR'].to_list()
    lat2 = pd.Series(la2,index=month)
    late2 = lat2/total
    data = flights.groupby(['MONTH','late3'],as_index=False).count()
    la3 = data[data['late3']==True]['YEAR'].to_list()
    lat3 = pd.Series(la3,index=month)
    late3 = lat3/total
    frame = { 'late1': late1, 'late2': late2,'late3':late3 } 
  
    result = pd.DataFrame(frame)
    return result



# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def cnts_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, how many flights were there (in 2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = cnts_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    >>> (out >= 0).all().all()
    True
    """
    table = flights.groupby(['DAY_OF_WEEK','AIRLINE'],as_index=False)['YEAR'].count().pivot('DAY_OF_WEEK','AIRLINE', 'YEAR')
    return table


def mean_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, what is the average ARRIVAL_DELAY (in
    2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = mean_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    """

    return flights.groupby(['DAY_OF_WEEK','AIRLINE'],as_index=False)['ARRIVAL_DELAY'].mean().pivot('DAY_OF_WEEK','AIRLINE', 'ARRIVAL_DELAY')


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def predict_null_arrival_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the ARRIVAL_DELAY is null and otherwise False.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `ARRIVAL_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('ARRIVAL_DELAY', axis=1).apply(predict_null_arrival_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """
    ser = row['DIVERTED']+row['CANCELLED']
    result = ser >0
    return result


def predict_null_airline_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the AIRLINE_DELAY is null and otherwise False. Since the
    function doesn't depend on AIRLINE_DELAY, it should work a row even if that
    index is dropped.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `AIRLINE_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('AIRLINE_DELAY', axis=1).apply(predict_null_airline_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """
    ser = row['SECURITY_DELAY']
    result = (ser==ser)
    return result


# ---------------------------------------------------------------------
# Question #7
# ---------------------------------------------------------------------

def perm4missing(flights, col, N):
    """
    perm4missing takes in flights, a column col, and a number N and returns the
    p-value of the test (using N simulations) that determines if
    DEPARTURE_DELAY is MAR dependent on col.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = perm4missing(flights, 'AIRLINE', 100)
    >>> 0 <= out <= 1
    True
    """

    flight_mar = flights.copy()
    distr = (
        flight_mar
        .assign(is_null=flight_mar['DEPARTURE_DELAY'].isnull())
        .pivot_table(index='is_null', columns=col, aggfunc='size',fill_value = 0)
        .apply(lambda x:x / x.sum(), axis=1)
    )
    obs = distr.diff().iloc[-1].abs().sum() / 2
    n_repetitions = 500
    tvds = []
    for i in range(n_repetitions):
        shuffled_col = (
            flight_mar[col]
            .sample(replace=False, frac=1)
            .reset_index(drop=True)
        )
    
        shuffled = (
            flight_mar
            .assign(**{
                col: shuffled_col,
                'is_null': flight_mar['DEPARTURE_DELAY'].isnull()
            })
        )
    
        shuffled = (
            shuffled
            .pivot_table(index='is_null', columns=col, aggfunc='size',fill_value=0)
            .apply(lambda x:x / x.sum(), axis=1)
        )
        tvd = shuffled.diff().iloc[-1].abs().sum() / 2
        tvds.append(tvd)
    p_value = np.mean(tvds>obs)
    return p_value


def dependent_cols():
    """
    dependent_cols gives a list of columns on which DEPARTURE_DELAY is MAR
    dependent on.

    :Example:
    >>> out = dependent_cols()
    >>> isinstance(out, list)
    True
    >>> cols = 'YEAR DAY_OF_WEEK AIRLINE DIVERTED CANCELLATION_REASON'.split()
    >>> set(out) <= set(cols)
    True
    """

    return 'YEAR DAY_OF_WEEK AIRLINE DIVERTED'.split()


def missing_types():
    """
    missing_types returns a Series
    - indexed by the following columns of flights:
    CANCELLED, CANCELLATION_REASON, TAIL_NUMBER, ARRIVAL_TIME.
    - The values contain the most-likely missingness type of each column.
    - The unique values of this Series should be MD, MCAR, MAR, MNAR, NaN.

    :param:
    :returns: A series with index and values as described above.

    :Example:
    >>> out = missing_types()
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) - set(['MD', 'MCAR', 'MAR', 'NMAR', np.NaN]) == set()
    True
    """

    return ['MAR','MCAR','MAR','MD']


# ---------------------------------------------------------------------
# Question #8
# ---------------------------------------------------------------------

def prop_delayed_by_airline(jb_sw):
    """
    prop_delayed_by_airline takes in a dataframe like jb_sw and returns a
    DataFrame indexed by airline that contains the proportion of each airline's
    flights that are delayed.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> (out >= 0).all().all() and (out <= 1).all().all()
    True
    >>> len(out.columns) == 1
    True
    """

    return ...


def prop_delayed_by_airline_airport(jb_sw):
    """
    prop_delayed_by_airline_airport that takes in a dataframe like jb_sw and
    returns a DataFrame, with columns given by airports, indexed by airline,
    that contains the proportion of each airline's flights that are delayed at
    each airport.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline_airport(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> ((out >= 0) | (out <= 1) | (out.isnull())).all().all()
    True
    >>> len(out.columns) == 6
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question #9
# ---------------------------------------------------------------------

def verify_simpson(df, group1, group2, occur):
    """
    verify_simpson verifies whether a dataset displays Simpson's Paradox.

    :param df: a dataframe
    :param group1: the first group being aggregated
    :param group2: the second group being aggregated
    :param occur: a column of df with values {0,1}, denoting
    if an event occurred.
    :returns: a boolean. True if simpson's paradox is present,
    otherwise False.

    :Example:
    >>> df = pd.DataFrame([[4,2,1], [1,2,0], [1,4,0], [4,4,1]], columns=[1,2,3])
    >>> verify_simpson(df, 1, 2, 3) in [True, False]
    True
    >>> verify_simpson(df, 1, 2, 3)
    False
    """

    return ...


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def search_simpsons(jb_sw, N):
    """
    search_simpsons takes in the jb_sw dataset and a number N, and returns a
    list of N airports for which the proportion of flight delays between
    JetBlue and Southwest satisfies Simpson's Paradox.

    Only consider airports that have '3 letter codes',
    Only consider airports that have at least one JetBlue and Southwest flight.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=1000)
    >>> pair = search_simpsons(jb_sw, 2)
    >>> len(pair) == 2
    True
    """

    return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_san', 'get_sw_jb'],
    'q02': ['data_kinds', 'data_types'],
    'q03': ['basic_stats'],
    'q04': ['depart_arrive_stats', 'depart_arrive_stats_by_month'],
    'q05': ['cnts_by_airline_dow', 'mean_by_airline_dow'],
    'q06': ['predict_null_arrival_delay', 'predict_null_airline_delay'],
    'q07': ['perm4missing', 'dependent_cols', 'missing_types'],
    'q08': ['prop_delayed_by_airline', 'prop_delayed_by_airline_airport'],
    'q09': ['verify_simpson'],
    'q10': ['search_simpsons']
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
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
