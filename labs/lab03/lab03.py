
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    return [3,6]


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    return [2,5]


def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    return [2,4]


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 3


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def clean_apps(play):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    >>> cleaned.Reviews.dtype == int
    True
    '''
    def Round(flo):
        a = round(flo)
        return a
    play['Rating'] = play['Rating'].fillna(0)
    play['Rating'] = play['Rating'].apply(Round)
    def find(Str):
        letter = Str[-1]
        num = float(Str[:-1])
        if letter=='M':
            return num*1024
        else:
            return num
    play['Size'] = play['Size'].apply(find)
    def find_install(Str):
        Str = Str.strip('+')
        Str = Str.replace(',','')
        return int(Str)
    play['Installs']=play['Installs'].apply(find_install)
    play['Type'] = play['Type'].replace('Free',1)
    play['Type'] = play['Type'].replace('Paid',0)
    play['Price'] = pd.to_numeric(play['Price'].str.strip('$'))
    def find_year(Str):
        Year = Str[-4:]
        return int(Year)
    play['Last Updated']=play['Last Updated'].apply(find_year)
    return play


def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''
    play = cleaned.copy()
    year = play.groupby('Last Updated').count()['App']
    year = year[year>=100]
    year_list = year.index.tolist()
    rec = -float('inf')
    first_result = ''
    for i in year_list:
        med = play[play['Last Updated']==i]['Rating'].median()
        if rec<med:
            rec=med
            first_result = i
    rate = play.groupby('Content Rating').count().index
    rec = -float('inf')
    sec_result = ''
    for i in rate:
        med = play[play['Content Rating']==i]['Rating'].min()
        if rec<med:
            rec=med
            sec_result = i
    rate = play.groupby('Category').count().index
    rec = -float('inf')
    third_result = ''
    for i in rate:
        med = play[play['Category']==i]['Price'].mean()
        if rec<med:
            rec=med
            third_result = i
    rate = play.groupby('Category').count()['Reviews']
    rate = rate[rate>=1000].index
    rec = float('inf')
    forth_result = ''
    for i in rate:
        med = play[play['Category']==i]['Rating'].mean()
        if rec>med:
            rec=med
            forth_result = i
    return [first_result,sec_result,third_result,forth_result]

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> clean_play = clean_apps(play)
    >>> out = std_reviews_by_app_cat(clean_play)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    copy = cleaned.copy()[['Category', 'Reviews']]
    mean =  cleaned.groupby('Category').mean()['Reviews']
    stdev = cleaned.groupby('Category').std()['Reviews']
    means_list = np.array(list(map(lambda x: mean[x],copy['Category'].tolist())))
    stdev_list = np.array(list(map(lambda x: stdev[x],copy['Category'].tolist())))

    content = (copy['Reviews'].values-means_list)/stdev_list
    copy['Reviews'] = content
    return copy


def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    return ['EQUAL','BEAUTY']


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """

    first = []
    last = []
    company = []
    job = []
    email = []
    university = []
    for i in os.listdir(dirname):
        df = pd.read_csv(dirname + '/' + i)
        df.columns = np.array(list(map(lambda x: x.lower().replace('_',' '), df.columns.tolist())))
        columns = df.columns
        first += df['first name'].tolist()
        last += df['last name'].tolist()
        company +=df['current company'].tolist()
        job+=df['job title'].tolist()
        university += df['university'].tolist()
        email +=df['email'].tolist()
        result = pd.DataFrame({'first name':first,'last name':last,'current company':company,'job title':job,'email':email,'university':university})
    return result


def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a list containing the most common first name, job held, 
    university attended, and current company
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> all([isinstance(x, str) for x in out])
    True
    """

    
    copy = df[df['email']==df['email']].copy()
    copy['ids'] = copy['email'].apply(lambda x: x[-4:]=='.com')
    copy = copy[copy['ids']]
    first_name = copy['first name'].value_counts().index[0]
    job_held = copy['job title'].value_counts().index[0]
    university_attend = copy['university'].value_counts().index[0]
    current_company = copy['current company'].value_counts().index[0]
    return [first_name,job_held,university_attend,current_company]


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    out = pd.DataFrame(index = range(1,1001))
    for i in os.listdir(dirname):
        df = pd.read_csv(dirname + '/' + i)
        df = df.set_index('id')
        out = pd.concat([out,df],axis=1)
    return out

def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """
    column = df.columns
    checker = False
    for i in column:
        if i!= column[0]:
            if (df[i].count()/len(df[i]))>=0.9:
                check= True
    lis = (df.count(axis='columns')-1)/(len(column)-1)
    def change(a):
        if a<0.75:
            return 0
        else:
            return 5
    result = lis.apply(change)
    if checker:
        result+=1
    name = df[column[0]]
    frame = { 'name': name, 'extra credit': result } 
  
    table = pd.DataFrame(frame)
    return table

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------


def at_least_once(pets, procedure_history):
    """
    How many pets have procedure performed at this clinic at least once.

    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = at_least_once(pets, procedure_history)
    >>> out < len(pets)
    True
    """
    return len(np.unique(pd.merge(pets,procedure_history,on=['PetID'])['PetID']))


def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    merged = pd.merge(pets,owners,on='OwnerID')
    id_list = merged[['Name_x','Name_y','OwnerID']].groupby('OwnerID').count()
    two_or_more = id_list[id_list['Name_x']!=1].index
    pet_nameslist = []
    owner_name = []
    for i in two_or_more:
        pet_nameslist.append(merged[merged['OwnerID']==i]['Name_x'].tolist())
        owner_name.append(merged[merged['OwnerID']==i]['Name_y'].tolist()[0])
    ones = id_list[id_list['Name_x']==1].index
    pet_name = []
    owner_name_one = []
    for i in ones:
        pet_name.append(merged[merged['OwnerID']==i]['Name_x'].tolist()[0])
        owner_name_one.append(merged[merged['OwnerID']==i]['Name_y'].tolist()[0])
    petslist = pet_nameslist + pet_name
    ownersindex = owner_name+ owner_name_one
    return pd.Series(petslist,index=ownersindex)


def total_cost_per_owner(owners, pets, procedure_history, procedure_detail):
    """
    total cost per owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')

    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_owner(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['OwnerID'])
    True
    """
    merged = pd.merge(procedure_detail,procedure_history,on=['ProcedureType','ProcedureSubCode'])
    return pd.merge(merged,pets,on='PetID').groupby('OwnerID').sum()['Price']



# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!


GRADED_FUNCTIONS = {
    'q01': [
        'car_null_hypoth', 'car_alt_hypoth',
        'car_test_stat', 'car_p_value'
    ],
    'q02': ['clean_apps', 'store_info'],
    'q03': ['std_reviews_by_app_cat','su_and_spread'],
    'q04': ['read_survey', 'com_stats'],
    'q05': ['combine_surveys', 'check_credit'],
    'q06': ['at_least_once', 'pet_name_by_owner', 'total_cost_per_owner']
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
