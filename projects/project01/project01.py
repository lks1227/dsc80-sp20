
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    header = grades.columns
    Dict = dict()
    result1 =  np.where(header.str.contains('lab') == True)
    result2 = np.where(header.str.contains('-')==False)
    result_lab = np.intersect1d(result1,result2)
    result1 =  np.where(header.str.contains('project') == True)
    result3 = np.where(header.str.contains('_') == False)
    result_p = np.intersect1d(result1,result2)
    result_p = np.intersect1d(result_p,result3)
    result1 =  np.where(header.str.contains('checkpoint') == True)
    result_checkpoint = np.intersect1d(result1,result2)
    result1 =  np.where(header.str.contains('disc') == True)
    result_disc = np.intersect1d(result1,result2)
    Dict['lab'] = list(header[result_lab])
    Dict['project'] = list(header[result_p])
    Dict['midterm'] = ['Midterm']
    Dict['final'] = ['Final']
    Dict['checkpoint']=list(header[result_checkpoint])
    Dict['disc']=list(header[result_disc])
    return Dict


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    header = grades.columns
    dic = get_assignment_names(grades)
    lis = dic['project']
    total = pd.Series(0,index = range(len(grades[lis[0]])))
    for i in lis:
        st = i+'_free_response'
        s = i+' - Max Points'
        fs = st+' - Max Points'
        if (st in header):
            g = grades[st].add(grades[i],fill_value=0)
            t = grades[s].add(grades[fs])
        else:
            g = grades[i]
            t = grades[s]
        proportion = g.divide(t,fill_value=0)
        total = proportion/len(lis)+total
    return total


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    header = grades.columns
    result1 =  np.where(header.str.contains('lab') == True)
    result2 = np.where(header.str.contains('Lateness')==True)
    result_p = np.intersect1d(result1,result2)
    df = grades[header[result_p]]
    l = []
    for i in header[result_p]:
        a = df[i].apply(cal)
        b = (a==True).sum()
        l.append(b)
    ser = pd.Series(l, index =get_assignment_names(grades)['lab'])
    return ser

def cal(st):
    """
    helper method
    """
    lst = st.split(':')
    i = int(lst[0])*3600+int(lst[1])*60+int(lst[2])
    if (i <= 36000 and i >0):
        return True
    else:
        return False

# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.8, 0.5}
    True
    """
        
    return col.apply(calculate)

def calculate(st):
    lst = st.split(':')
    i = int(lst[0])*3600+int(lst[1])*60+int(lst[2])
    if (i <= 604800 and i >36000):
        return 0.9
    elif(i>604800 and i <=1209600):
        return 0.8
    elif(i >1209600):
        return 0.5
    elif(i<=36000):
        return 1.0

# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    labs = get_assignment_names(grades).get('lab')
    result = pd.DataFrame()
    for col in labs:
        col_num = np.where(grades.columns.str.contains(col))[0]
        col_name = grades.columns[col_num]
       
        column =lateness_penalty(grades[col_name[2]])
        cols = grades[col_name[0]].multiply(column, fill_value=0)
        result[col] = cols.divide(grades[col_name[1]], fill_value=0)

    return result.fillna(0)


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    s = (processed.sum(axis=1)-processed.min(axis=1))/(len(processed.columns)-1)
    return round(s[0],2)


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    lab_totals = lab_total(process_labs(grades))
    proj_totals = projects_total(grades)
    mid = get_assignment_names(grades)['midterm']
    mid_total = 0
    for i in mid:
        j = i+' - Max Points'
        g = grades[i]
        t = grades[j]
        proportion = g.divide(t,fill_value=0)
        mid_total = proportion/len(mid)+mid_total
    final = get_assignment_names(grades)['final']
    fin_total = 0
    for i in final:
        j = i+' - Max Points'
        g = grades[i]
        t = grades[j]
        proportion = g.divide(t,fill_value=0)
        fin_total = proportion/len(final)+fin_total
    check = get_assignment_names(grades)['checkpoint']
    check_total = 0
    for i in check:
        j = i+' - Max Points'
        g = grades[i]
        t = grades[j]
        proportion = g.divide(t,fill_value=0)
        check_total = proportion/len(check)+check_total
    disc = get_assignment_names(grades)['disc']
    disc_total = 0
    for i in disc:
        j = i+' - Max Points'
        g = grades[i]
        t = grades[j]
        proportion = g.divide(t,fill_value=0)
        disc_total = proportion/len(disc)+disc_total
    result=lab_totals*0.2+proj_totals*0.3+mid_total*0.15+fin_total*0.3+check_total*0.025+disc_total*0.025
    return result


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    def lettergrade_scale(scores):
        if 0.90<= scores <= 1.00:
            return 'A'
        elif 0.80<= scores <0.90:
            return 'B'
        elif 0.70<= scores <0.80:
            return 'C'
        elif 0.60<=scores <0.70:
            return 'D'
        else: 
            return'F'

    return total.apply(lettergrade_scale)


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    abcdf_grade = final_grades(total_points(grades))
    result = abcdf_grade.value_counts()/len(grades)
    result = result.round(decimals=5)
    return result

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of sophomores
    was no better on average than the class
    as a whole (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """
    grades['totals'] = total_points(grades)
    
    So = grades[grades['Level'] == 'SO']
    So_mean = So['totals'].mean()
    
    random_sample = pd.DataFrame(np.random.choice(
        list(grades["totals"]), size=(100, len(So))))
    
    t_value = random_sample.mean(1)
    result = len(t_value[t_value > So_mean]) / len(t_value)
    grades.drop(columns=['totals'])
    return result


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    dic = get_assignment_names(grades)
    num_cols = 0
    num_rows = len(grades.index)
    for x in dic:
        num_cols += len(dic[x])
    ran = np.random.normal(0, 0.02, size=(num_rows, num_cols))
    df1 = pd.DataFrame()
    labs = process_labs(grades)
    df1 = pd.concat([df1,labs], axis=1)
    for x in dic:
        if x!= 'lab':
            df1 = pd.concat([df1,grades[dic[x]]], axis=1)
    header = grades.columns
    lis = dic['project']
    for i in lis:
        st = i+'_free_response'
        s = i+' - Max Points'
        fs = st+' - Max Points'
        if (st in header):
            g = grades[st].add(grades[i],fill_value=0)
            t = grades[s].add(grades[fs])
            df1[i] = g.divide(t,fill_value=0)
        else:
            g = grades[i]
            t = grades[s]
            df1[i] = g.divide(t,fill_value=0)
    df2 = pd.DataFrame(ran,columns = df1.columns)
    for i in df1.columns:
        if 'lab' in i or 'project' in i:
            pass
        else:
            j = i+' - Max Points'
            df1[i] = df1[i]/grades[j]
    df1 = df1.fillna(0)
    df1 = np.clip(df1.add(df2),0,1)
    df1 = df1.fillna(0)
    labs_total = df1[dic['lab']]
    labs = (labs_total.sum(axis = 1)-labs_total.min(axis=1))/(len(dic['lab'])-1)
    project_total = df1[dic['project']]
    projects = project_total.sum(axis = 1)/len(dic['project'])
    check_total = df1[dic['checkpoint']]
    check = check_total.sum(axis = 1)/len(dic['checkpoint'])
    di_total = df1[dic['disc']]
    dis = di_total.sum(axis = 1)/len(dic['disc'])
    mt = df1[dic['midterm']]
    mts = mt.sum(axis = 1)/len(dic['midterm'])
    fin = df1[dic['final']]
    final = fin.sum(axis = 1)/len(dic['final'])
    result = labs*0.2+projects*0.3+check*0.025+dis*0.025+mts*0.15+final*0.3
    return result


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4], bool)
    True
    """

    return [-0.0033565172041279496,74.57943925233644,[69.53271028037383, 75.51401869158878],0.08971962616822426,True]

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
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
