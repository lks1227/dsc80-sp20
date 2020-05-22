import os
import pandas as pd
import numpy as np
import requests
import bs4
import json


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.

    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!

    >>> os.path.exists('lab06_1.html')
    True
    """

    # Don't change this function body!
    # No python required; create the HTML file.
 
    return


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    soup = bs4.BeautifulSoup(text,features="lxml")
    result = []
    for i in soup.find('ol',attrs={'class':'row'}).find_all('li'):
        if (i.find('p',attrs = {'class':'star-rating'}).get('class')[1]=='Five')|(i.find('p',attrs = {'class':'star-rating'}).get('class')[1]=='Four'):
            st = i.find('p',attrs = {'class':'price_color'}).text
            ind = [x.isdigit() for x in st].index(True)
            num = float(st[ind:])
            if num<50:
                result.append(i.find('a').get('href'))
    return result


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    result = []
    result_dict={}
    soup = bs4.BeautifulSoup(text,features="lxml")
    def dele(st):
        index =0
        for i in range(len(st)):
            if st[i] ==' ':
                index+=1
            else:
                return st[index:]
        return st[index:]
    for i in soup.find_all('a'):
        att = i.attrs
        st = att.get('href')
        if 'category' in st:
            result.append(soup.find('a',attrs = att).text)
    for i in result:
        if i in categories:
            lst1=[]
            lst2=[]
            for k in soup.find_all('table'):
                for j in k.find_all('th'):
                    lst1.append(j.text)
                for x in k.find_all('td'):
                    lst2.append(x.text)
                if lst1[0] in result_dict.keys():
                    result_dict.update({lst1[ind]: lst2[ind] for ind in range(len(lst1))})
                else:
                    result_dict = {lst1[ind]: lst2[ind] for ind in range(len(lst1))} 
            result_dict.update({'Category':i})
            title=soup.find('title').text
            title = title.replace(' | Books to Scrape - Sandbox','')
            title = title.replace('\n','')
            title = dele(title)
            result_dict.update({'Title':title})
            rates = soup.find('p',attrs={'class':'star-rating'}).get('class')[1]
            result_dict.update({'Rating':rates})
            texts = soup.find(attrs = {'name':'description'}).get('content').replace('\n','')
            result_dict.update({'description':texts})
    if not result_dict:
        return None
    else:
        return result_dict


def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).

    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """    
    def helper(d1,d2):
        d3 = {}
        for (k, v) in list(d1.items())+list(d2.items()):
            try:
                d3[k] += [v]
            except KeyError:
                d3[k] = [v]
        return d3
    url = "http://books.toscrape.com/"
    d1 = dict()
    for i in range(k):
        r = requests.get(url)   
        urlText = r.text
        soup = bs4.BeautifulSoup(urlText, 'html.parser')
        lst = extract_book_links(urlText)
        for j in lst:
            r = requests.get(url+j)
            urlText = r.text
            d2 = get_product_info(urlText, categories)
            if d2!= None:
                d1 = helper(d1,d2)
        url = "http://books.toscrape.com/"+list(soup.find(attrs = {'class':'next'}).children)[0].get('href')
    return pd.DataFrame.from_dict(d1)


# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a dataframe

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[0]
    'June 03, 19'
    """
    stock_endpoint = 'https://financialmodelingprep.com/api/v3/historical-price-full/{}'
    new_endpoint = stock_endpoint.replace('{}',ticker)
    temp = requests.get(new_endpoint)
    dic = json.loads(temp.content).get('historical')
    result = pd.DataFrame()
    for i in dic:
        if (int(i['date'][0:4])==year)&(int(i['date'][5:7])==month):
            df = pd.DataFrame(i,index=[0])
            result = pd.concat([result,df],ignore_index=True)
    return result.reset_index(drop=True)


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billion dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    """
    percentage = '+'+str(np.round(((history.close.iloc[-1]-history.open.iloc[0])/(history.open.iloc[0]))*100,2))+'%'
    result = str(np.round((history.volume*((history.high+history.low)/2)/10**9).sum(),2))+'B'
    return (percentage,result)


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def get_comments(storyid):
    """
    Returns a dataframe of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    news_endpoint = "https://hacker-news.firebaseio.com/v0/item/{}.json"
    ids = 18344932
    parents_points = news_endpoint.replace('{}',str(ids))
    response = requests.get(parents_points)
    parents = response.json()
    stacks = []
    parents
    def dfs(elt, visited):
        if elt not in visited:
            if 'dead' in elt.keys():
                if elt['dead']==True:
                    pass
                else:
                    visited.append(elt['id'])
            else:
                visited.append(elt['id'])
            try:
                for e in elt['kids']:
                    parents_points = news_endpoint.replace('{}',str(e))
                    response = requests.get(parents_points)
                    parents = response.json()
                    dfs(parents, visited)
            except KeyError:
                pass
        return visited
    lst = dfs(parents,[])[1:]
    lis1 = ['id','by','parent','text','time']
    result = pd.DataFrame(columns = lis1)
    for i in lst:
        parents_points = news_endpoint.replace('{}',str(i))
        response = requests.get(parents_points)
        parents = response.json()
        lis2 = [parents[i] for i in lis1]
        dic = {lis1[i]:lis2[i] for i in range(len(lis1))}
        temp = pd.DataFrame(dic,index=[0])
        result = pd.concat([result,temp],ignore_index=True)
    result['time']=pd.to_datetime(result['time'], unit='s')
    return result


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['question1'],
    'q02': ['extract_book_links', 'get_product_info', 'scrape_books'],
    'q03': ['stock_history', 'stock_stats'],
    'q04': ['get_comments']
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
