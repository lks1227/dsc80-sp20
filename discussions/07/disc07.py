import requests
import time


def url_list():
    """
    A list of urls to scrape.

    :Example:
    >>> isinstance(url_list(), list)
    True
    >>> len(url_list()) > 1
    True
    """
    url_lis = []
    for i in range(26):
        url_lis.append('http://example.webscraping.com/places/default/index/'+str(i))
    return url_lis


def request_until_successful(url, N):
    """
    impute (i.e. fill-in) the missing values of each column 
    using the last digit of the value of column A.

    :Example:
    >>> resp = request_until_successful('http://quotes.toscrape.com', N=1)
    >>> resp.ok
    True
    >>> resp = request_until_successful('http://example.webscraping.com/', N=1)
    >>> isinstance(resp, requests.models.Response) or (resp is None)
    True
    """
    for i in range(N):
        request = requests.get(url)
        if request.ok ==True:
            return request
    return None