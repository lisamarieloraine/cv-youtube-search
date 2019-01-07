from Scripts.WebScraping.VerifyRequest import get_verified_response
from bs4 import BeautifulSoup
from re import compile
from Scripts.WebScraping.FilterTerms import SortBy, UploadDate


# @Description Filters hyper-links using regex pattern matching
# @argument <class 'list'>
# @return <class 'list'>
def filter_watch_only(_list):
    """Returns list of filtered youtube hyper-links"""
    result = list()
    pattern = compile('^(https?\:\/\/)?((www\.)?youtube\.com|youtu\.?be)\/watch.+$')
    for href in _list:
        if pattern.match(href):
            result.append(href)
    return result


# @Description Lists youtube response videos from given url
# @argument <class 'string'>
# @return <class 'list'>
def import_youtube_data(_youtube_url):
    """Returns list of urls from url response"""
    result = list()
    response = get_verified_response(_youtube_url)  # Get server response from url request
    soup = BeautifulSoup(response.data, 'html.parser')
    for vid in soup.findAll('a', attrs={'class': 'yt-uix-tile-link'}):  # Find all <a> tags on page
        result.append('https://www.youtube.com' + vid['href'])  # Extracting web links using 'href' property
    return filter_watch_only(result)


# @Description Concatenates Enum filter values to a valid string
# @argument <class 'Enum'>, ...
# @return <class 'string'>
def filter_string(_sortby_filter='', _uploadtime_filter=''):
    result = ''
    if _sortby_filter is not False:
        result = f'{_sortby_filter}{_uploadtime_filter}%253D&'
    return result


# @Description Higher order function to scrape and store data
# @argument <class 'string'> and <class 'string'>
# @return <class 'list'>
def search_and_store(_search_term, _filename, _sortby_filter='', _uploadtime_filter=''):
    """Returns list of search results while storing them in json"""
    print('Scraping Youtube data')
    # epoch_time = int(round(time.time() * 1000))
    formatted_search_term = _search_term.replace(" ", "+")  # Format search term with spaces
    filter_term = filter_string(_sortby_filter, _uploadtime_filter)  # Format filter term
    print(f'Using filter term: {filter_term}')
    youtube_url = f'https://www.youtube.com/results?{filter_term}search_query={formatted_search_term}'
    return import_youtube_data(youtube_url)


if __name__ == '__main__':
    _list = search_and_store('sark', 'unused', SortBy.ViewCount.value, UploadDate.LastHour.value)
    print('--- Search function ---')
    for link in _list:
        print(link)
    print(len(_list))

    # search_query = f'https://www.youtube.com/results?search_query=banana&pbj=1'
