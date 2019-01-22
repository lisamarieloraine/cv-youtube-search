from Scripts.WebScraping.VerifyRequest import get_verified_response
from Scripts.WebScraping.FilterTerms import SortBy, UploadDate, Features, Duration
from Scripts.WebScraping.PullThumbnail import get_thumbnail
from bs4 import BeautifulSoup
from re import compile


# @Description Filters hyper-links using regex pattern matching
# @argument <class 'list'>
# @return <class 'list'>
def filter_watch_only(_list):
    """Returns list of filtered youtube hyper-links"""
    result = list()
    pattern = compile('^(https?\:\/\/)?((www\.)?youtube\.com|youtu\.?be)\/watch.+$')
    for href in _list:
        if href == 'https://www.youtube.com/watch?v=4wb1RfnnqdY':  # Hard code remove naked cooking
            continue
        if type(href) is str and pattern.match(href):
            result.append(href)
    return result


# @Description Filters hyper-links if they do not have thumbnails
# @argument <class 'list'>
# @return <class 'list'>
def filter_thumbnail_only(_list):
    """Returns list of filtered youtube hyper-links"""
    result = list()
    for count, href in enumerate(_list):
        if count > 15:
            break
        if get_verified_response(get_thumbnail(href)).status == 200:
            result.append(href)
    return result


# @Description Lists youtube response videos from given url
# @argument <class 'string'>
# @return <class 'list'>
def import_youtube_data(_youtube_url):
    """Returns list of urls from url response"""
    result = list()

    for i in range(3):
        response = get_verified_response(_youtube_url + '&page={}'.format(i))  # Get server response from url request
        if response is None:
            continue
        # print(response.data)
        soup = BeautifulSoup(response.data, 'html.parser')
        for vid in soup.findAll('a', attrs={'class': 'yt-uix-tile-link'}):  # Find all <a> tags on page
            result.append('https://www.youtube.com' + vid['href'])  # Extracting web links using 'href' property

    print('Length before filter: {}'.format(len(result)))
    result = filter_watch_only(result)
    result = filter_thumbnail_only(result)
    print('Length after filter: {}'.format(len(result)))
    return result


# @Description Concatenates Enum filter values to a valid string
# @argument <class 'Enum'>, ...
# @return <class 'string'>
def filter_search_string(_search_term, _filter=list()):
    result = '{} +recipe'.format(_search_term)  # tutorial
    for term in _filter:
        if term != '':
            result += ', {}'.format(term)
    return result


# @Description Higher order function to scrape and store data
# @argument <class 'string'> and <class 'string'>
# @return <class 'list'>
def search_and_store(_search_term, _filename, _sortby_filter='', _uploadtime_filter='', _duration_filter='', _feature_filter=list()):
    """Returns list of search results while storing them in json"""
    print('Scraping Youtube data')

    filter_term = filter_search_string(_search_term, [_uploadtime_filter, _duration_filter] + _feature_filter)  # Format filter terms
    formatted_search_term = filter_term.replace("+", "%2B").replace(",", "%2C").replace(" ", "+")  # Format search term with pluses, commas, spaces
    print('Using filter search term: {}'.format(formatted_search_term))

    youtube_url = 'https://www.youtube.com/results?{}search_query={}'.format(_sortby_filter, formatted_search_term)
    print(youtube_url)
    return import_youtube_data(youtube_url)


if __name__ == '__main__':
    _list = search_and_store('broccoli', 'unused', SortBy.ViewCount.value, UploadDate.Default.value, Duration.Short.value, [])
    print('--- Search function ---')
    for link in _list:
        print('Video: {}'.format(link))
        # print('Thumbnail: {}'.format(get_thumbnail(link)))
        # webbrowser.open(get_thumbnail(link))
    print(len(_list))
