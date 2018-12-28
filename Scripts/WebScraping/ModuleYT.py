from Scripts.WebScraping.VerifyRequest import get_verified_response
from bs4 import BeautifulSoup


# TODO: Filter on only watchable videos (no user links)
# @Description Lists youtube response videos from given url
# @argument <class 'string'>
# @return <class 'list'>
def import_youtube_data(_youtube_url):
    """Returns list of urls from url response"""
    result = list()
    response = get_verified_response(_youtube_url)  # Get server response from url request
    soup = BeautifulSoup(response.data, 'html.parser')
    for vid in soup.findAll(attrs={'class': 'yt-uix-tile-link'}):
        result.append('https://www.youtube.com' + vid['href'])
    return result


# @Description Higher order function to scrape and store data
# @argument <class 'string'> and <class 'string'>
# @return <class 'list'>
def search_and_store(_search_term, _filename):
    """Returns list of search results while storing them in json"""
    print('Scraping Youtube data')
    # epoch_time = int(round(time.time() * 1000))
    youtube_url = f'https://www.youtube.com/results?search_query={_search_term}'
    return import_youtube_data(youtube_url)


if __name__ == '__main__':
    _list = search_and_store('qtpie', 'unused')
    print('--- Update data func ---')
    for link in _list:
        print(link)

    # search_query = f'https://www.youtube.com/results?search_query=banana&pbj=1'
