from Scripts.WebScraping.VerifyRequest import get_verified_response
from Scripts.WebScraping.HTMLConversion import response_to_xml
import time


# @Description Scrapes youtube video data from given url
# @argument <class 'string'>
# @return <class ''>
def import_youtube_data(_youtube_url):
    """Returns object with values read from url response"""
    response = get_verified_response(_youtube_url)  # Get server response from url request
    xml_element = response_to_xml(response.data)  # Convert response data to xml element
    # return stats_obj


# @Description Higher order function to scrape and store data
# @argument <class 'string'> and <class 'string'>
# @return <class ''>
def update_data(_filename):
    """Returns object"""
    print('Scraping Youtube data')
    epoch_time = int(round(time.time() * 1000))

    youtube_url = f'{epoch_time}'

    import_youtube_data(youtube_url)
    # return stats_obj

