from Scripts.WebScraping.VerifyRequest import get_verified_response


# @Description Pulls thumbnail from youtube video link
# @argument <class 'string'>
# @return <class 'urllib3.response.HTTPResponse'>
def get_thumbnail(_youtube_link):
    link_id = str(_youtube_link).split('v=')[1]
    url = f'https://img.youtube.com/vi/{link_id}/maxresdefault.jpg'
    response = get_verified_response(url)
    return url
