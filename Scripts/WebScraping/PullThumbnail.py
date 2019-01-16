

# @Description Pulls thumbnail from youtube video link
# @argument <class 'string'>
# @return <class 'string'> or <class 'None'>
def get_thumbnail(_youtube_link):
    link_id = str(_youtube_link).split('v=')
    if len(link_id) > 1:
        link_id = link_id[1]
    else:
        return None
    url = f'https://img.youtube.com/vi/{link_id}/maxresdefault.jpg'
    return url
