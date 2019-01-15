import certifi
import urllib3


# @Description HTTPResponse if url is SNI verified else None
# @argument <class 'string'>
# @return <class 'urllib3.response.HTTPResponse'> or None
def get_verified_response(_url):
    """Returns request response if verified"""
    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where(),
        timeout=urllib3.Timeout(connect=2.0, read=2.0)
    )
    try:
        requests = http.request('GET', _url)  # http.request
        return requests
    except Exception:
        return None


if __name__ == "__main__":
    response = get_verified_response('https://img.youtube.com/vi/yjMyTDS54HQ/maxresdefault.jpg')
    print(response.status)
