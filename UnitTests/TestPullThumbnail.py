import unittest
from Scripts.WebScraping.PullThumbnail import get_thumbnail
from Scripts.WebScraping.VerifyRequest import get_verified_response


class PullThumbnailRunner(unittest.TestCase):

    def test_happy_path(self):
        """Tests if url is a valid youtube-link"""
        verified_youtube_url = 'https://www.youtube.com/watch?v=1XW1Ygatsz4'
        thumbnail_url = get_thumbnail(verified_youtube_url)
        response = get_verified_response(thumbnail_url)

        self.assertIsInstance(thumbnail_url, str)
        self.assertEquals(200, response.status)

    def test_bad_path_no_thumbnail(self):
        """Tests if url is a valid youtube-link but no thumbnail"""
        unverified_youtube_url = 'https://www.youtube.com/watch?v=a8wrKC15QMk'
        thumbnail_url = get_thumbnail(unverified_youtube_url)
        response = get_verified_response(thumbnail_url)

        self.assertIsInstance(thumbnail_url, str)
        self.assertEquals(404, response.status)

    def test_bad_path_invalid_url(self):
        """Tests if url is not a valid youtube-video-link"""
        unverified_youtube_url = 'https://www.youtube.com/'
        thumbnail_url = get_thumbnail(unverified_youtube_url)
        self.assertIsInstance(thumbnail_url, type(None))
