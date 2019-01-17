import unittest
from Scripts.WebScraping.VerifyRequest import get_verified_response


class VerifyRequestRunner(unittest.TestCase):

    def test_positive_verification(self):
        """True if url is SNI verified"""
        verified_url = 'https://google.com/'
        self.assertNotEquals(None, get_verified_response(verified_url))

    def test_negative_verification(self):
        """True if url is not SNI verified"""
        unverified_url = 'https://badsni.com/'
        self.assertEquals(None, get_verified_response(unverified_url))

    def test_response_code200(self):
        """True if response code is correct"""
        verified_url = 'https://google.com/'
        self.assertEquals(200, get_verified_response(verified_url).status)

    def test_response_code404(self):
        """True if response code is correct"""
        verified_url = 'https://www.gov.uk/404'
        self.assertEquals(404, get_verified_response(verified_url).status)
