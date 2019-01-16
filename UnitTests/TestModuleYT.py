import unittest
from Scripts.WebScraping.ModuleYT import filter_watch_only, import_youtube_data
from Scripts.WebScraping.VerifyRequest import get_verified_response


class ModuleYTRunner(unittest.TestCase):

    def test_happy_path_filter_watchable(self):
        """Tests if youtube links are watchable"""
        youtube_hyperlinks = ['https://www.youtube.com/watch?v=1XW1Ygatsz4']
        filtered_list = filter_watch_only(youtube_hyperlinks)

        self.assertIsInstance(filtered_list, list)
        self.assertEquals(1, len(filtered_list))

    def test_happy_path_filter_non_watchable(self):
        """Tests if youtube links are not watchable"""
        youtube_hyperlinks = ['https://www.youtube.com/user/SonyPictures']
        filtered_list = filter_watch_only(youtube_hyperlinks)

        self.assertIsInstance(filtered_list, list)
        self.assertEquals(0, len(filtered_list))

    def test_bad_path_filter(self):
        """Tests if youtube links are invalid"""
        youtube_hyperlinks = ['']
        filtered_list = filter_watch_only(youtube_hyperlinks)

        self.assertIsInstance(filtered_list, list)
        self.assertEquals(0, len(filtered_list))

    def test_bad_path_filter_random(self):
        """Tests if youtube links are invalid"""
        youtube_hyperlinks = ['', None, 2]
        filtered_list = filter_watch_only(youtube_hyperlinks)

        self.assertIsInstance(filtered_list, list)
        self.assertEquals(0, len(filtered_list))

    def test_happy_path_import_from_search(self):
        """Tests if search term returns valid youtube links"""
        search_term = 'https://www.youtube.com/results?search_query=apple'
        imported_list = import_youtube_data(search_term)

        self.assertIsInstance(imported_list, list)

    def test_happy_path_import_is_filtered(self):
        """Tests if search term returns valid youtube links"""
        search_term = 'https://www.youtube.com/results?search_query=apple'
        imported_list = import_youtube_data(search_term)
        filtered_import_list = imported_list

        self.assertIsInstance(imported_list, list)
        self.assertEquals(len(filtered_import_list), len(imported_list))
        for href in imported_list:
            self.assertEquals(200, get_verified_response(href).status)

    def test_bad_path_import_un_valid_search(self):
        """Tests if search term returns valid youtube links"""
        search_term = 'apple'
        imported_list = import_youtube_data(search_term)

        self.assertIsInstance(imported_list, list)
        self.assertEquals(0, len(imported_list))
