from enum import Enum


# Ends with '%253D'
class SortBy(Enum):
    Relevance = 'sp=CAA%253D&'
    UploadTime = 'sp=CAI%253D&'
    ViewCount = 'sp=CAM%253D&'
    Rating = 'sp=CAE%253D&'


class UploadDate(Enum):
    ThisHour = 'hour'  # 'SBAgBEAE'
    ThisDay = 'today'  # 'SBAgCEAE'
    ThisWeek = 'week'  # 'SBAgDEAE'
    ThisMonth = 'month'  # 'SBAgEEAE'
    ThisYear = 'year'  # 'SBAgFEAE'


class Features(Enum):
    Live = 'live'
    FourKResolution = '4k'
    HighDefinition = 'hd'
    Subtitles = 'cc'


class Duration(Enum):
    Long = 'long'
    Short = 'short'
