from enum import Enum


# Ends with '%253D'
class SortBy(Enum):
    Default = ''
    Relevance = 'sp=CAA%253D&'
    UploadTime = 'sp=CAI%253D&'
    ViewCount = 'sp=CAM%253D&'
    Rating = 'sp=CAE%253D&'


class UploadDate(Enum):
    Default = ''
    ThisHour = 'hour'  # 'SBAgBEAE'
    ThisDay = 'today'  # 'SBAgCEAE'
    ThisWeek = 'week'  # 'SBAgDEAE'
    ThisMonth = 'month'  # 'SBAgEEAE'
    ThisYear = 'year'  # 'SBAgFEAE'


class Features(Enum):
    Default = ''
    Live = 'live'
    FourKResolution = '4k'
    HighDefinition = 'hd'
    Subtitles = 'cc'


class Duration(Enum):
    Default = ''
    Long = 'long'
    Short = 'short'
