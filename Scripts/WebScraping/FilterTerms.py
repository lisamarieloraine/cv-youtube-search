from enum import Enum


# Ends with '%253D'
class SortBy(Enum):
    Relevance = 'sp=CAA'
    UploadTime = 'sp=CAI'
    ViewCount = 'sp=CAM'
    Rating = 'sp=CAE'


class UploadDate(Enum):
    LastHour = 'SBAgBEAE'
    Today = 'SBAgCEAE'
    ThisWeek = 'SBAgDEAE'
    ThisMonth = 'SBAgEEAE'
    ThisYear = 'SBAgFEAE'
