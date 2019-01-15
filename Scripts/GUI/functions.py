from tkinter import *
from Scripts.WebScraping.ModuleYT import search_and_store
from Scripts.WebScraping.FilterTerms import SortBy, Features, UploadDate, Duration

# Global filter Variables
SORTBY = SortBy.Default.value  # String
FEATURES = []  # List<string>
UPLOADDATE = UploadDate.Default.value  # String
DURATION = Duration.Default.value  # String

SORTBY_DICT = {
    'Default': SortBy.Default.value,
    'Relevance': SortBy.Relevance.value,
    'Upload Time': SortBy.UploadTime.value,
    'View Count': SortBy.ViewCount.value,
    'Rating': SortBy.Rating.value
}
FEATURES_DICT = {
    'Subtitles': Features.Subtitles.value,
    'Live': Features.Live.value,
    '4K': Features.FourKResolution.value,
    'High Definition': Features.HighDefinition.value
}
UPLOADDATE_DICT = {
    'Default': UploadDate.Default.value,
    'Past hour': UploadDate.ThisHour.value,
    'Today': UploadDate.ThisDay.value,
    'This week': UploadDate.ThisWeek.value,
    'This month': UploadDate.ThisMonth.value,
    'This year': UploadDate.ThisYear.value
}
DURATION_DICT = {
    'Default': Duration.Default.value,
    'Long': Duration.Long.value,
    'Short': Duration.Short.value
}

def pic_from_gallery():
    file1=filedialog.askopenfile()


# # def window_transition(deletions):
# #     print (deletions)
# #     for label in deletions:
# #         print(label)
# #         try:
# #             label.place_forget()
# #         except:
# #             pass
#
def clear_window():
    empty_window = Frame(root,
                         height=800,
                         width=600,
                         bg=main_bg_colour)
    empty_window.place(x=0,
                       y=0)

def show_main_window():
    clear_window()
    self.main_window()
#
# # def show_main_window():
# #     clear_window()
# #     main_window()
#

# # Close window function
# def close_win():
#     root.destroy()
#
# def close_window():
#     close_que = messagebox.askquestion('Exit', 'Are you sure you want to exit?')
#     if close_que == 'yes':
#         close_win()
#


# # Dropdown menu
#
# menu = Menu(root)
# root.config(menu=menu)
#
# picture_menu = Menu(menu)
# menu.add_cascade(label='Picture',
#                  menu =  picture_menu)
#
# picture_menu.add_command(label = 'Camera')
#
# picture_menu.add_command(label = 'From gallery')
#
# settings = Menu(menu)
# menu.add_command(label='Settings')
#
# exit = Menu(menu)
# menu.add_command(label = 'Exit',command =  close_window)
#

    # def show_result_window():
# #     #         clear_window()
# #     #         result_window()
# #     #
# #     #
# #     #

# #     #
# #     #     def make_pic():
# #     #         pass


def print_URL(string, show_entry_func):
    print('Search')
    list = search_and_store(string, 'unused', SORTBY, UPLOADDATE, DURATION, FEATURES)
    print(list)
    show_entry_func(list)

# Description: Sets the global variable SORTBY everytime the combobox is updated
def sort_combo_func(event=None):
    global SORTBY
    SORTBY = SORTBY_DICT[event.widget.get()]
    print(f'event.widget: {event.widget.get()}')


def feature_func(feature_string):
    global FEATURES
    if feature_string in FEATURES:
        FEATURES.remove(feature_string)
    else:
        FEATURES.append(feature_string)
    print(f'Feature string: {feature_string}')

# Description: Sets the global variable UPLOADDATE everytime the combobox is updated
def upload_combo_func(event=None):
    global UPLOADDATE
    UPLOADDATE = UPLOADDATE_DICT[event.widget.get()]
    print(f'event.widget: {event.widget.get()}')


# Description: Sets the global variable DURATION everytime the combobox is updated
def duration_combo_func(event=None):
    global DURATION
    DURATION = DURATION_DICT[event.widget.get()]
    print(f'event.widget: {event.widget.get()}')

