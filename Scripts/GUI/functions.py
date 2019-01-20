from tkinter import *
import cv2
from Scripts.WebScraping.ModuleYT import search_and_store
from Scripts.WebScraping.FilterTerms import SortBy, Features, UploadDate, Duration
import Scripts.ImageRecognition.main as main
from tkinter import filedialog
# from VideoCapture import Device

# Global Picture Variable
SEARCHPICTURE = None
SEARCHPICTURE_DICT = {}  # search term <string> ##(, url list <list>)

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


# def TakePictureC():  # There is the change of the variable
#     global takePicture
#     takePicture = takePicture + 1  # Add "1" to the variable
#
#
# def show_frame():
#     _, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img = PIL.Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     lmain.imgtk = imgtk
#     lmain.configure(image=imgtk)
#     lmain.after(10, show_frame)
#     global takePicture
#     if takePicture == 1:  # My option for take the image
#         img.save("test.png")  # Save the instant image
#         takePicture = takePicture - 1  # Remove "1" to the variable

def prepair_browser_search(frame_data, frame_controller):
    """Opens up search page while checking if globally selected picture is available
    Otherwise, open file browser. Then throw picture through CNN and provide a search term"""
    global SEARCHPICTURE, SEARCHPICTURE_DICT
    if SEARCHPICTURE is None:
        upload_picture()  # Open file browser
        # Open Screen capture device
        # Possibility to upload picture and denied
        if SEARCHPICTURE is None:
            return None

    if SEARCHPICTURE in SEARCHPICTURE_DICT:
        frame_data.input_searchterm = SEARCHPICTURE_DICT[SEARCHPICTURE]
    else:
        new_search_term = main.run(write = False, predict = True, image = SEARCHPICTURE)
        print('CNN conclusion: {}'.format(new_search_term))
        frame_controller.frames[frame_data].input_searchterm = new_search_term
        frame_controller.frames[frame_data].search_term_label.config(text='CNN: {}'.format(new_search_term))
        SEARCHPICTURE_DICT[SEARCHPICTURE] = new_search_term

    print_URL(frame_controller.frames[frame_data].input_searchterm, frame_controller.frames[frame_data].show_thumbnails)
    return frame_controller.show_frame(frame_data)


def prepair_webcam_search(frame_data, frame_controller):
    """Opens up webcam search page while checking if globally selected picture is available
        Otherwise, open webcam. Then throw picture through CNN and provide a search term"""
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def upload_picture():
    file1 = str(filedialog.askopenfilename(initialdir="/", title="Select file",
                                           filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*"))))
    global SEARCHPICTURE
    SEARCHPICTURE = file1
    print(file1)
    return file1


# Close window function
def close_win():
    root.destroy()

def close_window():
    close_que = messagebox.askquestion('Exit', 'Are you sure you want to exit?')
    if close_que == 'yes':
        close_win()



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
    print('event.widget: {}'.format(event.widget.get()))


def feature_func(feature_string):
    global FEATURES
    if feature_string in FEATURES:
        FEATURES.remove(feature_string)
    else:
        FEATURES.append(feature_string)
    print('Feature string: {}'.format(feature_string))


# Description: Sets the global variable UPLOADDATE everytime the combobox is updated
def upload_combo_func(event=None):
    global UPLOADDATE
    UPLOADDATE = UPLOADDATE_DICT[event.widget.get()]
    print('event.widget: {}'.format(event.widget.get()))


# Description: Sets the global variable DURATION everytime the combobox is updated
def duration_combo_func(event=None):
    global DURATION
    DURATION = DURATION_DICT[event.widget.get()]
    print('event.widget: {}'.format(event.widget.get()))
