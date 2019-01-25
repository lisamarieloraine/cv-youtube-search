from tkinter import *
import cv2
import os
import sys
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
    # if SEARCHPICTURE is None:
    upload_picture()  # Open file browser
    # Open Screen capture device
    # Possibility to upload picture and denied
    return open_search_page(frame_data, frame_controller)


# @return FileLocation if image has been made else None
def prepair_webcam_search(frame_data, frame_controller):
    """Opens up webcam search page while checking if globally selected picture is available
        Otherwise, open webcam. Then throw picture through CNN and provide a search term"""
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = None

    while True:
        ret, frame = cam.read()
        cv2.putText(frame, '[Space] - Screenshot', (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, '[Esc] - Exit', (10, 70), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        sees = '... loading'  # main.run(write = False, predict = True, image = frame)
        cv2.putText(frame, 'CNN prediction: {}'.format(sees), (10, 110), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_0.jpg"
            location = os.path.join(sys.path[0], img_name)
            cv2.imwrite(location, frame)
            print("{} written! at {}".format(img_name, location))
            break

    cam.release()
    cv2.destroyAllWindows()

    if location is not None:
        print('Screenshot will be processed, opening search page')
        # Startup search page
        global SEARCHPICTURE
        SEARCHPICTURE = location
        # prepair_browser_search(frame_data, frame_controller)
    else:
        print('No screenshot taken, back to main menu')
    
    # Ugly copy
    return open_search_page(frame_data, frame_controller)


def open_search_page(frame_data, frame_controller):
    if SEARCHPICTURE is None:
        return None
    # if SEARCHPICTURE in SEARCHPICTURE_DICT:
    #     frame_data.input_searchterm = SEARCHPICTURE_DICT[SEARCHPICTURE]
    # else:
    new_search_term = main.run(write=False, predict=True, image=SEARCHPICTURE)
    print('CNN conclusion: {}'.format(new_search_term))
    frame_controller.frames[frame_data].input_searchterm = new_search_term
    frame_controller.frames[frame_data].search_term_label.config(text='CNN: {}'.format(new_search_term))
    SEARCHPICTURE_DICT[SEARCHPICTURE] = new_search_term
    print_URL(frame_controller.frames[frame_data].input_searchterm, frame_controller.frames[frame_data].show_thumbnails)

    return frame_controller.show_frame(frame_data)


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


def print_URL(string, show_entry_func):
    if string == 'object not supported':
        list_links = ['https://www.youtube.com/watch?v=dQw4w9WgXcQ']
    else:
        list_links = search_and_store(string, 'unused', SORTBY, UPLOADDATE, DURATION, FEATURES)
    show_entry_func(list_links)


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
