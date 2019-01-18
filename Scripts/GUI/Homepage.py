import Scripts.GUI.functions as funcs
from Scripts.GUI.page_search import PageSearch

from Scripts.GUI.page_webcam_search import PageWebcamSearch
from PIL import Image, ImageTk
import cv2
from tkinter import *

from Scripts.GUI.functions import upload_picture

main_bg_colour = '#e1c793'
main_button_colour = '#ead7b2'
main_font = 'Comfortaa'
side_bar_colour = '#e5cfa3'


class Homepage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller
        self.config(background=main_bg_colour)

        # Create Buttons

        photo_take = PhotoImage(file='camera.png')
        self.take_picture_button = Button(self,
                                          text='Take picture',
                                          bg=main_button_colour,
                                          image=photo_take,
                                          font=main_font,
                                          command=lambda: controller.show_frame(PageWebcamSearch))
        self.take_picture_button.image = photo_take

        photo_upload = PhotoImage(file='Afbeelding3.png')
        self.upload_picture_button = Button(self,
                                            text='Upload picture',
                                            bg=main_button_colour,
                                            font=main_font,
                                            image=photo_upload,
                                            command=upload_picture)
        self.upload_picture_button.image = photo_upload

        photo_search = PhotoImage(file='search.png')
        self.self_search_button = Button(self,
                                         text='Search',
                                         font=main_font,
                                         bg=main_button_colour,
                                         image=photo_search,
                                         command=lambda: controller.show_frame(PageSearch))
        self.self_search_button.image = photo_search

        self.webcame_test_button = Button(self,
                                          text='test')

        # Button placements
        self.take_picture_button.place(x=200,
                                       y=300,
                                       anchor='center',
                                       width=120,
                                       height=120)

        self.upload_picture_button.place(x=400,
                                         y=300,
                                         anchor='center',
                                         width=120,
                                         height=120)

        self.self_search_button.place(x=600,
                                      y=300,
                                      anchor='center',
                                      width=120,
                                      height=120)
        self.webcame_test_button.place(x=10,
                                       y=10)


# Import pictures

#
# gallery_picture = PhotoImage(file = 'C:\\Users\Daniek\Afbeelding3.png')
# upload_picture_button.config(image = gallery_picture)
