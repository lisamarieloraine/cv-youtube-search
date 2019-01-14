from tkinter import *
import Scripts.GUI.functions as funcs
from Scripts.GUI.page_search import PageSearch

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
        self.take_picture_button = Button(self,
                                     text = 'Take picture',
                                     bg= main_button_colour,
                                     font = main_font,
                                     command = lambda: controller.show_frame(PageSearch))


        self.upload_picture_button =  Button(self,
                                       text='Upload picture',
                                        bg = main_button_colour,
                                        font = main_font,
                                        command = funcs.pic_from_gallery)

        # Button placements
        self.take_picture_button.place(x=300,
                                      y=300,
                                      anchor='center',
                                      width=120,
                                      height=120)

        self.upload_picture_button.place(x=500,
                                        y=300,
                                        anchor='center',
                                        width=120,
                                        height=120)
# Import pictures
# camera_picture = PhotoImage(file = 'C:\\Users\Daniek\camera.png')
# take_picture_button.config(image = camera_picture)
#
# gallery_picture = PhotoImage(file = 'C:\\Users\Daniek\Afbeelding3.png')
# upload_picture_button.config(image = gallery_picture)