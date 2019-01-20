from tkinter import *
import PIL
from PIL import Image, ImageTk
#import cv2

from Scripts.GUI.Homepage import Homepage
from Scripts.GUI.page_search import PageSearch
from Scripts.GUI.page_webcam_search import PageWebcamSearch
from tkinter import messagebox #To be able to have pop up message
from tkinter import filedialog
from tkinter import ttk
from Scripts.WebScraping.ModuleYT import search_and_store


class GUI(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        container = Frame(self)
        self.geometry("700x800")
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)


        # Main lay out choices
        self.main_bg_colour = '#e1c793'
        self.main_button_colour = '#ead7b2'
        self.main_font = 'Comfortaa'
        self.side_bar_colour = '#e5cfa3'

        # Lay out main window
        self.geometry("800x600")  # You want the size of the app to be 600x600
        self.resizable(0, 0)  # Don't allow resizing in the x or y direction
        self.configure(bg= self.main_bg_colour)
        self.title('Name')


        #Building Frames
        self.frames = {}

        for F in (Homepage, PageSearch, PageWebcamSearch):
            frame = F(container, self)
            self.frames[F] = frame

            frame.grid(row=0,
                   column=0,
                   sticky='nsew')

        self.show_frame(Homepage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        frame.update()

    # def open_webcame(self):
    #     webcame_win= Toplevel(self,
    #                                bg='#e1c793')
    #     webcame_win.geometry('400x300')
    #
    #     width, height = 800, 600
    #     cap = cv2.VideoCapture(0)
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #
    #     webcame_win.bind('<Escape>', lambda e: webcame_win.quit())
    #     lmain = Label(webcame_win)
    #     lmain.pack()
    #
    #     takePicture = 0  # My variable
    #
    #     takePicture = 0  # My variable
    #
    #     def TakePictureC():  # There is the change of the variable
    #         global takePicture
    #         takePicture = takePicture + 1  # Add "1" to the variable
    #
    #     def show_frame():
    #         _, frame = cap.read()
    #         frame = cv2.flip(frame, 1)
    #         cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #         img = PIL.Image.fromarray(cv2image)
    #         imgtk = ImageTk.PhotoImage(image=img)
    #         lmain.imgtk = imgtk
    #         lmain.configure(image=imgtk)
    #         lmain.after(10, show_frame)
    #         global takePicture
    #         if takePicture == 1:  # My option for take the image
    #             img.save("test.png")  # Save the instant image
    #             takePicture = takePicture - 1  # Remove "1" to the variable
    #
    #     screen_take = Button(webcame_win, text='ScreenShot', command=TakePictureC)  # The button for take picture
    #     screen_take.pack()  # Pack option to see it
    #
    #     show_frame()
#
# gui=GUI()
# gui.mainloop()
