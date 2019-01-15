from tkinter import *
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

gui=GUI()
gui.mainloop()
