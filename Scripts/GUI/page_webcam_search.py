import io
import webbrowser
from Scripts.GUI.functions import upload_picture
from tkinter import *
import Scripts.GUI.functions as funcs
from tkinter import ttk
from PIL import Image, ImageTk
from Scripts.WebScraping.FilterTerms import Features
from Scripts.WebScraping.VerifyRequest import get_verified_response
from Scripts.WebScraping.PullThumbnail import get_thumbnail
import Scripts.GUI.Homepage
import Scripts.ImageRecognition
import Scripts.ImageRecognition.main as main
import Scripts.ImageRecognition.data
import os
import sys

main_bg_colour = '#e1c793'
main_button_colour = '#ead7b2'
main_font = 'Comfortaa'
side_bar_colour = '#e5cfa3'


class PageWebcamSearch(Frame):
    _image = []
    _thumbnail_list = []

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller
        self.config(background=main_bg_colour)

        # Build elements
        self.side_bar = Frame(self, height=600,
                              width=200,
                              bg=side_bar_colour)
        picture = upload_picture()
        self.input_searchterm = main.run(write = False, predict = True, image = picture)
        self.search_term_label = Label(self, text=self.input_searchterm,
                                       font=(main_font, 15),
                                       bg=side_bar_colour)

        self.sort_by_label = Label(self, text='Sort by',
                                   font=(main_font, 10),
                                   bg=side_bar_colour)

        self.sort_combobox = ttk.Combobox(self, values=('Default', 'Relevance', 'Upload Time', 'View Count', 'Rating'),
                                          width=20,
                                          font=(main_font, 10))

        self.sort_combobox.bind('<<ComboboxSelected>>', funcs.sort_combo_func)

        self.upload_date_label = Label(self, text='Upload Date',
                                       font=(main_font, 10),
                                       bg=side_bar_colour)

        self.upload_date_combobox = ttk.Combobox(self, values=(
            'Default', 'Past hour', 'Today', 'This week ', 'This month', 'This year'),
                                                 width=20,
                                                 font=(main_font, 10))

        self.upload_date_combobox.bind('<<ComboboxSelected>>', funcs.upload_combo_func)

        self.duration_label = Label(self,
                                    text='Duration',
                                    font=(main_font, 10),
                                    bg=side_bar_colour)

        self.duration_combobox = ttk.Combobox(self,
                                              values=('Default', 'Long', 'Short'),
                                              width=20,
                                              font=(main_font, 10))
        self.duration_combobox.bind('<<ComboboxSelected>>', funcs.duration_combo_func)

        self.filter_label = Label(self,
                                  text='Filter',
                                  font=(main_font, 10),
                                  bg=side_bar_colour)

        self.subtitle_checkbox = Checkbutton(self,
                                             text='Subtitles',
                                             width=20,
                                             bg=side_bar_colour,
                                             command=lambda: funcs.feature_func(Features.Subtitles.value))

        self.live_checkbox = Checkbutton(self,
                                         text='Live',
                                         width=20,
                                         bg=side_bar_colour,
                                         command=lambda: funcs.feature_func(Features.Live.value))

        self.FourKResolution_checkbox = Checkbutton(self,
                                                    text='4K',
                                                    width=20,
                                                    bg=side_bar_colour,
                                                    command=lambda: funcs.feature_func(Features.FourKResolution.value))

        self.HighDefinition_checkbox = Checkbutton(self,
                                                   text='High Definition',
                                                   width=20,
                                                   bg=side_bar_colour,
                                                   command=lambda: funcs.feature_func(Features.HighDefinition.value))

        photo_go_back = PhotoImage(file=os.path.join(sys.path[0], 'Images\go_back.png'))
        self.go_back_button = Button(self,
                                     width=35,
                                     height=35,
                                     bg=main_button_colour,
                                     image=photo_go_back,
                                     command=lambda: controller.show_frame(Scripts.GUI.Homepage.Homepage))
        self.go_back_button.image = photo_go_back

        self.result_label = Label(self,
                                  text='Results',
                                  font=(main_font, 25),
                                  bg=main_bg_colour)

        self.search_button = Button(self, text="Search",
                                    height=1,
                                    width=8,
                                    bg=main_button_colour,
                                    font= (main_font, 9),
                                    command=lambda: funcs.print_URL(self.input_searchterm, self.show_thumbnails))



        # Place elements

        self.search_term_label.place(x=10,
                                     y=10)

        self.sort_by_label.place(x=10,
                                 y=60)

        self.sort_combobox.place(x=10,
                                 y=90)

        self.upload_date_label.place(x=10,
                                     y=120)

        self.upload_date_combobox.place(x=10,
                                        y=150)

        self.duration_label.place(x=10,
                                  y=180)

        self.duration_combobox.place(x=10,
                                     y=210)

        self.filter_label.place(x=10,
                                y=240)

        self.subtitle_checkbox.place(x=10,
                                     y=260)

        self.FourKResolution_checkbox.place(x=-6,
                                            y=280)

        self.HighDefinition_checkbox.place(x=28,
                                           y=300)

        self.live_checkbox.place(x=-2,
                                 y=320)

        self.side_bar.place(x=0,
                            y=0)

        self.go_back_button.place(x=710,
                                  y=20)

        self.result_label.place(x=260,
                                y=10)

        self.search_button.place(x=50,
                                 y=360)



    def show_thumbnails(self, _link_list):
        print('Converting url to image')

        # Thumbnail image
        def open_url(url):
            return lambda x: webbrowser.open_new(url)

        w = 150
        h = 80

        # self._image.place_forget();
        [button.place_forget() for button in self._thumbnail_list]

        for i, url in enumerate(_link_list):
            if i == 15:
                break
            thumbnail_url = get_thumbnail(url)
            print(thumbnail_url)
            response = get_verified_response(thumbnail_url)
            im = Image.open(io.BytesIO(response.data))
            im = im.resize((w, h), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(im)

            thumbnail_button = Button(self, image=image,
                                      width=w,
                                      height=h)
            thumbnail_button.place(x=260 + (0 if i < 5 else (1.1 * w if i < 10 else 2.2 * w)),
                                   y=150 + (i % 5) * (h * 1.1))
            # self._lambda_list.append(lambda x: webbrowser.open_new(_link_list[i]))
            thumbnail_button.bind("<Button-1>", open_url(url))

            self._image.append(image)
            self._thumbnail_list.append(thumbnail_button)
