from tkinter import *
import Scripts.GUI.functions as funcs
from tkinter import ttk
from Scripts.WebScraping.FilterTerms import SortBy, Features, UploadDate, Duration

main_bg_colour = '#e1c793'
main_button_colour = '#ead7b2'
main_font = 'Comfortaa'
side_bar_colour = '#e5cfa3'


class PageSearch(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller
        self.config(background=main_bg_colour)
                    # Build elements
        self.side_bar = Frame(self,
                                  height = 600,
                                     width = 200,
                                     bg= side_bar_colour )

        self.search_term_label= Label(self,
                                             text= 'Search term',
                                             font = (main_font,15),
                                             bg= side_bar_colour)

        self.sort_by_label = Label(self,
                                          text= 'Sort by',
                                          font= (main_font,10),
                                          bg= side_bar_colour)

        self.sort_combobox = ttk.Combobox(self,
                                                 values=('Default', 'Relevance', 'Upload Time', 'View Count', 'Rating'),
                                                 width=20,
                                                 font=(main_font,10))
        self.sort_combobox.bind('<<ComboboxSelected>>', funcs.sort_combo_func)

        self.upload_date_label = Label(self,
                                              text='Upload Date',
                                              font=(main_font,10),
                                              bg= side_bar_colour)

        self.upload_date_combobox = ttk.Combobox(self,
                                                        values=('Default','Past hour', 'Today', 'This week ', 'This month', 'This year'),
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

        self.FourKResolution_checkbox= Checkbutton(self,
                                                    text='4K',
                                                    width=20,
                                                    bg=side_bar_colour,
                                                   command=lambda: funcs.feature_func(Features.FourKResolution.value))

        self.HighDefinition_checkbox = Checkbutton(self,
                                                 text='High Definition',
                                                 width=20,
                                                 bg=side_bar_colour,
                                                   command=lambda: funcs.feature_func(Features.HighDefinition.value))
        self.go_back_button = Button(self,
                                            width=5,
                                            height=2,
                                            bg= main_button_colour)
        self.result_label =Label (self,
                                         text= 'Results',
                                         font=(main_font,25),
                                         bg= main_bg_colour)

        self.input_searchterm = Entry(self,
                                             textvariable= StringVar)
        self.search_button = Button(self,
                                           command=lambda: funcs.print_URL(self.input_searchterm))




        #Place elements

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
                                        y= 320)

        self.side_bar.place(x=0,
                                   y=0)

        self.go_back_button.place(x=710,
                                         y=20)

        self.result_label.place(x=300,
                                       y=10)

        self.input_searchterm.place(x=300,
                                           y=100)

        self.search_button.place(x=450,
                                       y=100)