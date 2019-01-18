import PIL
from PIL import Image, ImageTk
import cv2
from tkinter import *

root=Tk()
def open_web():

    width, height = 800, 600
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    webcam_win = Toplevel(root)
    Toplevel.bind('<Escape>', lambda e: webcam_win.quit())
    lmain = Label(webcam_win)
    lmain.pack()

    takePicture = 0  # My variable


    def TakePictureC():  # There is the change of the variable
        global takePicture
        takePicture = takePicture + 1  # Add "1" to the variable


    def show_frame():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)
        global takePicture
        if takePicture == 1:  # My option for take the image
            img.save("test.png")  # Save the instant image
            takePicture = takePicture - 1  # Remove "1" to the variable

    screenTake = Button(webcam_win, text='ScreenShot', command=TakePictureC)  # The button for take picture
    screenTake.pack()  # Pack option to see it

    show_frame()


bu1=Button(root, text = 'click', command = open_web)
bu1.pack()

root.mainloop()


