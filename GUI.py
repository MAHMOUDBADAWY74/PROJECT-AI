#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter import filedialog

from PIL import ImageTk, Image

from convcam import video_classify, image_classify


def open_files():
    H = Label(root, text="predict image reaction", font=("Times", 25), fg='black', bg="white")
    H.pack()
    H.place(x=400, y=30)
    global filename
    filename = 0
    filename = filedialog.askopenfilename(initialdir=r"/img", title="Select A File",
                                          filetypes=(("jpg files", "*.png"), ("all files", "*.*")))

    # T = Label(root, text="Input", font=("Times", 22), fg='black')
    # T.pack()
    # T.place(x=510,y=100)

    global inp
    inp = ImageTk.PhotoImage(Image.open(filename).resize((400, 250), Image.Resampling.LANCZOS))
    p = w.create_image(350, 150, anchor=NW, image=inp)

    pred = image_classify(filename)

    T = Label(text=pred, font=("Times", 28), fg='black', bg='white')
    T.pack()
    T.place(x=530, y=410)
    pred = []


# global out = ImageTk.PhotoImage(Image.open("C:\\Users\\Lenovo\\Desktop\\project AI\\bahga.jpg").resize((400, 250),
# Image.Resampling.LANCZOS)) q=w.create_image(850,150, anchor=NW, image=out)

def image_c():
    open_file = Button(root, text='Open Image', height=2, width=20, bg='black', fg='white', command=open_files,
                       activebackground='black')
    open_file.pack()
    open_file.place(x=430, y=500)


root = Tk()
root.title("Facial Expression Recognition")
root.resizable(False, False)
w = Canvas(root, width=800, height=670, bg="white")
w.place(x=500, y=500)
w.pack()

Video_execute = Button(root, text='Video Classifier', height=3, width=25, bg='navy blue', fg='white',
                       command=video_classify)
Video_execute.pack()
Video_execute.place(x=20, y=220)

#Image_execute = Button(root, text='Image Classifier', height=3, width=25, bg='navy blue', fg='white', command=image_c)
#Image_execute.pack()
#Image_execute.place(x=20, y=320)

exit = Button(root, text='Exit', height=2, width=15, fg='white', bg='black', command=quit)
exit.pack()
exit.place(x=510, y=600)

w.create_line(240, 0, 240, 890, fill="black")
w.create_line(240, 590, 1350, 590, fill="black")

root.mainloop()

# In[ ]:
