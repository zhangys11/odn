from pathlib import Path
from tkinter import Canvas
from PIL import ImageTk,Image
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import utility
from ttkbootstrap.toast import ToastNotification

PATH = Path(__file__).parent
global IMAGE_SIZE 
IMAGE_SIZE = (400,300)

class ToastxFrame(ttk.Frame):

    def __init__(self, master, image):
        super().__init__(master)
        self.pack(fill=BOTH, expand=YES)
        self.master = master
        pil_img = Image.open(image)
        
        global IMAGE_SIZE 

        # resize image when image dim exceed the preset size
        if pil_img.size[0] <= IMAGE_SIZE[0] and pil_img.size[1] <= IMAGE_SIZE[1]:
            IMAGE_SIZE = pil_img.size
        else:
            pil_img = pil_img.resize(IMAGE_SIZE)

        self.canvas = Canvas(self.master, width = IMAGE_SIZE[0], height = IMAGE_SIZE[1])
        self.img = ImageTk.PhotoImage(pil_img)  
        self.canvas.create_image(0, 0, anchor=NW, image=self.img) 
        self.canvas.pack()

class ToastNotificationX(ToastNotification):
    """
    An extended version of ToastNotification that supports image
    """

    def __init__(
        self,
        image=None,
        title='',        
        message='',
        duration=None,
        bootstyle=LIGHT,
        alert=False,
        icon=None,
        iconfont=None,
        position=None,
        **kwargs,
    ):
        if image is not None:
            self.img = ImageTk.PhotoImage(Image.open(image).resize(IMAGE_SIZE))  # must persist as a class member
        else:
            self.img = None

        ToastNotification.__init__(self, title,
        message,
        duration,
        bootstyle,
        alert,
        icon,
        iconfont,
        position,
        **kwargs)

    def show_toast(self, *_):
        """Create and show the toast window."""

        # build toast
        self.toplevel = ttk.Toplevel(**self.kwargs)
        self._setup(self.toplevel)

        self.container = ttk.Frame(self.toplevel, 
        bootstyle=self.bootstyle)
        self.container.pack(fill=BOTH, expand=YES)

        if self.img is None:
            # img = ImageTk.PhotoImage(Image.open(self.image).resize((80,80)))
            ttk.Label(
                self.container,
                text=self.icon,
                font=self.iconfont,
                #image=img,
                bootstyle=f"{self.bootstyle}-inverse",
                anchor=NW,
            ).grid(row=0, column=0, rowspan=2, sticky=NSEW, padx=(5, 0))

            ttk.Label(
                self.container,
                text=self.title,
                font=self.titlefont,
                bootstyle=f"{self.bootstyle}-inverse",
                anchor=NW,
            ).grid(row=0, column=1, sticky=NSEW, padx=10, pady=(5, 0))
            
            ttk.Label(
                self.container,
                text=self.message,
                wraplength=utility.scale_size(self.toplevel, 300),
                bootstyle=f"{self.bootstyle}-inverse",
                anchor=NW,
            ).grid(row=1, column=1, sticky=NSEW, padx=10, pady=(0, 5))
        else:
            canv = Canvas(self.container, width = IMAGE_SIZE[0], height=IMAGE_SIZE[1])
            canv.create_image(round(IMAGE_SIZE[0]/2), round(IMAGE_SIZE[1]/2),image=self.img) # 20, 20, anchor=NW
            # canv.create_rectangle(10,10,400,300)
            # canv.create_oval(10, 10, 400, 300)
            # canv.pack(fill=BOTH, expand=YES)
            canv.grid(row=0, column = 0, rowspan=2, columnspan=2, sticky=NSEW)         

        self.toplevel.bind("<ButtonPress>", self.hide_toast)

        # alert toast
        if self.alert:
            self.toplevel.bell()

        # specified duration to close
        if self.duration:
            self.toplevel.after(self.duration, self.hide_toast)

if __name__ == "__main__":

    app = ttk.Window()

    ToastxFrame(app, "C:/Users/eleve/Desktop/panda.jpg")

    ToastNotificationX(
        "ttkbootstrap toast message",
        "This is a toast message; you can place a symbol on the top-left that is supported by the selected font. You can either make it appear for a specified period of time, or click to close.",
        image = 'C:/Users/eleve/Desktop/panda.jpg',
        duration=4000,
    ).show_toast()

    app.mainloop()
