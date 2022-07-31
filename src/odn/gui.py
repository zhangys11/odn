import pathlib
from queue import Queue
from threading import Thread
from tkinter.filedialog import askdirectory
from turtle import width
from regex import B
from send2trash import send2trash
import shutil
import requests
import os
import sys
from datetime import datetime

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import utility
from ttkbootstrap.dialogs.dialogs import Messagebox
from ttkbootstrap.icons import Emoji
from ttkbootstrap.toast import ToastNotification
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

if __package__:
    from .toastx import ToastNotificationX
    from . import predict_fundus_folder
    from .waiting_frame import WaitingFrame
else:
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # root directory, i.e., odn
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        print('*** Add odn root to sys.path: ', ROOT)

    from toastx import ToastNotificationX
    from __init__ import predict_fundus_folder
    from waiting_frame import WaitingFrame

USE_PROGRESSBAR = False
USE_GRID_LAYOUT = False
DEFAULT_WINSIZE = '640x480'
TREEVIEW_ROW_HEIGHT = 21
VERBOSE = False

class FileSearchFrame(ttk.Frame):

    queue = Queue()
    searching = False

    def __init__(self, master, _path, init = True, fix_path = False):
        super().__init__(master, padding=15)
        self.pack(fill=BOTH, expand=YES)
        self.master = master

        '''
        Parameters
        ----------
        _path : default target folder
        init : whether perform search on startup
        fix_path : whether allow user to change target folder
        '''
        
        # print(_path)

        # application variables
        if _path is None or _path == '':
            _path = pathlib.Path().absolute().as_posix()

        self.path_var = ttk.StringVar(value=_path)
        self.term_var = ttk.StringVar(value='.jpg,.png')
        self.type_var = ttk.StringVar(value='endswidth')

        # header and labelframe option container
        # option_text = "Complete the form to begin your search"
        # self.option_lf = ttk.Labelframe(self, text=option_text, padding=5) 
        # self.option_lf.pack(fill=X, expand=YES, anchor=N)

        #ttk.Label(self,text="用户名：").grid(row=0,column=0,sticky=EW)
        #ttk.Label(self,text="密  码：").grid(row=1,column=0,sticky=EW)
        #ttk.Entry(self).grid(row=0,column=1,sticky=EW)
        #ttk.Entry(self,show="*").grid(row=1,column=1,sticky=EW)


        self.create_path_row(fix_path)
        self.create_term_row()
        self.create_type_row()
        self.create_results_view()
        
        if USE_PROGRESSBAR:

            self.progressbar = ttk.Progressbar(
                master=self, 
                mode=INDETERMINATE, 
                bootstyle=(STRIPED, SUCCESS)
            )
            self.progressbar.pack(fill=X, expand=YES)

        self.create_context_menu() # preview | delete | upload, etc.

        if init:
            self.on_search() # perform search on startup
            # print(self.resultview.get_children())
            # self.resultview.focus_set(END)
            # self.resultview.selection_clear()

        # self.first_load = True
        self.window_height = self.master.winfo_height()
        self.master.bind('<Configure>', self.window_resize)

    def window_resize(self, event = None):
        if event is not None:
            if self.master.winfo_height() != self.window_height:
                rows = round( self.master.winfo_height() / 1.4 / TREEVIEW_ROW_HEIGHT - 7 )
                if VERBOSE:
                    print('winfo_width/height = ', self.master.winfo_width(), self.master.winfo_height() ) 
                    print('treeview rows = ', rows)
                self.resultview.configure( height = rows)
                self.window_height = self.master.winfo_height()

    def create_path_row(self, fix_path = False, use_grid_layout = USE_GRID_LAYOUT):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self, height = 20)
        if use_grid_layout:
            path_row.grid(row=0, sticky=EW)
        else:
            path_row.pack(fill=X, expand=YES)
        
        path_lbl = ttk.Label(path_row, text="目录", width=8)
        if use_grid_layout:
            path_lbl.grid(row=0, column=0, columnspan=1, sticky=W)
        else:
            path_lbl.pack(side=LEFT, padx=(15, 0))

        path_ent = ttk.Entry(path_row, textvariable=self.path_var, state = DISABLED if fix_path else NORMAL)
        if use_grid_layout:
            path_ent.grid(row=0, column=1, columnspan=1, sticky=W)
        else:
            path_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)

        browse_btn = ttk.Button(
            master=path_row, 
            text="设置目录", 
            command=self.on_browse, 
            width=8,
            state = DISABLED if fix_path else NORMAL
        )
        if use_grid_layout:
            browse_btn.grid(row=0, column=2, columnspan=1, sticky=W)
        else:
            browse_btn.pack(side=LEFT, padx=5)

    def create_term_row(self, use_grid_layout = USE_GRID_LAYOUT):
        """Add term row to labelframe"""
        term_row = ttk.Frame(self, height = 20)
        if use_grid_layout:
            term_row.grid(row=1, sticky=EW)
        else:
            term_row.pack(fill=X, expand=YES, pady=15)

        term_lbl = ttk.Label(term_row, text="关键词", width=8)
        if use_grid_layout:
            term_lbl.grid(row=0, column=0, columnspan=1, sticky=W)
        else:
            term_lbl.pack(side=LEFT, padx=(15, 0))

        term_ent = ttk.Entry(term_row, textvariable=self.term_var)
        if use_grid_layout:
            term_ent.grid(row=0, column=1, columnspan=1, sticky=W)
        else:
            term_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)

        search_btn = ttk.Button(
            master=term_row, 
            text="检索图片", 
            command=self.on_search, 
            bootstyle=OUTLINE, 
            width=8
        )
        if use_grid_layout:
            search_btn.grid(row=0, column=2, columnspan=1, sticky=W)
        else:
            search_btn.pack(side=LEFT, padx=5)

    def create_type_row(self, use_grid_layout = USE_GRID_LAYOUT):
        """Add type row to labelframe"""
        type_row = ttk.Frame(self)
        if use_grid_layout:
            type_row.grid(row=2, sticky=EW)
        else:
            type_row.pack(fill=X, expand=YES)

        type_lbl = ttk.Label(type_row, text=" 关键词类型", width=12)
        if use_grid_layout:
            type_lbl.grid(row=0, column=0, columnspan=1, sticky=E)
        else:
            type_lbl.pack(side=LEFT, padx=(15, 0))

        contains_opt = ttk.Radiobutton(
            master=type_row, 
            text="包含", 
            variable=self.type_var, 
            value="contains"
        )
        if use_grid_layout:
            contains_opt.grid(row=0, column=1, columnspan=1, sticky=W)
        else:
            contains_opt.pack(side=LEFT)
        
        startswith_opt = ttk.Radiobutton(
            master=type_row, 
            text="前缀", 
            variable=self.type_var, 
            value="startswith"
        )
        if use_grid_layout:
            startswith_opt.grid(row=0, column=2, columnspan=1, sticky=W)
        else:
            startswith_opt.pack(side=LEFT, padx=15)
        
        endswith_opt = ttk.Radiobutton(
            master=type_row, 
            text="后缀", 
            variable=self.type_var, 
            value="endswith"
        )
        if use_grid_layout:
            endswith_opt.grid(row=0, column=3, columnspan=1, sticky=W)
        else:
            endswith_opt.pack(side=LEFT)
        endswith_opt.invoke()

    def clear_results_view(self):
        # for item in self.resultview.get_children():
        #    self.resultview.delete(item)
        self.resultview.delete(*self.resultview.get_children())

    def create_results_view(self, use_grid_layout = USE_GRID_LAYOUT):
        """Add result treeview to labelframe"""
        self.resultview = ttk.Treeview(
            master=self, 
            bootstyle=INFO, 
            columns=[0, 1, 2, 3, 4],
            show=HEADINGS,
            # height= 6, # The desired height of the widget, in rows. This will be dynamically set in window_size()
        )
        ttk.Style().configure('Treeview', rowheight = TREEVIEW_ROW_HEIGHT, font=('', round(TREEVIEW_ROW_HEIGHT * 0.5)))

        if use_grid_layout:
            self.resultview.grid(row=3,sticky=EW)
        else:
            self.resultview.pack(fill=X, expand=YES, pady=5)

        # setup columns and use `scale_size` to adjust for resolution
        self.resultview.heading(0, text='文件名', anchor=W)
        self.resultview.heading(1, text='时间戳', anchor=W)
        self.resultview.heading(2, text='类型', anchor=E)
        self.resultview.heading(3, text='大小', anchor=E)
        self.resultview.heading(4, text='路径', anchor=W)
        self.resultview.column(
            column=0, 
            anchor=W, 
            width=utility.scale_size(self, 125), 
            stretch=False
        )
        self.resultview.column(
            column=1, 
            anchor=W, 
            width=utility.scale_size(self, 140), 
            stretch=False
        )
        self.resultview.column(
            column=2, 
            anchor=E, 
            width=utility.scale_size(self, 50), 
            stretch=False
        )
        self.resultview.column(
            column=3, 
            anchor=E, 
            width=utility.scale_size(self, 50), 
            stretch=False
        )
        self.resultview.column(
            column=4, 
            anchor=W, 
            width=utility.scale_size(self, 300)
        )

    def create_context_menu(self, use_grid_layout = USE_GRID_LAYOUT):

        PADDING = 9

        container = ttk.Frame(self)
        if use_grid_layout:
            container.grid(row = 4, sticky=EW)
        else:
            container.pack(fill=X, expand=YES, anchor=S)
        ttk.Style().configure('TButton', font="-size 13")
        
        btn = ttk.Button(
            master=container,
            text='预览',
            padding=PADDING,
            command= self.preview_image
        )
        btn.pack(side=LEFT, fill=X, expand=YES)  

        btn = ttk.Button(
            master=container,
            text='删除',
            padding=PADDING,
            command= self.delete_image
        )
        btn.pack(side=LEFT, fill=X, expand=YES)     

        btn = ttk.Button(
            master=container,
            text='FRCNN',
            padding=PADDING,
            state=DISABLED,
            command= lambda : self.predict('FRCNN')
        )
        btn.pack(side=LEFT, fill=X, expand=YES)    

        btn = ttk.Button(
            master=container,
            text='SSD',
            padding=PADDING,
            state=DISABLED,
            command= lambda : self.predict('SSD')
        )
        btn.pack(side=LEFT, fill=X, expand=YES)    

        btn = ttk.Button(
            master=container,
            text='YOLO5',
            padding=PADDING,
            # state=DISABLED,
            command= lambda : self.predict('YOLO5')
        )
        btn.pack(side=LEFT, fill=X, expand=YES)    

        btn = ttk.Button(
            master=container,
            text='退出',
            padding=PADDING,
            command= self.master.destroy
        )
        btn.pack(side=LEFT, fill=X, expand=YES)         

    def get_current_selection(self):

        if self.resultview.selection() is None or len(self.resultview.selection()) <= 0:
            if self.master.attributes('-fullscreen'):
                toast = ToastNotification(
                    title='Error',
                    message='需要先选中一张图片', 
                    duration=1000,
                )
                toast.show_toast()
            else:
                Messagebox.show_error('需要先选中一张图片','Error')
            return None

        curItem = self.resultview.selection()[0] # self.resultview.focus()
        return curItem

    def preview_image(self):

        curItem = self.get_current_selection()
        if curItem:
            imgpath = self.resultview.item(curItem, 'values')[4]            
            toast = ToastNotificationX(
                image=imgpath,            
                duration=3000,
            )
            toast.show_toast()

    def delete_image(self):

        curItem = self.get_current_selection()
        if curItem:            
            imgpath = self.resultview.item(curItem, 'values')[4]
            
            confirmed = True
            if not self.master.attributes('-fullscreen'):
                confirmed = Messagebox.okcancel(imgpath,'确定删除该图片?') == 'OK'

            if confirmed:
                self.resultview.delete(curItem)
                # send2trash(imgpath) # raize OSError
                shutil.move (imgpath, imgpath+'.tmp') # don't really delete

    def predict(self, method = 'FRCNN'):
        '''
        method : 'FRCNN', 'SSD', 'YOLO5'
        '''
        wapp = ttk.Toplevel(title=method, size=(600,300))
        wframe = WaitingFrame(wapp)
        wframe.task_to_run = lambda: self.predict_fundus_folder_thread(method, wframe.message.set)
            
    def predict_fundus_folder_thread(self, method, callback):

        predict_fundus_folder(self.path_var.get(), 
             method = method, dir_output = 'inplace', callback = callback)
        WaitingFrame.threadqueue.task_done()

    def upload_image(self):

        curItem = self.get_current_selection()
        if curItem:          
            imgpath = self.resultview.item(curItem, 'values')[4]
            self.send_data_to_server(imgpath, {'operator':'zys','timestamp': datetime.now()})
        
    def send_data_to_server(self, image_path, metadata):
            
        image_filename = os.path.basename(image_path)    
        multipart_form_data = {
            'image': (image_filename, open(image_path, 'rb')),
            'metadata': ('', str(metadata)),
        }

        if self.master.attributes('-fullscreen'):
            toast = ToastNotification(
                title='Upload to Server',
                message=str(multipart_form_data), 
                duration=1000,
            )
            toast.show_toast()
        else:
            Messagebox.show_info(str(multipart_form_data), 'Upload to Server')
        
        return    
        response = requests.post('http://www.example.com/api/v1/sensor_data/',
                                files=multipart_form_data)    
        print(response.status_code)

    def on_browse(self):
        """Callback for directory browse"""
        path = askdirectory(title="Browse directory")
        if path:
            self.path_var.set(path)

    def on_search(self):
        """Search for a term based on the search type"""
        search_term = self.term_var.get()
        search_path = self.path_var.get()
        search_type = self.type_var.get()
        
        if search_term == '':
            return

        self.clear_results_view()
        
        # start search in another thread to prevent UI from locking
        Thread(
            target=FileSearchFrame.file_search, 
            args=(search_term, search_path, search_type), 
            daemon=True
        ).start()

        if USE_PROGRESSBAR:
            self.progressbar.start(10)
        
        iid = self.resultview.insert(
            parent='', 
            index=END, 
        )
        self.resultview.item(iid, open=True)
        self.after(100, lambda: self.check_queue(iid))

    def check_queue(self, iid):
        """Check file queue and print results if not empty"""
        if all([
            FileSearchFrame.searching, 
            not FileSearchFrame.queue.empty()
        ]):
            filename = FileSearchFrame.queue.get()
            self.insert_row(filename, iid)
            self.update_idletasks()
            self.after(100, lambda: self.check_queue(iid))
        elif all([
            not FileSearchFrame.searching,
            not FileSearchFrame.queue.empty()
        ]):
            while not FileSearchFrame.queue.empty():
                filename = FileSearchFrame.queue.get()
                self.insert_row(filename, iid)
            self.update_idletasks()
            if USE_PROGRESSBAR:
                self.progressbar.stop()
        elif all([
            FileSearchFrame.searching,
            FileSearchFrame.queue.empty()
        ]):
            self.after(100, lambda: self.check_queue(iid))
        else:
            if USE_PROGRESSBAR:
                self.progressbar.stop()

    def insert_row(self, file, iid):
        """Insert new row in tree search results"""
        try:
            _stats = file.stat()
            _name = file.stem
            _timestamp = datetime.fromtimestamp(_stats.st_mtime)
            _modified = _timestamp.strftime(r'%m/%d/%Y %I:%M:%S%p')
            _type = file.suffix.lower()
            _size = FileSearchFrame.convert_size(_stats.st_size)
            _path = file.as_posix()
            iid = self.resultview.insert(
                parent='', 
                index=END, 
                values=(_name, _modified, _type, _size, _path)
            )
            self.resultview.selection_set(iid)
            self.resultview.see(iid)
        except OSError:
            return

    @staticmethod
    def file_search(term, search_path, search_type):
        """Recursively search directory for matching files"""
        FileSearchFrame.set_searching(1)
        if search_type == 'contains':
            FileSearchFrame.find_contains(term, search_path)
        elif search_type == 'startswith':
            FileSearchFrame.find_startswith(term, search_path)
        elif search_type == 'endswith':
            FileSearchFrame.find_endswith(term, search_path)

    @staticmethod
    def find_contains(term, search_path):
        """Find all files that contain the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if term in file:
                        record = pathlib.Path(path) / file
                        FileSearchFrame.queue.put(record)
        FileSearchFrame.set_searching(False)

    @staticmethod
    def find_startswith(term, search_path):
        """Find all files that start with the search term"""
        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    if file.startswith(term):
                        record = pathlib.Path(path) / file
                        FileSearchFrame.queue.put(record)
        FileSearchFrame.set_searching(False)

    @staticmethod
    def find_endswith(s, search_path):
        '''
        Find all files that end with the search terms    

        s : file extensions. separated by comma. e.g., '.jpg,.png,.gif'   
        '''
        terms = s.split(',')

        for path, _, files in pathlib.os.walk(search_path):
            if files:
                for file in files:
                    for term in terms:
                        if file.endswith(term):
                            record = pathlib.Path(path) / file
                            FileSearchFrame.queue.put(record)

        FileSearchFrame.set_searching(False)

    @staticmethod
    def set_searching(state=False):
        """Set searching status"""
        FileSearchFrame.searching = state

    @staticmethod
    def convert_size(size):
        """Convert bytes to mb or kb depending on scale"""
        kb = size // 1000
        mb = round(kb / 1000, 1)
        if kb > 1000:
            return f'{mb:,.1f} MB'
        else:
            return f'{kb:,d} KB'        


if __name__ == '__main__':
  
    app = ttk.Window("ODN") # , "cyborg")
    FileSearchFrame(app, None)
    app.geometry(DEFAULT_WINSIZE)
    app.attributes('-fullscreen', True)
    app.mainloop()