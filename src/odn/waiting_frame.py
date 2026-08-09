# a slightly different copy of pi.gui.waiting_frame.py

import textwrap
from time import sleep
from queue import Queue
from random import randint
from threading import Thread
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox

class WaitingFrame(ttk.Frame):

    threadqueue = Queue()

    def __init__(self, master, task_to_run = None):
        super().__init__(master, padding=5, bootstyle=INFO)
        self.pack(fill=BOTH, expand=YES)
        self.message = ttk.StringVar(value='')
        self.create_elements()

        if task_to_run is None:
            task_to_run = self.simulate_io_task

        self.task_to_run = task_to_run

    def create_elements(self):
        """Create the layout elements."""
        container = ttk.Frame(self, padding=10)
        container.pack(fill=BOTH, expand=YES)
        
        _text = ("Click the START button to begin a long-running task. \n根据数据量的多少，该过程可能持续很久，请耐心等待完成。")
        wrapped = '\n'.join(textwrap.wrap(_text, width=35))
        lbl = ttk.Label(container, text=wrapped, justify=LEFT)
        lbl.pack(fill=X, pady=10, expand=YES)

        self.start_btn = ttk.Button(
            master=container, 
            text='START | 开始',
            command=self.start_task
        )
        self.start_btn.pack(fill=X, pady=10)
        self.progressbar = ttk.Progressbar(
            master=container, 
            maximum=10, 
            value=0,
            bootstyle=SUCCESS,
            mode=INDETERMINATE
        )
        # print(self.progressbar['value']) # = 0
        self.progressbar.pack(fill=X, expand=YES)
        msg_lbl = ttk.Label(container, textvariable=self.message, anchor=CENTER)
        msg_lbl.pack(fill=X, pady=10)

    def start_task(self):
        """Start the progressbar and run the task in another thread"""
        self.progressbar.start()
        self.start_btn.configure(state=DISABLED)
        
        thread = Thread(
            target=self.task_to_run, 
            # args=[i], 
            daemon=True
        )
        WaitingFrame.threadqueue.put(thread.start())
        
        self.listen_for_complete_task()

    def listen_for_complete_task(self):
        """Check to see if task is complete; if so, stop the 
        progressbar and show and alert
        """

        # print(WaitingFrame.threadqueue)
        if WaitingFrame.threadqueue.unfinished_tasks == 0:
            self.progressbar.stop()
            Messagebox.ok(title='alert', message="process complete")
            self.start_btn.configure(state=NORMAL)
            self.message.set('')
            return
        self.after(500, self.listen_for_complete_task)

    def simulate_io_task(self):
        """Simulate an IO operation to run for a random interval 
        between 1 and 15 seconds.
        """
        seconds_to_run = randint(1, 15)
        sleep(seconds_to_run)
        WaitingFrame.threadqueue.task_done()
        self.message.set('Finished task on Thread.')

if __name__ == '__main__':

    app = ttk.Window(title="task", themename="lumen")
    WaitingFrame(app, task_to_run=None)
    app.mainloop()