import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
from tkinter import StringVar
import cv2
from PIL import Image, ImageTk
import time
from image_detection import yolonet

class Surface(ttk.Frame):
    pic_path = ""
    vid_path = ""
    viewhigh = 1000
    viewwide = 1200
    screen_pos_x = 100
    screen_pos_y = 20
    update_time = 0
    thread = None
    thread_run = False
		
    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)

        win.title("PlateRecognition")
        screen_resolution = '1440x860' + '+' + str(self.screen_pos_x) + '+' + str(self.screen_pos_y)
        win.geometry(screen_resolution)

        # Load model
        self.net = yolonet('models/yolov5s.onnx', 0.5, 0.5, 0.5)
        
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side="left",expand=1,fill="both")
        frame_right1.pack(side="top",expand=1,fill=tk.Y)
        frame_right2.pack(side="right",expand=0)
        ttk.Label(frame_left, text='src image:').pack(anchor="nw") 
        # ttk.Label(frame_right1, text='license plate position:').grid(column=0, row=0, sticky=tk.W)
        ttk.Label(frame_right2, text='choose model:').pack(anchor="se") 
        
        data = ["yolov5s", "yolov5l"]
        self.value = StringVar()
        self.value.set("yolov5s")
        self.model_ctl = ttk.Combobox(frame_right2, width=17, height=8, state='normal', textvariable=self.value)
        self.model_ctl.bind("<<ComboboxSelected>>", self.SetModel)
        self.model_ctl["values"] = data
        self.model_ctl.current(0)
        self.model_ctl.pack()

        from_pic_ctl = ttk.Button(frame_right2, text="load image", width=20, command=self.from_pic)
        from_vid_ctl = ttk.Button(frame_right2, text="load video", width=20, command=self.from_video)
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        # start_recog_ctl = ttk.Button(frame_right2, text="start", width=20, command=self.start_recognition)
        
        # self.roi_ctl = ttk.Label(frame_right1)
        # self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        # ttk.Label(frame_right1, text='result:').grid(column=0, row=2, sticky=tk.W)
        # self.r_ctl = ttk.Label(frame_right1, text="")
        # self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        # self.color_ctl = ttk.Label(frame_right1, text="12321", width="20")
        # self.color_ctl.grid(column=0, row=4, sticky=tk.W)

        from_pic_ctl.pack(anchor="se", pady="5")
        from_vid_ctl.pack(anchor="se", pady="5")
        # start_recog_ctl.pack(anchor="se", pady="5")



    def start_recognition(self):
        srcimg = cv2.imread(self.pic_path)
        high, wide, _ = srcimg.shape
        # print(srcimg.shape)
        if wide > 2000 or high > 2000:
            self.net.change_text_size(2)
        else:
            self.net.change_text_size(1)
        roi = self.net.detect(srcimg)
        return roi
        

    def SetModel(self, event):
        model = self.value.get()
        # print('选中的数据:{}'.format(self.model_ctl.get()))
        # print('value的值:{}'.format(self.value.get()))
        # print(model)
        if model == "yolov5s":
            self.net = yolonet('models/yolov5s.onnx', 0.5, 0.5, 0.5)
            print("select model: yolov5s")
        elif model == "yolov5l":
            self.net = yolonet('models/yolov5l.onnx', 0.5, 0.5, 0.5)
            print("select model: yolov5l")

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            
            wide = int(wide * factor)
            if wide <= 0 : wide = 1
            high = int(high * factor)
            if high <= 0 : high = 1
            im=im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk
    
    def show_roi(self, roi):
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = Image.fromarray(roi)
        imgtk = ImageTk.PhotoImage(image=roi)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            
            wide = int(wide * factor)
            if wide <= 0 : wide = 1
            high = int(high * factor)
            if high <= 0 : high = 1
            roi = roi.resize((wide, high), Image.ANTIALIAS)
        self.imgtk_roi = ImageTk.PhotoImage(image=roi)
        self.image_ctl.configure(image=self.imgtk_roi, state='enable')
        self.update_time = time.time()
        
    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="choose the image", filetypes=[("jpg", "*.jpg"), ('png','*.png')]) # ('All files','*.*'), 
        # if self.pic_path:
        #     # r, roi = PlateRecognition.PlateRecognition(self.pic_path, False, self.model, self.modelType)
        #     img_bgr = cv2.imread(self.pic_path)
        #     self.imgtk = self.get_imgtk(img_bgr)
        #     self.image_ctl.configure(image=self.imgtk)
            # self.show_roi(r, roi)
        roi = self.start_recognition()
        self.show_roi(roi)
        # print("done")

    def from_video(self):
        self.thread_run = False
        self.vid_path = askopenfilename(title="choose the image", filetypes=[('mp4','*.mp4')])
        if self.vid_path:
            self.video = cv2.VideoCapture(self.vid_path)
            ok, frame = self.video.read()
            if ok:
                # print(frame.shape)
                high, wide, _ = frame.shape
                if wide > 2000 or high > 2000:
                    self.net.change_text_size(2)
                else:
                    self.net.change_text_size(1)
                if wide > self.viewwide or high > self.viewhigh:
                    wide_factor = self.viewwide / wide
                    high_factor = self.viewhigh / high
                    factor = min(wide_factor, high_factor)
                    
                    wide = int(wide * factor)
                    if wide <= 0 : wide = 1
                    high = int(high * factor)
                    if high <= 0 : high = 1
            self.video_loop(wide, high)

    def video_loop(self, wide, high):
        ok, frame = self.video.read()
        if ok: # frame captured without any errors
            roi = self.net.detect(frame)
            cv2image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGBA)
            cv2image = cv2.resize(cv2image, (wide, high), interpolation=cv2.INTER_LANCZOS4)
            self.current_image = Image.fromarray(cv2image)
            self.video_roi = ImageTk.PhotoImage(image=self.current_image)
            self.image_ctl.config(image=self.video_roi)
            self.after(1, self.video_loop, wide, high)


def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win=tk.Tk()
    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()