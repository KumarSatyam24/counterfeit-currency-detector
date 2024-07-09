import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

class ImageCropper:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.tk_img = ImageTk.PhotoImage(self.img)
        self.canvas = Canvas(root, width=self.img.width, height=self.img.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_img)
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.crop_rect_id = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.root.mainloop()

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if not self.crop_rect_id:
            self.crop_rect_id = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y, outline="red"
            )

    def on_mouse_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.crop_rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        self.crop(event.x, event.y)

    def crop(self, end_x, end_y):
        if self.rect:
            self.canvas.delete(self.rect)
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        self.rect = (x1, y1, x2, y2)
        cropped_img = self.img.crop(self.rect)
        cropped_img.save("cropped_image.jpg")
        self.root.destroy()

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
    cap.release()

def delete_image(image_path):
    import os
    if os.path.exists(image_path):
        os.remove(image_path)

if __name__ == "__main__":
    # Step 1: Capture image from webcam
    capture_image()

    # Step 2: Create GUI for cropping
    root = Tk()
    cropper = ImageCropper(root, "captured_image.jpg")

    # Step 3: Delete the captured image after cropping
    delete_image("captured_image.jpg")
