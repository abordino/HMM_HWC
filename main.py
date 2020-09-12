import logging
import os
import tkinter as tk
from tkinter import *
from tkinter import messagebox

import PIL
from PIL import Image
from PIL import ImageDraw
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from model import digits_train

model = digits_train.DigitsHmm({})
path = os.path.realpath('HMM_HWC')

root = tk.Tk()
root.title("HMM digits classifier")
width_canvas = 512
height_canvas = 512
img = None
pic = PIL.Image.new('RGB', (width_canvas, height_canvas), 'black')
draw_img = ImageDraw.Draw(pic)


def draw(e):
    x, y = e.x, e.y

    if canvas.old_coords:
        x_old, y_old = canvas.old_coords
        canvas.create_line(x, y, x_old, y_old, fill="white", width=7)
        draw_img.line((x, y, x_old, y_old), fill="white", width=35)
    canvas.old_coords = x, y


def get_image():
    global img
    filename = 'image.png'
    pic.save(path + filename)

    img = image.load_img(path + filename, target_size=(28, 28), color_mode="grayscale")
    y = np.asarray(img)
    x = []
    for d in range(len(y)):
        for j in range(len(y[d])):
            x.append(y[d][j])

    for i in range(len(x)):
        x[i] = int(x[i] > 125)

    data = x
    data_processed = []
    for j in range(112):
        value = 0
        for h in range(7):
            value += (2 ** h) * data[7 * j + h]
        data_processed.append(value)

    plt.imshow(y)
    plt.show()

    return data_processed


def reset_coords():
    canvas.old_coords = None


def clear_canvas():
    global pic, draw_img
    canvas.delete("all")
    del draw_img
    pic = PIL.Image.new('RGB', (width_canvas, height_canvas), 'black')
    draw_img = ImageDraw.Draw(pic)


def train_model_gui():
    global model
    try:
        model = model.train_model()
        messagebox.showinfo("Training", "Model trained")
    except Exception as e:
        logging.exception("TRAIN GUI", e)


def load_model_gui():
    global model
    if model.zinga == {}:
        try:
            model.load_digit_model()
            messagebox.showinfo("Loading", "Model loaded")
        except Exception as e:
            messagebox.showinfo("Loading", "Error in loading model")
            logging.exception("LOAD GUI", e)
    else:
        messagebox.showinfo("Prediction", "Model already loaded")


def predict_gui():
    global model, img
    x = get_image()

    if model.zinga == {}:
        messagebox.showinfo("Prediction", "No model loaded")
    else:
        try:
            prediction = model.model_predict(x)
            messagebox.showinfo("Prediction", "You drew a " + str(prediction))
        except Exception as e:
            logging.exception("PREDICT GUI", e)


def test_model_gui():
    global model
    if model.zinga == {}:
        messagebox.showinfo("Testing", "No model loaded")
    else:
        try:
            model.test_model()
            messagebox.showinfo("Testing", "Results in cli")
        except Exception as e:
            logging.exception("TEST GUI", e)


# canvas
canvas = tk.Canvas(root, width=width_canvas, height=height_canvas, background='black')
canvas.pack()
canvas.old_coords = None

# widget button
root.bind('<B1-Motion>', draw)
root.bind('<ButtonRelease-1>', reset_coords)

Btn_clear = Button(text="Clear", bg="orange",
                   width=69, height=4, command=clear_canvas)
Btn_clear.pack()

Btn_train = Button(text="Train Model", bg="orange",
                   width=69, height=4, command=train_model_gui)
Btn_train.pack()

Btn_load = Button(text="Load Model", bg="orange",
                  width=69, height=4, command=load_model_gui)
Btn_load.pack()

Btn_predict = Button(text="Predict", bg="orange",
                     width=69, height=4, command=predict_gui)
Btn_predict.pack()

Btn_predict = Button(text="Accuracy model", bg="orange",
                     width=69, height=4, command=test_model_gui)
Btn_predict.pack()

root.mainloop()
