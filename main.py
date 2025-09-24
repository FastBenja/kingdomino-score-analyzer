import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

paths = filedialog.askopenfilenames()

images = [None] * len(paths)

for num, path in enumerate(paths):
    print(num)
    images[num] = cv2.imread(path)