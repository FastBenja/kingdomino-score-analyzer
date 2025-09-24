import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

W_size = 5
H_size = 5

root = tk.Tk()
root.withdraw()

paths = filedialog.askopenfilenames()

images = [None] * len(paths)

for num, path in enumerate(paths):
    print(num)
    images[num] = cv2.imread(path)
    

tiles = [[None]*H_size]*W_size

images[0] = np.array(images[0])

height, width, channels  = images[0].shape



for  ih in range(H_size):
    for iw in range(W_size):
        x = width/W_size * iw 
        y = height/H_size * ih
        h = (height / H_size)
        w = (width / W_size )
        print(x,y,h,w)
        tiles[iw][ih] = images[0][int(y):int(y+h), int(x):int(x+w)]
        cv2.imshow(f"images[{iw},{ih}]",tiles[iw][ih])
        
       




cv2.imshow("dkdk",images[0])
cv2.waitKey() 
    


