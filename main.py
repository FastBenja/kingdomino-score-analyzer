import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

images = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg")



# Get image height and width from a numpy array
img_h, img_w, img_d = images.shape

#test to see if the image i loaded
print("height", images.shape)
cv2.imshow("image", images)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Patch-size
patch_w, patch_h = 100, 100

# liste til patches
roi_zone = []

# Loop to crop the image into 25 patches
for y in range(5):
    for x in range(5):
        x_start = x * patch_w
        y_start = y * patch_h
        x_end = x_start + patch_w
        y_end = y_start + patch_h

        cropped_img = images[y_start:y_end, x_start:x_end]

        roi_zone.append(cropped_img)

        """ 
        #vis hver patch
        cv2.imshow(f"Patch {y},{x}", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

cv2.imshow("patch 1", roi_zone[1])
cv2.waitKey(0)
cv2.destroyAllWindows()



