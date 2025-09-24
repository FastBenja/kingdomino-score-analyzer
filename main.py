import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

images = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg")

# Crown template
temp_up = cv2.imread("Croped_crown.jpg")

# Template matching
matched_up = cv2.matchTemplate(images, temp_up, cv2.TM_CCOEFF_NORMED)
output = cv2.normalize(matched_up, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Thresholding
ret, threshold = cv2.threshold(output,220,255,cv2.THRESH_BINARY)

cv2.imshow("template matched", threshold)
cv2.waitKey()
cv2.destroyAllWindows()

# Patch-size
patch_w, patch_h = 100, 100

# list for patches
roi_zone = [[None for _ in range(5)] for _ in range(5)]

# Loop to crop the image into 25 patches
for y in range(5):
    for x in range(5):
        x_start = x * patch_w
        y_start = y * patch_h
        x_end = x_start + patch_w
        y_end = y_start + patch_h

        cropped_img = images[y_start:y_end, x_start:x_end]

        roi_zone[y][x] = cropped_img

        """ 
        #vis hver patch
        cv2.imshow(f"Patch {y},{x}", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
"""
for y in range(5):
    for x in range(5):

        # Template matching
        output_template = cv2.matchTemplate(roi_zone[x][y], temp, cv2.TM_SQDIFF_NORMED)

        # normalize
        #output = cv2.normalize(output_template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        cv2.imshow("lala", output_template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""
"""
cv2.imshow("patch 1", roi_zone[0][0])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


