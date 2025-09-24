import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

images = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/2.jpg")

# Crown template
temp_up = cv2.imread("Croped_crown.jpg")
temp_down = cv2.imread("Croped_crown_down.jpg")
temp_left = cv2.imread("Croped_crown_left.jpg")
temp_right = cv2.imread("Croped_crown_right.jpg")

# Template matching
#-----------------------------------------------------------------------------------------------
matched_up = cv2.matchTemplate(images, temp_up, cv2.TM_CCOEFF_NORMED)
output_up = cv2.normalize(matched_up, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

matched_down = cv2.matchTemplate(images, temp_down, cv2.TM_CCOEFF_NORMED)
output_down = cv2.normalize(matched_down, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

matched_left = cv2.matchTemplate(images, temp_left, cv2.TM_CCOEFF_NORMED)
output_left = cv2.normalize(matched_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

matched_right = cv2.matchTemplate(images, temp_right, cv2.TM_CCOEFF_NORMED)
output_right = cv2.normalize(matched_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#-----------------------------------------------------------------------------------------------
# Get variables after template matching
mainValUp, maxValUp, minLocUp, maxLocUp = cv2.minMaxLoc(matched_up)
mainValDown, maxValDown, minLocDown, maxLocDown = cv2.minMaxLoc(matched_down)
mainValL, maxValL, minLocL, maxLocL = cv2.minMaxLoc(matched_left)
minValR, maxValR, minLocR, maxLocR = cv2.minMaxLoc(matched_right)
#-----------------------------------------------------------------------------------------------
# Thresholding if template match is high enough
#-----------------------------------------------------------------------------------------------
if maxValUp > 0.70: 
    ret, threshold_up = cv2.threshold(output_up,220,255,cv2.THRESH_BINARY)
    cv2.imshow("crown up", threshold_up)
else:
    print("no crown pointing up")

if maxValDown > 0.85:    
    ret, threshold_down = cv2.threshold(output_down,220,255,cv2.THRESH_BINARY)
    cv2.imshow("crown down", threshold_down)
else:
    print("no crown pointing down")

if maxValL > 0.85:
    ret, threshold_left = cv2.threshold(output_left,220,255,cv2.THRESH_BINARY)
    cv2.imshow("crown left", threshold_left)
else:
    print("no crown pointing left")

if maxValR > 0.70:    
    ret, threshold_right = cv2.threshold(output_right,220,255,cv2.THRESH_BINARY)
    cv2.imshow("crown right", threshold_right)
else:
    print("no crown pointing right")

cv2.waitKey()
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------


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


