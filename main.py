import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

images = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/24.jpg")

def rot_template_match(template_path):
    # Crown template
    temp = cv2.imread(template_path)
    
    #rotating template
    rotate1 = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
    rotate2 = cv2.rotate(rotate1, cv2.ROTATE_90_CLOCKWISE)
    rotate3 = cv2.rotate(rotate2, cv2.ROTATE_90_CLOCKWISE)


    # Template matching
    #-----------------------------------------------------------------------------------------------
    matched = cv2.matchTemplate(images, temp, cv2.TM_CCOEFF_NORMED)
    output = cv2.normalize(matched, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    output_uint8 = output.astype("uint8")

    matched1 = cv2.matchTemplate(images, rotate1, cv2.TM_CCOEFF_NORMED)
    output1 = cv2.normalize(matched1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    output1_uint8 = output1.astype("uint8")

    matched2 = cv2.matchTemplate(images, rotate2, cv2.TM_CCOEFF_NORMED)
    output2 = cv2.normalize(matched2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    output2_uint8 = output2.astype("uint8")

    matched3 = cv2.matchTemplate(images, rotate3, cv2.TM_CCOEFF_NORMED)
    output3 = cv2.normalize(matched3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    output3_uint8 = output3.astype("uint8")
    #-----------------------------------------------------------------------------------------------
    # Get variables after template matching
    mainValUp, maxValUp, minLocUp, maxLocUp = cv2.minMaxLoc(matched)
    mainValDown, maxValDown, minLocDown, maxLocDown = cv2.minMaxLoc(matched1)
    mainValL, maxValL, minLocL, maxLocL = cv2.minMaxLoc(matched2)
    minValR, maxValR, minLocR, maxLocR = cv2.minMaxLoc(matched3)
    #-----------------------------------------------------------------------------------------------
    # Thresholding if template match is high enough
    #-----------------------------------------------------------------------------------------------
    box = []

    if maxValUp > 0.75: 
        ret, threshold_up = cv2.threshold(output_uint8,220,255,cv2.THRESH_BINARY)
        cv2.imshow("crown up", threshold_up)
        # bounding boxes
        contours, _ = cv2.findContours(threshold_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw boxes
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
            box.append(x)
    else:
        print("no crown pointing up")

    if maxValDown > 0.75:    
        ret, threshold_down = cv2.threshold(output1_uint8,220,255,cv2.THRESH_BINARY)
        cv2.imshow("crown down", threshold_down)
        # bounding boxes
        contours, _ = cv2.findContours(threshold_down, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw boxes
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
            box.append(x)
    else:
        print("no crown pointing down")

    if maxValL > 0.75:
        ret, threshold_left = cv2.threshold(output2_uint8,220,255,cv2.THRESH_BINARY)
        cv2.imshow("crown left", threshold_left)
        # bounding boxes
        contours, _ = cv2.findContours(threshold_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw boxes
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
            box.append(x)
            
    else:
        print("no crown pointing left")

    if maxValR > 0.75:    
        ret, threshold_right = cv2.threshold(output3_uint8,220,255,cv2.THRESH_BINARY)
        cv2.imshow("crown right", threshold_right)
        # bounding boxes
        contours, _ = cv2.findContours(threshold_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw boxes
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
            box.append(x)
    else:
        print("no crown pointing right")

    box_count = len(box)

    print("number of bounding boxes:")
    print(box_count)
    cv2.imshow("test", images)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #-----------------------------------------------------------------------------------------------

rot_template_match("Templates/Yellow_crown.jpg")

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

