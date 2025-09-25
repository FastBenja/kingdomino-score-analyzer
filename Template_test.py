import cv2
import numpy as np

images = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/3.jpg")

# Crown template
temp_up = cv2.imread("Croped_crown.jpg")
temp_down = cv2.imread("Croped_crown_down.jpg")
temp_left = cv2.imread("Croped_crown_left.jpg")
temp_right = cv2.imread("yellow_test_crown.jpg")

# Template matching
#-----------------------------------------------------------------------------------------------
matched_up = cv2.matchTemplate(images, temp_up, cv2.TM_CCOEFF_NORMED)
output_up = cv2.normalize(matched_up, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
output_up_uint8 = output_up.astype("uint8")

matched_down = cv2.matchTemplate(images, temp_down, cv2.TM_CCOEFF_NORMED)
output_down = cv2.normalize(matched_down, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
output_down_uint8 = output_down.astype("uint8")

matched_left = cv2.matchTemplate(images, temp_left, cv2.TM_CCOEFF_NORMED)
output_left = cv2.normalize(matched_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
output_left_uint8 = output_left.astype("uint8")

matched_right = cv2.matchTemplate(images, temp_right, cv2.TM_CCOEFF_NORMED)
output_right = cv2.normalize(matched_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
output_right_uint8 = output_right.astype("uint8")
#-----------------------------------------------------------------------------------------------
# Get variables after template matching
mainValUp, maxValUp, minLocUp, maxLocUp = cv2.minMaxLoc(matched_up)
mainValDown, maxValDown, minLocDown, maxLocDown = cv2.minMaxLoc(matched_down)
mainValL, maxValL, minLocL, maxLocL = cv2.minMaxLoc(matched_left)
minValR, maxValR, minLocR, maxLocR = cv2.minMaxLoc(matched_right)
#-----------------------------------------------------------------------------------------------
# Thresholding if template match is high enough
#-----------------------------------------------------------------------------------------------
if maxValUp > 0.75: 
    ret, threshold_up = cv2.threshold(output_up_uint8,220,255,cv2.THRESH_BINARY)
    cv2.imshow("crown up", threshold_up)
    # bounding boxes
    contours, _ = cv2.findContours(threshold_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
else:
    print("no crown pointing up")

if maxValDown > 0.75:    
    ret, threshold_down = cv2.threshold(output_down_uint8,220,255,cv2.THRESH_BINARY)
    cv2.imshow("crown down", threshold_down)
    # bounding boxes
    contours, _ = cv2.findContours(threshold_down, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
else:
    print("no crown pointing down")

if maxValL > 0.75:
    ret, threshold_left = cv2.threshold(output_left_uint8,220,255,cv2.THRESH_BINARY)
    cv2.imshow("crown left", threshold_left)
    # bounding boxes
    contours, _ = cv2.findContours(threshold_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
else:
    print("no crown pointing left")

if maxValR > 0.75:    
    ret, threshold_right = cv2.threshold(output_right_uint8,220,255,cv2.THRESH_BINARY)
    cv2.imshow("crown right", threshold_right)
    # bounding boxes
    contours, _ = cv2.findContours(threshold_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
else:
    print("no crown pointing right")

cv2.imshow("test", images)
cv2.waitKey()
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------