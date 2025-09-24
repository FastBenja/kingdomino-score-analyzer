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

# Patch size
patch_w, patch_h = 100, 100  # Adjust as needed
 
# Counter for patch numbering
patch_id = 0

# Create empty np.array
#roi_zone = np.array((100,100,3))
roi_zone = []

# Convert image into grayscale
gray_img = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

# Loop through the image with step size = patch size
for y in range(0, img_h, patch_h):
    for x in range(0, img_w, patch_w):
 
        # Ensure patch does not exceed image boundaries
        x_end = min(x + patch_w, img_w)
        y_end = min(y + patch_h, img_h)
 
        # Crop the patch
        patch = images[y:y_end, x:x_end]

        roi_zone = np.append(roi_zone, patch)

        # Draw a rectangle on the original image (visualization)
        cv2.rectangle(images, (x, y), (x_end, y_end), (0, 255, 0), 2)
        patch_id += 1
        print(roi_zone[patch_id])
        print("patch id", patch_id)

# Show the original image with drawn patches
cv2.imshow("Patches", images)
cv2.waitKey(0)
cv2.destroyAllWindows()