import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

W_size = 5
H_size = 5

# comment kode 
# root = tk.Tk()
# root.withdraw()

# paths = filedialog.askopenfilenames()

# images = [None] * len(paths)

# for num, path in enumerate(paths):
#     print(num)
#     images[num] = cv2.imread(path)
    
    
# test_images = images[0] = np.array(images[0])    



tiles = np.array([[None for _ in range(H_size)] for _ in range(W_size)])




tiles_x = 1
tiles_y = 3

test_images = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg")

height, width, channels  = test_images.shape


# spilte images 5 tilles 
for  ih in range(H_size):
    for iw in range(W_size):
        x = width/W_size * iw 
        y = height/H_size * ih
        h = (height / H_size)
        w = (width / W_size )
        tiles[ih][iw] = test_images[int(y):int(y+h), int(x):int(x+w)]
        
        
        

def greencheck(tile):
    green_tresholde = 3000
    upper_green = np.array([44,255,255])
    lower_green = np.array([39,0,0]) 
    
    lower_redhouse = np.array([0,0,0])
    high_redhouse = np.array([17,255,255])
    

    
    hsv_img = cv2.cvtColor(tile,cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv_img,lower_green,upper_green)
    mask_redhouse = cv2.inRange(hsv_img,lower_redhouse,high_redhouse)
    
    
    cv2.imshow(f"images[{tile}]",mask_green)
  
    value = (mask_green.sum()+mask_redhouse.sum())/255
    print(value)
    if(value>green_tresholde):
        return "green"
    else:
        return "not"
    

def colorcheck(tile,
               upper,
               lower,
               tresholde,
               Name,
               ekstra = None,
               upper2 = None,
               lower2 = None):
    hsv_img = cv2.cvtColor(tile,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img,lower,upper)
    mask2 = 0
    if (ekstra == True):
        mask2 = cv2.inRange(hsv_img,lower2,upper2)
        mask2 = mask2.sum()
          
    #cv2.imshow(f"images[{tile}]",mask_green)
  
    value = (mask.sum()+mask2)/255
    print(f"color value: {value} og tresholde: {tresholde} og {Name}")
    if(value>tresholde):
        return Name
    else:
        return "not"
    

    

# check kode color
def nothing(x):
    pass

def create_trackbars():
    cv2.namedWindow('HSV Adjust')
    cv2.createTrackbar('H Low', 'HSV Adjust', 0, 179, nothing)
    cv2.createTrackbar('H High', 'HSV Adjust', 179, 179, nothing)
    cv2.createTrackbar('S Low', 'HSV Adjust', 0, 255, nothing)
    cv2.createTrackbar('S High', 'HSV Adjust', 255, 255, nothing)
    cv2.createTrackbar('V Low', 'HSV Adjust', 0, 255, nothing)
    cv2.createTrackbar('V High', 'HSV Adjust', 255, 255, nothing)

def get_hsv_values_from_trackbars():
    h_low = cv2.getTrackbarPos('H Low', 'HSV Adjust')
    h_high = cv2.getTrackbarPos('H High', 'HSV Adjust')
    s_low = cv2.getTrackbarPos('S Low', 'HSV Adjust')
    s_high = cv2.getTrackbarPos('S High', 'HSV Adjust')
    v_low = cv2.getTrackbarPos('V Low', 'HSV Adjust')
    v_high = cv2.getTrackbarPos('V High', 'HSV Adjust')
    
    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    return lower, upper


def controlcolor():
# Brug eksempel med dit billede
    create_trackbars()

    while True:
        lower, upper = get_hsv_values_from_trackbars()
        
        # Anvend på dit tile
        hsv_tile = cv2.cvtColor(tiles[tiles_x][tiles_y], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_tile, lower, upper)
        result = cv2.bitwise_and(tiles[tiles_x][tiles_y], tiles[tiles_x][tiles_y], mask=mask)
        
        # Vis resultater
        cv2.imshow('Original', tiles[tiles_x][tiles_y])
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)
        
        print(mask.sum()/250)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    

#colorcheck()

cv2.imshow(f"images[{0},{0}] no color",tiles[tiles_x][tiles_y])

print(greencheck(tiles[tiles_x][tiles_y]))

print("her")
tresholde_green = 3000
upper_green = np.array([44,255,255])
lower_green = np.array([39,0,0]) 

lower_redhouse = np.array([0,0,0])
upper_redhouse = np.array([17,255,255])
green = "green"
greencount = 0

treshold_forest = 2300
upper_forest = np.array([62,255,255])
lower_forest = np.array([44,0,0]) 
lower_brunhouse = np.array([0,0,0])
upper_brunehouse = np.array([17,255,255])
forest = "forest"
forestcount = 0

colorcheck(tiles[tiles_x][tiles_y],
                        upper_forest,
                        lower_forest,
                        treshold_forest,
                        forest,
                        True,
                        upper_brunehouse,
                        lower_brunhouse)

#loop every tile
for x in range(tiles.shape[0]):
    for y in range(tiles.shape[1]):
        # check terrain  
        if(colorcheck(tiles[x][y],
                           upper_green,
                           lower_green,
                           tresholde_green,
                           green,
                           True,
                           upper_redhouse,
                           lower_redhouse) == "green"):
            print(f"green  ({x,y})")
            greencount = 1 + greencount
        elif(colorcheck(tiles[x][y],
                        upper_forest,
                        lower_forest,
                        treshold_forest,
                        forest,
                        True,
                        upper_brunehouse,
                        lower_brunhouse) == "forest"):
            print(f"forest  {x,y}")
        
            forestcount = forestcount + 1
            
        cv2.imshow(f"firekant [{x},{y}] + {greencheck(tiles[x][y])}",tiles[x][y])

#controlcolor()
        
print(f"green count: {greencount} + forest count: {forestcount}")

cv2.imshow("dkdk",test_images)
cv2.waitKey() 
    


