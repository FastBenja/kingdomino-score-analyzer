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

tiles_x = 0
tiles_y = 0

tiles_x_2 = 0
tiles_y_2 = 2

test_images = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/2.jpg")

height, width, channels  = test_images.shape


# spilte images 5 tilles 
for  ih in range(H_size):
    for iw in range(W_size):
        x = width/W_size * iw 
        y = height/H_size * ih
        h = (height / H_size)
        w = (width / W_size )
        tiles[ih][iw] = test_images[int(y):int(y+h), int(x):int(x+w)]
        
     

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
    cv2.createTrackbar('H Low', 'HSV Adjust', 0, 255, nothing)
    cv2.createTrackbar('H High', 'HSV Adjust', 255, 255, nothing)
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
        
        hsv_tile2 = cv2.cvtColor(tiles[tiles_x_2][tiles_y_2], cv2.COLOR_BGR2HSV)
        mask_2 = cv2.inRange(hsv_tile2, lower, upper)
        result_2 = cv2.bitwise_and(tiles[tiles_x_2][tiles_y_2], tiles[tiles_x_2][tiles_y_2], mask=mask)
        
        
        # Vis resultater
        cv2.imshow('Original', tiles[tiles_x][tiles_y])
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)
        
        cv2.imshow('Original 2', tiles[tiles_x_2][tiles_y_2])
        cv2.imshow('Mask 2', mask_2)
        cv2.imshow('Result 2', result_2)
        
        print(mask.sum()/250, mask_2.sum()/250)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    

#colorcheck()

cv2.imshow(f"images[{0},{0}] no color",tiles[tiles_x][tiles_y])


#da
print("her")
tresholde_green = 5000
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

treshold_blue = 3000
lower_blue = np.array([58,0,0])
upper_blue = np.array([175,255,255])
blue = "blue"
bluecount = 0

treshold_brune = 7000
lower_brune = np.array([0,0,0])
upper_brune= np.array([26,255,149])
brune = "brune"
brunecount = 0

treshold_yellow = 5500
lower_yellow = np.array([19,22,108])
upper_yellow= np.array([29,255,213])
yellow = "yellow"
yellowcount = 0

no_color_count = 0

print( colorcheck(tiles[tiles_x][tiles_y],
                        upper_yellow,
                        lower_yellow,
                        treshold_yellow,
                        yellow,
                        False))
                    

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
            #print(f"green  ({x,y})")
            
            greencount = 1 + greencount
            cv2.imshow(f"green {greencount}", tiles[x][y])
        elif(colorcheck(tiles[x][y],
                        upper_forest,
                        lower_forest,
                        treshold_forest,
                        forest,
                        True,
                        upper_brunehouse,
                        lower_brunhouse) == "forest"):
            #print(f"forest  {x,y}")
        
            forestcount = forestcount + 1
            cv2.imshow(f"forest {forestcount}", tiles[x][y])
            
        elif(colorcheck(tiles[x][y],
                        upper_blue,
                        lower_blue,
                        treshold_blue,
                        blue,
                        False) == "blue"):
            #print(f"blue {x,y}")
            bluecount = bluecount + 1
            cv2.imshow(f"blue {bluecount}", tiles[x][y])

             
        elif(colorcheck(tiles[x][y],
                        upper_brune,
                        lower_brune,
                        treshold_brune,
                        brune,
                        False) == "brune"):
            #print(f"brune {x,y}")
            brunecount = 1 + brunecount
            cv2.imshow(f"brune {brunecount}", tiles[x][y])

        elif(colorcheck(tiles[x][y],
                        upper_yellow,
                        lower_yellow,
                        treshold_yellow,
                        yellow,
                        False) == "yellow"):
            #print(f"yellow {x,y}")
            yellowcount = yellowcount + 1
        else:
            no_color_count = no_color_count + 1
            print(f"no color {x,y}") 
            
            
            
# controlcolor()
        
print(f"green count: {greencount} forest count: {forestcount}  blueconut: {bluecount}  brunecount: {brunecount} yellowcount: {yellowcount} no color: {no_color_count} ")
print(f"number of tiles: {greencount+forestcount+bluecount+brunecount+yellowcount+no_color_count}")
cv2.imshow("dkdk",test_images)
cv2.waitKey() 
    


