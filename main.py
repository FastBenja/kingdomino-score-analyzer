import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections import deque
import os
from matplotlib import pyplot as plt
import glob
from pathlib import Path
from crownFinder import crown_finder

def count(color,crown):
    # array holde value of crown,size,id 
    id_array = np.array([[ [0]*3 ]*5 ]*5)
    burn_queue = deque()
    id = 1
        
    for y, row in enumerate(crown):
        for x,colmen in enumerate(row):
            old_id = True
            #check for crown and no id
            if(crown[y][x] != 0 and id_array[y][x][2] == 0):
                # save value of crown,size and id
                id_array[y][x][0] = crown[y][x]
                id_array[y][x][1] += 1
                id_array[y][x][2] = id
               
                burn_queue = burn_queue + add_too_burn_queue(x,y,color[y][x],color,id_array)
                count = 0
                while(len(burn_queue)>0):
                    count += 1                 
                    burn = burn_queue.pop()
                    # save value of crown,size and id
                    if ( crown[burn[0]][burn[1]] != 0):
                         id_array[burn[0]][burn[1]][0] += crown[burn[0]][burn[1]] 
                    id_array[burn[0]][burn[1]][1] += 1
                    id_array[burn[0]][burn[1]][2] = id
                    burn_queue = burn_queue + add_too_burn_queue(burn[1],burn[0],color[burn[0]][burn[1]],color,id_array)
                id +=1  

    return  count_the_board(id_array,id-1)
                
   
def count_the_board(board,id):
    point = np.array([[0]*2]*id)
    
    #sort every blob 
    for y,row in enumerate(board):
        for x,felt in enumerate(row):
            if(felt[2] !=  0):
                point[felt[2]-1][0] += felt[0]
                point[felt[2]-1][1] += 1
                
    #count point 
    result = 0
    for x in range(point.shape[0]):
        result += point[x][0]*point[x][1]
    return result    
    
    
# how greas fire methond core                 
def add_too_burn_queue(x,y,color,array,id_array):
    new_queue = deque()
    up = y-1
    down = y+1
    left = x-1
    right = x+1
    # check sourounding blocks 
    if(up >= 0):        
        if(array[up][x] == color and id_array[up][x][2] == 0):
            new_queue.append(np.array([up,x]))
    if(left >= 0):
        if(array[y][left] == color and id_array[y][left][2] == 0):
            new_queue.append(np.array([y,left]))
    if(down < array.shape[1]):
        if(array[down][x] == color and id_array[down][x][2] == 0 ):
            new_queue.append(np.array([down,x]))
    if(right < array.shape[0]):
        if(array[y][right] == color and id_array[y][right][2] == 0):
            new_queue.append(np.array([y,right]))
    return new_queue


# print(count(color21,crown21))

biomes: dict[str, dict[str, int]] = {
    "forrest": {"H_lower": 31, "H_upper": 61,  "S_lower": 80,  "S_upper": 232, "V_lower": 35,  "V_upper": 81},
    "desert":  {"H_lower": 19, "H_upper": 27,  "S_lower": 108, "S_upper": 156, "V_lower": 38,  "V_upper": 176},
    "mine":    {"H_lower": 0,  "H_upper": 0,   "S_lower": 0,   "S_upper": 0,   "V_lower": 0,   "V_upper": 0},
    "water":   {"H_lower": 100,"H_upper": 110, "S_lower": 127, "S_upper": 255, "V_lower": 101, "V_upper": 255},
    "field":   {"H_lower": 22, "H_upper": 30,  "S_lower": 231, "S_upper": 255, "V_lower": 176, "V_upper": 218},
    "grass":   {"H_lower": 35, "H_upper": 51,  "S_lower": 104, "S_upper": 255, "V_lower": 136, "V_upper": 174},
}

class ImageScore:
    """
    Class to load images and evaluate them
    """
    
    def __init__(self):
        global biomes
        self.paths = [] # List to hold filepaths for all images
        self.image_dict = {} # Init dictionary to hold images and their filenames
        dataset_folder_path = "./King Domino dataset/Cropped and perspective corrected boards/"
        desert_tile_1_file_path = "./desert_sample_1.jpg"
        desert_tile_2_file_path = "./desert_sample_2.jpg"

        # Load all images in the dataset folder and store in dictionary
        for name in os.listdir(dataset_folder_path):
            self.paths.append(os.path.join(dataset_folder_path, name))        
        for dataset_folder_path in self.paths:
            path_segments = dataset_folder_path.split('/')
            file_name = path_segments[len(path_segments)-1]
            self.image_dict[file_name] = np.array(cv2.imread(dataset_folder_path))
        
        # Load tile and calculate histogram
        desert_tile_1_img = cv2.imread(desert_tile_1_file_path)
        desert_tile_2_img = cv2.imread(desert_tile_2_file_path)
        desert_tile_1_img = cv2.cvtColor(desert_tile_1_img, cv2.COLOR_BGR2HSV)
        desert_tile_2_img = cv2.cvtColor(desert_tile_1_img, cv2.COLOR_BGR2HSV)
        self.desert_hist_1 = self.__create_histogram(desert_tile_1_img)
        self.desert_hist_2 = self.__create_histogram(desert_tile_2_img)
        
    def __create_histogram(self, img):
        hist_list = []
        img = np.array(img)
        hist_list.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
        cv2.normalize(hist_list[0], hist_list[0], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_list.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
        cv2.normalize(hist_list[1], hist_list[1], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_list.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
        cv2.normalize(hist_list[2], hist_list[2], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist_list
    
    def __detectBiome(self, img):
        detected = np.zeros((5, 5),dtype=object)

        # Loop thrugh all cells on the board and test for biome type
        for i in range(5):
            for j in range(5):
                y1, y2 = i*100, (i+1)*100
                x1, x2 = j*100, (j+1)*100
                cell = img[y1:y2, x1:x2]
                
                # Calculate histograms for cell
                hist_cell = self.__create_histogram(cell)
                hist_h_cor = cv2.compareHist(hist_cell[0], self.desert_hist_1[0], cv2.HISTCMP_BHATTACHARYYA)
                hist_s_cor = cv2.compareHist(hist_cell[1], self.desert_hist_1[1], cv2.HISTCMP_BHATTACHARYYA)
                hist_v_cor = cv2.compareHist(hist_cell[2], self.desert_hist_1[2], cv2.HISTCMP_BHATTACHARYYA)
                
                # if hist_h_cor > 0.8 and hist_s_cor > 0.8 and hist_v_cor > 0.8:
                #     if detected[i, j]:
                #             print(f"Warning! desert is detected at {i},{j} thrugh hist matching, but {detected[i, j]} is allerady assigned! Overwriting now!")
                #     detected[i, j] = "desert"

                
                # Loop thrugh all biomes and their search parameters
                for biome, values in biomes.items():                 
                    lowerb = np.array([values["H_lower"], values["S_lower"], values["V_lower"]], dtype=np.uint8)
                    upperb = np.array([values["H_upper"], values["S_upper"], values["V_upper"]], dtype=np.uint8)
                    mask = cv2.inRange(cell, lowerb, upperb)
                    mask_median = cv2.medianBlur(mask, 5)
                    if cv2.countNonZero(mask_median) > 100*100*0.1:
                        if detected[i, j]:
                            print(f"Warning! {biome} is detected at {i},{j} but {detected[i, j]} is allerady assigned! Overwriting now!")
                        detected[i, j] = biome
        return detected
                            
        #Show results
        # titles = ["original", "Mask", "Mask median"]
        # images = [img, mask, mask_median]
        # for i in range(5):
        #     plt.subplot(2,2,i+1)
        #     plt.imshow(images[i],"gray")
        #     plt.title(titles[i])
        #     plt.xticks([])
        #     plt.yticks([])
        # plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
             
        print(detected)
        plt.subplot()
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()


  
    def eval_img(self, img): # Function to evaluate a single image
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        biomes = self.__detectBiome(imgHSV)
        crowns = crown_finder(img)
        res = count(biomes, crowns)
        return res
    
    def run(self): # Run evaluation on all images
        res = {}
        for file, img in self.image_dict.items():
            res[file] = self.eval_raw_img(img)
        return res

if __name__ == "__main__":
    test = ImageScore()
    test.run()
