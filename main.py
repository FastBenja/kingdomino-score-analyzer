import cv2
import numpy as np
#import tkinter as tk
#from tkinter import filedialog
import os
from matplotlib import pyplot as plt


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
        forrest_tile_file_path = "./King Domino dataset/forrest_sample.jpg"

        # Load all images in the dataset folder and store in dictionary
        for name in os.listdir(dataset_folder_path):
            self.paths.append(os.path.join(dataset_folder_path, name))
        for dataset_folder_path in self.paths:
            path_segments = dataset_folder_path.split('/')
            file_name = path_segments[len(path_segments)-1]
            self.image_dict[file_name] = np.array(cv2.imread(dataset_folder_path))
            
        # Load tile and calculate histogram
        forrest_tile_img = np.array(cv2.imread(forrest_tile_file_path))
        cv2.cvtColor(forrest_tile_img, cv2.COLOR_BGR2HSV, forrest_tile_img)
        self.forrest_hist_h = cv2.calcHist([forrest_tile_img], [0], None, [256], [0, 256])
        cv2.normalize(self.forrest_hist_h,self.forrest_hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        self.forrest_hist_s = cv2.calcHist([forrest_tile_img], [1], None, [256], [0, 256])
        cv2.normalize(self.forrest_hist_s,self.forrest_hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        self.forrest_hist_v = cv2.calcHist([forrest_tile_img], [2], None, [256], [0, 256])
        cv2.normalize(self.forrest_hist_v,self.forrest_hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

 
        
            
    def create_HS_histogram(self, img):
        # hist = cv2.calcHist(img, [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256], )
        # plt.hist(hist.ravel(),256,[0,256]); plt.show()        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        histrH = cv2.calcHist([img], [0], None, [256],[0, 256])
        histrS = cv2.calcHist([img], [1], None, [256],[0, 256])
        histr = cv2.calcHist([img], [0, 1], None, [256, 256], [0, 256, 0, 256])
        plt.plot(histr)
        #plt.plot(histrH, color='r')
        #plt.plot(histrS, color='g')
        plt.xlim([0,256])
        plt.show()
        cv2.waitKey(0)
        

    def eval_raw_img(self, img): # Function to evaluate an image
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detected = np.zeros((len(biomes), 5, 5))
        
        # Loop thrugh all cells on the board and test for biome type
        for i in range(5):
            for j in range(5):
                y1, y2 = i*100, (i+1)*100
                x1, x2 = j*100, (j+1)*100
                cell = imgHSV[y1:y2, x1:x2]
                
                # Calculate histograms for cell
                hist_h = cv2.calcHist([cell], [0], None, [256], [0, 256])
                cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_s = cv2.calcHist([cell], [1], None, [256], [0, 256])
                cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_v = cv2.calcHist([cell], [2], None, [256], [0, 256])
                cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                hist_h_cor = cv2.compareHist(self.forrest_hist_h, hist_h, cv2.HISTCMP_BHATTACHARYYA)
                hist_s_cor = cv2.compareHist(self.forrest_hist_s, hist_s, cv2.HISTCMP_BHATTACHARYYA)
                hist_v_cor = cv2.compareHist(self.forrest_hist_v, hist_v, cv2.HISTCMP_BHATTACHARYYA)
                    
                #print(f'Histogram comp: H:{hist_h_cor}\tS:{hist_s_cor}\tV:{hist_v_cor}')
                #plt.plot(hist_h,hist_s,hist_v)
                #plt.show()
                
                # Loop thrugh all biomes and their search parameters
                for biome, values in biomes.items():                 
                    lowerb = np.array([values["H_lower"], values["S_lower"], values["V_lower"]], dtype=np.uint8)
                    upperb = np.array([values["H_upper"], values["S_upper"], values["V_upper"]], dtype=np.uint8)
                    mask = cv2.inRange(cell, lowerb, upperb)
                    mask_median = cv2.medianBlur(mask, 5)
                    if cv2.countNonZero(mask_median) > 100*100*0.1:
                        detected[list(biomes.keys()).index(biome), i, j] = 1
                    # print(biome)
                    # print(detected[list(biomes.keys()).index(biome), :, :])
                            
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
        plt.subplot(1,2)
        plt.imshow(img)
        plt.imshow
        
                
        score = 60
        return score
    
    def run(self): # Run evaluation on all images
        res = {}
        for file, img in self.image_dict.items():
            res[file] = self.eval_raw_img(img)
        return res

if __name__ == "__main__":
    test = ImageScore()
    test.run()
