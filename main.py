import cv2
import numpy as np
#import tkinter as tk
#from tkinter import filedialog
import os

biomes: dict[str, dict[str, int]] = {
    "forrest": {"H_lower": 44, "H_upper": 96,  "S_lower": 46,  "S_upper": 173, "V_lower": 0,   "V_upper": 115},
    "desert":  {"H_lower": 15, "H_upper": 41,  "S_lower": 28,  "S_upper": 170, "V_lower": 4,   "V_upper": 139},
    "mine":    {"H_lower": 0,  "H_upper": 0,   "S_lower": 0,   "S_upper": 0,   "V_lower": 0,   "V_upper": 0},
    "water":   {"H_lower": 134,"H_upper": 155, "S_lower": 205, "S_upper": 255, "V_lower": 108, "V_upper": 217},
    "field":   {"H_lower": 35, "H_upper": 42,  "S_lower": 230, "S_upper": 255, "V_lower": 194, "V_upper": 206},
    "grass":   {"H_lower": 52, "H_upper": 71,  "S_lower": 174, "S_upper": 238, "V_lower": 130, "V_upper": 174},
}

class ImageScore:
    """
    Class to load images and evaluate them
    """
    def __init__(self):
        #root = tk.Tk()
        #root.withdraw()
        #self.paths = filedialog.askopenfilenames()

        self.biomes = biomes

        path = "./King Domino dataset/Cropped and perspective corrected boards"
        self.paths = []
        for name in os.listdir(path):
            self.paths.append(os.path.join(path, name))

        self.image_dict = {} # Init dictonary to hold images and their filenames
        
        for path in self.paths: # Import selected images
            path_segments = path.split('/')
            pretty_path = path_segments[len(path_segments)-1]
            self.image_dict[pretty_path] = np.array(cv2.imread(path))
            #print(f"Imported {pretty_path}")

    def eval_raw_img(self, img): # Function to evaluate an image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            for biome, values in self.biomes.items():
                print(biome)
                _, maskH = cv2.threshold(img[:,:,0], values["H_lower"], values["H_upper"], cv2.THRESH_BINARY)
                _, maskS = cv2.threshold(img[:,:,1], values["S_lower"], values["S_upper"], cv2.THRESH_BINARY)
                _, maskV = cv2.threshold(img[:,:,2], values["V_lower"], values["V_upper"], cv2.THRESH_BINARY)

                res = cv2.bitwise_and(maskH, cv2.bitwise_and(maskS, maskV))
                cv2.imshow("test", img)
                cv2.waitKey(0)
                break


            



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
