import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

"""Class to load images and evaluate them

"""
class ImageScore:
    def __init__(self):
        #root = tk.Tk()
        #root.withdraw()
        #self.paths = filedialog.askopenfilenames()

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

        self.biomes.forrest.H_lower = 100
        self.biomes.forrest.H_upper = 140
        self.biomes.forrest.S_lower = 150
        self.biomes.forrest.S_upper = 255
        self.biomes.forrest.V_lower = 0
        self.biomes.forrest.V_upper = 255

        self.biomes.desert.H_lower = 100
        self.biomes.desert.H_upper = 140
        self.biomes.desert.S_lower = 150
        self.biomes.desert.S_upper = 255
        self.biomes.desert.V_lower = 0
        self.biomes.desert.V_upper = 255

        self.biomes.mine.H_lower = 100
        self.biomes.mine.H_upper = 140
        self.biomes.mine.S_lower = 150
        self.biomes.mine.S_upper = 255
        self.biomes.mine.V_lower = 0
        self.biomes.mine.V_upper = 255

        self.biomes.water.H_lower = 100
        self.biomes.water.H_upper = 140
        self.biomes.water.S_lower = 150
        self.biomes.water.S_upper = 255
        self.biomes.water.V_lower = 0
        self.biomes.water.V_upper = 255

        self.biomes.field.H_lower = 100
        self.biomes.field.H_upper = 140
        self.biomes.field.S_lower = 150
        self.biomes.field.S_upper = 255
        self.biomes.field.V_lower = 0
        self.biomes.field.V_upper = 255

        self.biomes.grass.H_lower = 100
        self.biomes.grass.H_upper = 140
        self.biomes.grass.S_lower = 150
        self.biomes.grass.S_upper = 255
        self.biomes.grass.V_lower = 0
        self.biomes.grass.V_upper = 255


    def eval_raw_img(self, img): # Function to evaluate an image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)


            



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
