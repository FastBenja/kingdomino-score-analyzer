import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

"""Class to load images and evaluate them

"""
class ImageScore:
    def __init__(self):
        root = tk.Tk()
        root.withdraw()
        self.paths = filedialog.askopenfilenames()
        
        self.image_dict = {} # Init dictonary to hold images and their filenames
        
        for path in self.paths: # Import selected images
            path_segments = path.split('/')
            pretty_path = path_segments[len(path_segments)-1]
            self.image_dict[pretty_path] = np.array(cv2.imread(path))
            #print(f"Imported {pretty_path}")

    def eval_raw_img(self, img): # Function to evaluate an image
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
