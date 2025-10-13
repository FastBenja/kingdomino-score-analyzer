import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections import deque

# color = np.array([["green","blue","skov","skov","skov"],
#                   ["green","skov","skov","skov","green"],
#                   ["green","brun","start","skov","green"],   
#                   ["green","brun","blue","green","green"],
#                   ["skov","blue","blue","green","yellow"]
#                   ])

# crown = np.array([[0,0,0,0,0],
#                   [0,0,0,1,0],
#                   [0,1,0,0,0],
#                   [0,2,0,2,1],
#                   [0,1,0,1,0]])



# color2 = np.array([["yellow","brune","yellow","black","brune"],
#                   ["black","brune","yellow","yellow","skov"],
#                   ["black","brune","start","yellow","yellow"],   
#                   ["black","black","green","green","yellow"],
#                   ["yellow","yellow","brune","brune","green"]
#                   ])

# crown2 = np.array([[0,0,0,2,0],
#                   [1,1,0,0,0],
#                   [2,0,0,0,0],
#                   [2,3,0,0,1],
#                   [0,0,0,0,0]])


# color33 = np.array([["green","yellow","blue","blue","blue"],
#                   ["yellow","yellow","brune","yellow","skov"],
#                   ["blue","yellow","start","skov","skov"],   
#                   ["blue","skov","skov","skov","table"],
#                   ["blue","skov","yellow","yellow","table"]
#                   ])

# crown33 = np.array([[2,0,1,0,1],
#                   [0,1,0,0,0],
#                   [1,0,0,0,0],
#                   [1,0,0,0,0],
#                   [1,0,0,0,0]])


# color21 = np.array([["brune","green","brune","yellow","skov"],
#                   ["brune","green","brune","yellow","skov"],
#                   ["brune","green","start","yellow","skov"],   
#                   ["brune","green","blue","blue","skov"],
#                   ["yellow","green","blue","blue","blue"]
#                   ])

# crown21 = np.array([[0,0,2,0,0],
#                   [0,0,0,0,0],
#                   [2,0,0,0,0],
#                   [1,1,0,1,0],
#                   [0,2,0,0,0]])

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
