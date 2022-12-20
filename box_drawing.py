# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:49:52 2022

@author: egorp
"""
#%%

import cv2
import numpy as np  

#%% image loading

path = 'C:/Users/egorp/Documents/1_3.png'
img = cv2.imread(path)

#%% mask creating

# thresholds of the lower and higher levels of light
lower = np.array([30, 30, 30])
higher = np.array([250, 250, 250])

mask = cv2.inRange(img, lower, higher)

#plt.imshow(mask, 'gray')

#%% contours finding

cont,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#cont_img = cv2.drawContours(img, cont, -1, 255, 3)

#plt.imshow(cont_img)

#%% area calculating

max_area = max(cont, key = cv2.contourArea)

#%% boundig box drawing

"""
x - координата по оси Х
y - координата по оси Y
w - ширина
h - высота
"""
#calculating a recatangle coordinates
x, y, width, height = cv2.boundingRect(max_area)

# drawing the rectangle
cv2.rectangle(img, (x, y), (x + width, y +  height), (0, 255, 0), 5)

#plt.imshow(img)

#%% image saving
 
"""
# Filename
img_name = '1_2.jpg'
  
# Using cv2.imwrite() method to save the image
cv2.imwrite(img_name, img)
"""

#%% preparing data for CSV file

# image, ex "part.png"
image = path.split("/")[-1]

# annotator - наверное класс (1, 2 и т.д.)
annotator = None

# id - номер изображения в папке. Ex. 1_1, 1_2, 1_3 и т.д.
id_mark = str(((path.split(".")[-2]).split("/")[-1]).split("_")[-1])

# annotation_id - совпадают с id. Возможно что-то другое
annotation_id = id_mark

# rectanglelabels 
rectanglelabels = []
rectanglelabels_data = image.split(".")[-2]
rectanglelabels.append(rectanglelabels_data)

# label
label_keys = ["x", "y", "width", 
           "height", "rotation", "rectanglelabels",
           "original_width", "original_height"]

label_keys_meanings = [x, y, width, 
                    height , 0, rectanglelabels,
                    img.shape[0], img.shape[1]]

label = []
label.append(dict(zip(label_keys, label_keys_meanings)))

# table header creation
table_header = ["Image", "id", "label", "Annotator", "Annotation_id"]

# list of image data for CSV table
image_data = [image, id_mark, label, annotator, annotation_id]

#%% CSV file creating

# import csv library
import csv

with open(f'{path.split(".")[-2]}.csv', 'w') as file:
    
    #Create a CSV writer
    writer = csv.writer(file)
    
    #Write data (header and data itself) to the file
    writer.writerow(table_header)
    writer.writerow(image_data)
