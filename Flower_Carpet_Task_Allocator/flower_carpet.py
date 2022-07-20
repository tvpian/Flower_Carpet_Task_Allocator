# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:40:03 2021

@author: Tharun V.P
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt




# Python program to identify
#color in images
  
# Importing the libraries OpenCV and numpy
import cv2
import numpy as np
  
# Read the images
img = cv2.imread('Test_Images/2021_08_31-03_55_54_PM_abox.jpg')
  
# Resizing the image
image = cv2.resize(img, (700, 600))
  
# Convert Image to Image HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
# Defining lower and upper bound HSV values
lower = np.array([55, 100, 50])
upper = np.array([75, 255, 255])
  
# Defining mask for detecting color
mask = cv2.inRange(hsv, lower, upper)
  
# Display Image and Mask
cv2.imshow("Image", image)
cv2.imshow("Mask", mask)
  
# Make python sleep for unlimited time
cv2.waitKey(0)


# apply a series of dilations
for i in range(0, 3):
	dilated = cv2.dilate(mask.copy(), None, iterations=i + 1)
	cv2.imshow("Dilated {} times".format(i + 1), dilated)
	cv2.waitKey(0)
    
cv2.imshow("Final", dilated)
cv2.waitKey(0)


contours = cv2.findContours(dilated, 
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, cnts[2], -1, (0,0,255), thickness = 2)
#fig, ax = plt.subplots(1, figsize=(12,8))
cv2.imshow("Final", image)
cv2.waitKey(0)
cv2.clos



for c in contours[1]:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
cv2.imshow("Detected Crop Maps", image)
cv2.waitKey(0)



# find contours in the binary image
#im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours[1]:
   # calculate moments for each contour
   M = cv2.moments(c)

   # calculate x,y coordinate of center
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   cv2.circle(dilated, (cX, cY), 5, (255, 255, 255), -1)
   cv2.putText(dilated, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

   # display the image
   cv2.imshow("Image", img)
   cv2.waitKey(0)




# convert image to grayscale image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image,127,255,0)

# calculate moments of binary image
M = cv2.moments(dilated)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# display the image
cv2.imshow("Image", img)
cv2.waitKey(0)