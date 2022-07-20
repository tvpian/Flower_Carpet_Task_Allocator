# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 18:59:00 2021

@author: Tharun V.P
"""

# import the necessary packages
import argparse
import imutils
import cv2
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
img = cv2.imread('test.png')
#img = cv2.resize(img, (700, 600))

blurred = cv2.GaussianBlur(img, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Resizing the image
image = cv2.resize(img, (700, 600))
image = img

image

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
cv2  
# Make python sleep for unlimited time
cv2.waitKey(0)





# and threshold it
#image = cv2.imread('Test_Images/2021_08_31-03_55_54_PM_abox.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


# apply a series of dilations
for i in range(0, 4):
	dilated = cv2.closing(mask.copy(), None, iterations=i + 1)
	#cv2.imshow("Dilated {} times".format(i + 1), dilated)
	#cv2.waitKey(0)
    
cv2.imshow("Final", dilated)
cv2.waitKey(0)


# find contours in the thresholded image
cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

crop_centers = []
crop_cx = []
crop_cy=[]
# loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
    crop_centers.append([cX,cY])
    crop_cx.append(cX)
    crop_cy.append(cY)
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    #cv2.putText(image, "center", (cX - 20, cY - 20),
        #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
# show the image
cv2.imshow("Image", image)
cv2.waitKey(0)


stack_xmin = min(crop_cx)
stack_xmax = max(crop_cx)
stack_ymin = min(crop_cy)
stack_ymax = max(crop_cy)
stack_xmid = int((stack_xmin + stack_xmax)/2)
stack_ymid = int((stack_ymin + stack_ymax)/2)


q1_xmid = int((stack_xmid + stack_xmin)/2)
q1_ymid = int((stack_ymid + stack_ymax)/2)

q2_xmid = int((stack_xmid + stack_xmin)/2)
q2_ymid = int((stack_ymid + stack_ymin)/2)

q3_xmid = int((stack_xmid + stack_xmax)/2)
q3_ymid = int((stack_ymid + stack_ymin)/2)

q4_xmid = int((stack_xmid + stack_xmax)/2)
q4_ymid = int((stack_ymid + stack_ymax)/2)



cv2.circle(image, (stack_xmid, stack_ymid), 12, (0, 0, 255), -1)
cv2.putText(image, "stack_center", (stack_xmid - 20, stack_ymid - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


cv2.circle(image, (q1_xmid, q1_ymid), 12, (0, 0, 255), -1)
cv2.putText(image, "stackq1_center", (q1_xmid - 20, q1_ymid - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


cv2.circle(image, (q2_xmid, q2_ymid), 12, (0, 0, 255), -1)
cv2.putText(image, "stackq2_center", (q2_xmid - 20, q2_ymid - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


cv2.circle(image, (q3_xmid, q3_ymid), 12, (0, 0, 255), -1)
cv2.putText(image, "stackq3_center", (q3_xmid - 20, q3_ymid - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


cv2.circle(image, (q4_xmid, q4_ymid), 12, (0, 0, 255), -1)
cv2.putText(image, "stackq4_center", (q4_xmid - 20, q4_ymid - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# show the image
cv2.imshow("Image", image)
cv2.waitKey(0)

















