#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 #Import OpenCV


# In[2]:


#Preprocessing Image
img = cv2.pyrDown(cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED))                      #Load Input Image
orig_img = img.copy()                                                                #Keep original image separate
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                     #Convert it into greyscale
thresh = cv2.threshold(gray_img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  #Threshold it using Otsu’s thresholding method
#cv2.threshold(source greyscale image, thresholdValue, maxVal, thresholdingTechnique) 
#[1] is used as cv2.threshold returns two values (thresholdvalue used, threshold img)


# In[3]:


#Apply Connected Component Analysis to the thresholded image
output = cv2.connectedComponentsWithStats(thresh,8, cv2.CV_32S)    #we are using 8-way connectivity and CV_32S data type
(numLabels, labels, stats, centroids) = output         #cv2.connectedComponentsWithStats returns a 4-tuple that we store in output


# In[4]:


for i in range(0, numLabels):
    x = stats[i, cv2.CC_STAT_LEFT]                                     #Leftmost (x) coordinate for each label
    y = stats[i, cv2.CC_STAT_TOP]                                      #Topmost (y) coordinate for each label
    w = stats[i, cv2.CC_STAT_WIDTH]                                    #Width of bounding box for each label
    h = stats[i, cv2.CC_STAT_HEIGHT]                                   #Height of bounding box for each label
    (cX, cY) = centroids[i]                                            #store centroids 
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)   #draw green rectangles over input image
    img = cv2.circle(img, (int(cX), int(cY)), 4, (0, 0, 255), -1)      #draw red circles as centroids over input image
    print("X:",cX,"Y:",cY)


# In[ ]:


cv2.imshow("Input Image", orig_img)         #show input image
cv2.imshow("Bounding Boxes", img)           #show Image with bounding boxes
cv2.waitKey(0)
cv2.destroyallWindows()
