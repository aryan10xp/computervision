#trackbar as color pallete 
import numpy as np
import cv2 as cv

def nothing(x):
    pass

#creating black image
img = np.zeros((300,512,3),np.uint8)
cv.namedWindow('image')

#creating 3 trackbars for red green blue
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)

#creating switch for on/off
switch='0 : OFF \n1: ON'
cv.createTrackbar(switch,'image',0,1,nothing)

while(1):
    cv.imshow('image',img)
    k=cv.waitKey(1) & 0xFF
    if k == 27:
        break

    #getting current position of the 4 trackbars
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    s = cv.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]
cv.destroyAllWindows 
          
