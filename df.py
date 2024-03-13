#drawing functions

import numpy as np
import cv2 as cv

img = np.zeros((512,512,3),np.uint8)

cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

#adding text to images
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500),font,4,(255,255,255),2)

#this code displays the output
cv.imshow('Image',img)
cv.waitKey(0)
cv.destroyAllWindows()