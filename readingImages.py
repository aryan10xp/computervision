import cv2 as cv 
import sys

img = cv.imread(cv.samples.findFile('photos/dd.png'))

if img is None:
    sys.exit("Could not read the image")

cv.imshow('Aryans first image',img)
k=cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("photos/dd.png", img)