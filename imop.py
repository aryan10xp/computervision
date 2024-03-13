#basic operations on images
import cv2 as cv
import numpy as np

image = cv.imread('photos/test.jpg')

px = image[100,100]
##print(px)

blue=image[100,100,0]
#print(blue)

green=image[100,100,1]
#print(green)

red=image[100,100,2]
#print(red)

array=np.array([red,green,blue])
#print(array)

image [10,10] = [0,255,0]
print (image.shape)

new=cv.resize(image,(512,512))
print (new.shape)

#to set all red pixels to 0
#b=new[:,:,2]=0


a=new[280:340, 330:390]
new[273:333, 100:160]=a

cv.imshow("resized",new)
cv.waitKey(0)
cv.destroyAllWindows()