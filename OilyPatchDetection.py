# Import library
from flask import Flask,request,Response,json
from flask.views import MethodView
from flask_cors import CORS
from imutils import contours
from skimage import measure
import numpy as np
from datetime import time
from mtcnn.mtcnn import MTCNN
import dlib
import imutils
import json
import cv2
import datetime


# Oily patch detection
class OilyImagePatch(MethodView):

    def oilyPatch(self):

        jsonRegister = request.get_json()
        image = jsonRegister['path']
        loadImage = cv2.imread(image, cv2.IMREAD_COLOR)
        try:
            
            detectorDlib = dlib.get_frontal_face_detector()
            # Load facila landmarks model
            predictor = dlib.shape_predictor(r'C:\inetpub\wwwroot\Facemapping\Model\Facial Landmark model\shape_predictor_68_face_landmarks.dat')
            
            grayImg = cv2.cvtColor(loadImage, cv2.COLOR_BGR2GRAY)
            blurredImg = cv2.GaussianBlur(grayImg, (11, 11), 0)
            facesDetect = detectorDlib(blurredImg)
            
            # Face detector
            detectorMTCNN =MTCNN() 
            detectFace = detectorMTCNN.detect_faces(loadImage)
            
            if detectFace==[]:
                return 'Face not detected'
            
            bounding_box = detectFace[0]['box']
            
            x3=bounding_box[0]
            y3=bounding_box[1]
            w=x3 + bounding_box[2]
            w1=x3 + bounding_box[2]/2
            h=y3 + bounding_box[3]
            
            for face in facesDetect:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                print(x1,y1,x2,y2)

                a = []
                b = []
                
                landmarks = predictor(loadImage, face)
                
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    
                    y = landmarks.part(n).y
                    
                    a.append(x)
                    b.append(y)
                
                facialpoint = np.array([[a[0],b[0]], [a[1],b[1]], [a[2],b[2]], [a[3],b[3]], [a[4],b[4]], [a[5],b[5]], [a[6],b[6]],
                                 [a[7],b[7]], [a[8],b[8]], [a[9],b[9]], [a[10],b[10]], [a[11],b[11]], [a[12],b[12]], [a[13],b[13]],
                                 [a[14],b[14]], [a[15],b[15]], [a[16],b[16]], [a[23],y3], [a[20],y3]])
                                 
                

                mask = np.zeros((loadImage.shape[0], loadImage.shape[1]))
                
                cv2.fillConvexPoly(mask, facialpoint, 1)
                
                mask = mask.astype(np.bool)
                
                cropImage = np.zeros_like(loadImage)
                cropImage[mask] = loadImage[mask]
                
            grayImage = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
            
            blurredImage = cv2.GaussianBlur(grayImage, (11, 11), 0)
            
            thresh = cv2.threshold(blurredImage, 200, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)
            

            labels = measure.label(thresh, background=0)

            mask = np.zeros(thresh.shape, dtype="uint8")
            # loop over the unique components
            for label in np.unique(labels):
                # if this is the background label, ignore it
                if label == 0:
                    continue
                # otherwise, construct the label mask and count the
                # number of pixels
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)

                if numPixels > 500:
                    mask = cv2.add(mask, labelMask)

            contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                cv2.drawContours(loadImage, [cnt], -1, (255, 255, 255), 1)
                
                path=image
                time = datetime.datetime.now()
                imagePath = path +  '_skinType.jpg'
                # Saving image
                cv2.imwrite(imagePath, loadImage)

            path=image
            time = datetime.datetime.now()
            imagePath = path +  '_skinType.jpg'
            # Saving image
            cv2.imwrite(imagePath, loadImage)

        except:
            
            path=image
            time = datetime.datetime.now()
            imagePath = path +  '_skinType.jpg'
            # Saving image
            cv2.imwrite(imagePath, loadImage)
        
        return imagePath
