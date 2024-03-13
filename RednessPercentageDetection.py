# Import library
from imutils import contours
from skimage import measure
import numpy as np
from datetime import time
from mtcnn.mtcnn import MTCNN
from flask import request,Response,json
from flask.views import MethodView
import dlib
import imutils
import json
import cv2
import datetime

# Redness percentage detection
class RednessPercentageDetection(MethodView):

    def rednessPercentage(self):
    
        jsonRegister = request.get_json()
        image = jsonRegister['path']
        loadImage = cv2.imread(image, cv2.IMREAD_COLOR)

        try:
            detectorDlib = dlib.get_frontal_face_detector()
            # Facial landmarks model
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

            a = []
            b = []
            for face in facesDetect:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                print(x1,y1,x2,y2)


                landmarks = predictor(loadImage, face)
                
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    a.append(x)
                    b.append(y)


                facialpoint = np.array([[a[0],b[0]], [a[1],b[1]], [a[2],b[2]], [a[3],b[3]],[a[51],b[51]],[a[14],b[14]], [a[24],y3], [a[19],y3]])

                mask = np.zeros((loadImage.shape[0], loadImage.shape[1]))

                cv2.fillConvexPoly(mask, facialpoint, 1)

                mask = mask.astype(np.bool)


                cropImage = np.zeros_like(loadImage)
                cropImage[mask] = loadImage[mask]

                
            lowerRedColorRange = np.array([0,120,70])
            upperRedColorRange = np.array([180,255,255])


            imgHSV = cv2.cvtColor(cropImage, cv2.COLOR_BGR2HSV)
            grayImage = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(imgHSV, lowerRedColorRange, upperRedColorRange)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            areaAppend=[]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area>500:
                    areaAppend.append(area)

                    cv2.drawContours(loadImage, [cnt], -1, (255, 0, 0), 1)
                

            rednessSkin = None
            areaSum = sum(areaAppend)
            if areaSum<10000 and areaSum>200:
                rednessSkin = 'Moderate'
            elif areaSum>10000:
                rednessSkin = 'High'
            else:
                rednessSkin ='low'
                
        except:
            
            'Image path not valid'
            
        return rednessSkin



