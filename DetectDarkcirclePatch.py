# Import library
import cv2
import dlib
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

# Darkcircle patch detection
class DetectDarkCirclePatch(MethodView):

    def darkcirclePatch(self):
    
        jsonRegister = request.get_json()
        image = jsonRegister['path']
        print('imageDark',image)
        loadImage = cv2.imread(image, cv2.IMREAD_COLOR)

        try:
            faceDetector = dlib.get_frontal_face_detector()
            # Load facial landmarks model
            predictorModel = dlib.shape_predictor(r'C:\inetpub\wwwroot\Facemapping\Model\Facial Landmark model\shape_predictor_68_face_landmarks.dat')

            grayImage = cv2.cvtColor(loadImage, cv2.COLOR_BGR2GRAY)
            blurredImage = cv2.GaussianBlur(grayImage, (11, 11), 0)
            facesDetect = faceDetector(blurredImage)

            for face in facesDetect:

                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                xCoordinate = []
                yCoordinate = []
                landmarks = predictorModel(loadImage, face)

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y

                    xCoordinate.append(x)
                    yCoordinate.append(y)

               # coordinates
                yLength=yCoordinate[29]-yCoordinate[28]
                yLength=int(yLength/2)

                xLength=xCoordinate[45] - xCoordinate[47]

                cv2.ellipse(loadImage, (xCoordinate[41],yCoordinate[41]+yLength), (xLength,yLength),
                            0, 0, 360, (255,255,255), 1)

                cv2.ellipse(loadImage, (xCoordinate[46], yCoordinate[46]+yLength), (xLength,yLength),
                            0, 0, 360, (255, 255, 255), 1)

                path=image
                time = datetime.datetime.now()
                imagePath = path +  '_darkcircle.jpg'
                # saving image
                cv2.imwrite(imagePath, loadImage)

        except:
        
            path=image
            time = datetime.datetime.now()
            imagePath = path +  '_darkcircle.jpg'
            # saving image
            cv2.imwrite(imagePath, loadImage)
            
        return imagePath







