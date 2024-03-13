# Import library
import cv2
import numpy as np
import dlib
import imutils
from imutils import contours
from PIL import Image
from skimage import measure
import cv2
from flask import Flask,request,Response,json
from flask.views import MethodView
import numpy as np
from mtcnn.mtcnn import MTCNN
import dlib
import imutils
from imutils import contours
from PIL import Image
from skimage import measure

# Wrinkle percentage detection
class WrinklePercentage(MethodView):

    def __init__(self):
        self.jsonRegister = request.get_json()
        self.image = self.jsonRegister['path']
        self.loadImage = cv2.imread(self.image, cv2.IMREAD_COLOR)

        self.detectorDlib = dlib.get_frontal_face_detector()
        # Load facial landmarks 
        self.modelPredictor = dlib.shape_predictor(r'C:\inetpub\wwwroot\Facemapping\Model\Facial Landmark model\shape_predictor_68_face_landmarks.dat')

        self.grayImg = cv2.cvtColor(self.loadImage, cv2.COLOR_BGR2GRAY)
        self.blurredImg = cv2.GaussianBlur(self.grayImg, (11, 11), 0)
        self.facesDetect = self.detectorDlib(self.blurredImg)
        
        # Fcae detector
        self.detectorMTCNN = MTCNN()
        self.detectFace = self.detectorMTCNN.detect_faces(self.loadImage)

        if self.detectFace == []:
            
            return 'Face not detected'

        self.bounding_box = self.detectFace[0]['box']

        self.x3 = self.bounding_box[0]
        self.y3 = self.bounding_box[1]
        self.w = self.x3 + self.bounding_box[2]
        self.w1 = self.x3 + self.bounding_box[2] / 2
        self.h = self.y3 + self.bounding_box[3]

        self.xCoordinate = []
        self.yCoordinate = []
        for face in self.facesDetect:
            self.x1 = face.left()
            self.y1 = face.top()
            self.x2 = face.right()
            self.y2 = face.bottom()

            self.landmarks = self.modelPredictor(self.loadImage, face)
            
            for n in range(0, 68):
                x = self.landmarks.part(n).x
                y = self.landmarks.part(n).y

                self.xCoordinate.append(x)
                self.yCoordinate.append(y)


    def foreheadRegion(self):

        try:
            x = None
            if self.yCoordinate[24] > self.yCoordinate[19]:
                x = self.yCoordinate[19]

            else:
                x = self.yCoordinate[24]
                
            foreheadRegion = self.loadImage[self.y3:x, self.xCoordinate[20]:self.xCoordinate[23]]
            grayImage = cv2.cvtColor(foreheadRegion, cv2.COLOR_BGR2GRAY)
            invertImage = 255 - grayImage
            blurrImage = cv2.GaussianBlur(invertImage, (21, 21), 0)
            invertedBlurrImage = 255 - blurrImage
            pencilImage = cv2.divide(grayImage, invertedBlurrImage, scale=256.0)
            edgeImage = cv2.Canny(pencilImage, 15, 200, 1)
            dilateImage3 = cv2.dilate(edgeImage, None, iterations=2)

            contours1, key = cv2.findContours(dilateImage3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areaForehead = []
            for cnt1 in contours1:
                area = cv2.contourArea(cnt1)
                if area > 100:
                    cv2.drawContours(foreheadRegion, [cnt1], -1, (255, 255, 255), 1)
                    areaForehead.append(int(area))
                foreheadArea=sum(areaForehead)
            return foreheadArea

        except:
            return 'ROI not found 1'

    def leftCheekRegion(self):
        try:
            y = None
            if self.xCoordinate[17] < self.xCoordinate[0]:
                y = self.yCoordinate[0]
            elif self.xCoordinate[17] < self.xCoordinate[1]:
                y = self.yCoordinate[1]
            elif self.xCoordinate[17] < self.xCoordinate[2]:
                y = self.yCoordinate[2]
            else:
                y = self.yCoordinate[3]
            
            leftCheekRegion = self.loadImage[self.yCoordinate[28]:y, self.xCoordinate[3]:self.xCoordinate[39]]
            grayImage = cv2.cvtColor(leftCheekRegion, cv2.COLOR_BGR2GRAY)
            invertImage = 255 - grayImage
            blurrImage = cv2.GaussianBlur(invertImage, (21, 21), 0)
            invertedBlurrImage = 255 - blurrImage
            pencilImage = cv2.divide(grayImage, invertedBlurrImage, scale=256.0)
            edgeImage = cv2.Canny(pencilImage, 15, 200, 1)
            dilateImage3 = cv2.dilate(edgeImage, None, iterations=2)

            contours2, key = cv2.findContours(dilateImage3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areaLeftCheek = []
            for cnt2 in contours2:
                area = cv2.contourArea(cnt2)
                if area > 100:
                    cv2.drawContours(leftCheekRegion, [cnt2], -1, (255, 255, 255), 1)
                    areaLeftCheek.append(area)
                leftCheekArea=sum(areaLeftCheek)
            
            return leftCheekArea
        except:
            return 'ROI not found 2'

    def rightCheekRegion(self):
        try:
            z = None
            if self.xCoordinate[26] > self.xCoordinate[16]:
                z = self.yCoordinate[16]
            elif self.xCoordinate[26] > self.xCoordinate[15]:
                z = self.yCoordinate[15]
            elif self.xCoordinate[26] > self.xCoordinate[14]:
                z = self.yCoordinate[14]
            else:
                z = self.yCoordinate[13]
                
            rightCheekRegion = self.loadImage[self.yCoordinate[28]:z, self.xCoordinate[42]:self.xCoordinate[13]]
            grayImage = cv2.cvtColor(rightCheekRegion, cv2.COLOR_BGR2GRAY)
            invertImage = 255 - grayImage
            blurrImage = cv2.GaussianBlur(invertImage, (21, 21), 0)
            invertedBlurrImage = 255 - blurrImage
            pencilImage = cv2.divide(grayImage, invertedBlurrImage, scale=256.0)
            edgeImage = cv2.Canny(pencilImage, 15, 200, 1)
            dilateImage3 = cv2.dilate(edgeImage, None, iterations=2)

            contours3, key = cv2.findContours(dilateImage3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areaRightCheek = []
            for cnt3 in contours3:
                area = cv2.contourArea(cnt3)
                if area > 100:
                    cv2.drawContours(rightCheekRegion, [cnt3], -1, (255, 255, 255), 1)
                    areaRightCheek.append(area)
                rightCheekArea=sum(areaRightCheek)
            
            return rightCheekArea
        except:
            return 'ROI not found 3'



            
