# Import library
from imutils import contours
from skimage import measure
import numpy as np
from datetime import time
from flask import request,Response,json
from flask.views import MethodView
from mtcnn.mtcnn import MTCNN
import dlib
import imutils
import json
import cv2
import datetime

# Redness patch detection
class RednessPatchDetection(MethodView):

    def rednessPatch(self): 
        jsonRegister = request.get_json()
        image = jsonRegister['path']
        loadImage = cv2.imread(image, cv2.IMREAD_COLOR)

        try:
            detectorDlib = dlib.get_frontal_face_detector()
            # Load facial landmarks model
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


                facialpoint = np.array([[a[48],b[48]], [a[49],b[49]],
                                 [a[50],b[50]], [a[51],b[51]], [a[52],b[52]], [a[53],b[53]], [a[54],b[54]], [a[55],b[55]],
                                 [a[56],b[56]], [a[57],b[57]], [a[58],b[58]], [a[59],b[59]]])

                facialpoint1 = np.array(
                    [[a[0], b[0]], [a[1], b[1]], [a[2], b[2]], [a[3], b[3]], [a[4], b[4]], [a[5], b[5]], [a[6], b[6]],
                     [a[7], b[7]], [a[8], b[8]], [a[9], b[9]], [a[10], b[10]], [a[11], b[11]], [a[12], b[12]], [a[13], b[13]],
                     [a[14], b[14]], [a[15], b[15]], [a[16], b[16]], [a[24], y3], [a[19], y3]])

                faceMask = np.zeros((loadImage.shape[0], loadImage.shape[1]))
                mouthMask = np.zeros((loadImage.shape[0], loadImage.shape[1]))

                cv2.fillConvexPoly(faceMask, facialpoint, 1)
                cv2.fillConvexPoly(mouthMask, facialpoint1, 1)

                faceMask = faceMask.astype(np.bool)
                mouthMask = mouthMask.astype(np.bool)


                faceImage = np.zeros_like(loadImage)
                mouthImage = np.zeros_like(loadImage)

                faceImage[faceMask] = loadImage[faceMask]
                mouthImage[mouthMask] = loadImage[mouthMask]

            imageSubstract = mouthImage - faceImage
            cv2.subtract(mouthImage, faceImage, imageSubstract)

            redRangeLower = np.array([0,120,70])
            redRangeupper = np.array([180,255,255])


            imgHSV = cv2.cvtColor(imageSubstract, cv2.COLOR_BGR2HSV)
            grayImage = cv2.cvtColor(imageSubstract, cv2.COLOR_BGR2GRAY)
            redFaceMask = cv2.inRange(imgHSV, redRangeLower, redRangeupper)

            contours, hierarchy = cv2.findContours(redFaceMask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            areaAppend=[]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area>500:
                    areaAppend.append(area)


                    cv2.drawContours(loadImage, [cnt], -1, (255, 255, 255), 1)


            path=image
            time = datetime.datetime.now()
            imagePath = path +  '_redness.jpg'
            # Saving image
            cv2.imwrite(imagePath, loadImage)
        except:

            path=image
            time = datetime.datetime.now()
            imagePath = path +  '_redness.jpg'
            # Saving image
            cv2.imwrite(imagePath, loadImage)
            
        return imagePath



