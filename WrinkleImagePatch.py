# Import library
import cv2
import numpy as np
import dlib
import imutils
from imutils import contours
from flask import Flask,request,Response,json
from flask.views import MethodView
from PIL import Image
from skimage import measure
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import dlib
import imutils
from imutils import contours
from PIL import Image
from skimage import measure

# Wrinkle patch detection
class DetectWrinklePatch(MethodView):

    def wrinklePatch(self):
        jsonRegister = request.get_json()
        image = jsonRegister['path']
        loadImage = cv2.imread(image, cv2.IMREAD_COLOR)
        
        try:

            detectorDlib = dlib.get_frontal_face_detector()
            # Load facial landmarks model
            modelPredictor = dlib.shape_predictor(r'C:\inetpub\wwwroot\Facemapping\Model\Facial Landmark model\shape_predictor_68_face_landmarks.dat')

            grayImg = cv2.cvtColor(loadImage, cv2.COLOR_BGR2GRAY)
            blurredImg = cv2.GaussianBlur(grayImg, (11, 11), 0)
            facesDetect = detectorDlib(blurredImg)

            # Face detector
            detectorMTCNN = MTCNN()
            detectFace = detectorMTCNN.detect_faces(loadImage)

            if detectFace == []:
                
                return 'Face not detected'

            bounding_box = detectFace[0]['box']

            x3 = bounding_box[0]
            y3 = bounding_box[1]
            w = x3 + bounding_box[2]
            w1 = x3 + bounding_box[2] / 2
            h = y3 + bounding_box[3]

            xCoordinate = []
            yCoordinate = []
            for face in facesDetect:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                landmarks = modelPredictor(loadImage, face)
                
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y

                    xCoordinate.append(x)
                    yCoordinate.append(y)


                try:
                    x = None
                    if yCoordinate[24] > yCoordinate[19]:
                        x = yCoordinate[19]

                    else:
                        x = yCoordinate[24]
                    # Forehead region
                    foreheadRegion = loadImage[y3:x, xCoordinate[19]:xCoordinate[24]]
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


                except:
                    return 'ROI not found 1'


                try:
                    y = None
                    if xCoordinate[17] < xCoordinate[0]:
                        y = yCoordinate[0]
                    elif xCoordinate[17] < xCoordinate[1]:
                        y = yCoordinate[1]
                    elif xCoordinate[17] < xCoordinate[2]:
                        y = yCoordinate[2]
                    else:
                        y = yCoordinate[3]
                        
                    # Left cheek region
                    leftCheekRegion = loadImage[yCoordinate[28]:y, xCoordinate[3]:xCoordinate[39]]
                    grayImage = cv2.cvtColor(leftCheekRegion, cv2.COLOR_BGR2GRAY)
                    invertImage = 255 - grayImage
                    blurrImage = cv2.GaussianBlur(invertImage, (21, 21), 0)
                    invertedBlurrImage = 255 - blurrImage
                    pencilImage = cv2.divide(grayImage, invertedBlurrImage, scale=256.0)
                    edgeImage = cv2.Canny(pencilImage, 15, 200, 1)
                    dilateImage3 = cv2.dilate(edgeImage, None, iterations=2)

                    contours2, key = cv2.findContours(dilateImage3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # areaLeftCheek = []
                    for cnt2 in contours2:
                        area = cv2.contourArea(cnt2)
                        if area > 100:
                            cv2.drawContours(leftCheekRegion, [cnt2], -1, (255, 255, 255), 1)

                except:
                    return 'ROI not found 2'


                try:
                    z = None
                    if xCoordinate[26] > xCoordinate[16]:
                        z = yCoordinate[16]
                    elif xCoordinate[26] > xCoordinate[15]:
                        z = yCoordinate[15]
                    elif xCoordinate[26] > xCoordinate[14]:
                        z = yCoordinate[14]
                    else:
                        z = yCoordinate[13]
                    rightCheekRegion = loadImage[yCoordinate[28]:z, xCoordinate[42]:xCoordinate[13]]
                    grayImage = cv2.cvtColor(rightCheekRegion, cv2.COLOR_BGR2GRAY)
                    invertImage = 255 - grayImage
                    blurrImage = cv2.GaussianBlur(invertImage, (21, 21), 0)
                    invertedBlurrImage = 255 - blurrImage
                    pencilImage = cv2.divide(grayImage, invertedBlurrImage, scale=256.0)
                    edgeImage = cv2.Canny(pencilImage, 15, 200, 1)
                    dilateImage3 = cv2.dilate(edgeImage, None, iterations=2)

                    contours3, key = cv2.findContours(dilateImage3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # areaRightCheek = []

                    for cnt3 in contours3:
                        area = cv2.contourArea(cnt3)
                        if area > 100:
                            cv2.drawContours(rightCheekRegion, [cnt3], -1, (255, 255, 255), 1)


                except:
                    return 'ROI not found 3'

                path=image
                imagePath = path +  '_wrinkle.jpg'
                cv2.imwrite(imagePath, loadImage)
               
                
        except:
            path=image
            imagePath = path +  '_wrinkle.jpg'
            cv2.imwrite(imagePath, loadImage)
            
        return imagePath
