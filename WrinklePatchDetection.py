import cv2
import numpy as np
import dlib
import imutils
from imutils import contours
from PIL import Image
from skimage import measure
import cv2
from flask.views import MethodView
import numpy as np
from mtcnn.mtcnn import MTCNN
import dlib
import imutils
from imutils import contours
from PIL import Image
from skimage import measure
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



class DetectWrinklepatch(MethodView):

    def wrinklePatch(self):
    
        jsonRegister = request.get_json()
        image = jsonRegister['path']
        print('image',image)
        loadImage = cv2.imread(image, cv2.IMREAD_COLOR)

        try:
            detectorDlib = dlib.get_frontal_face_detector()
            modelPredictor = dlib.shape_predictor(r'D:\Face Mapping\api\Model\Facial Landmark model\shape_predictor_68_face_landmarks.dat')

            grayImg = cv2.cvtColor(loadImage, cv2.COLOR_BGR2GRAY)
            blurredImg = cv2.GaussianBlur(grayImg, (11, 11), 0)
            facesDetect = detectorDlib(blurredImg)

            detectorMTCNN =MTCNN()
            detectFace = detectorMTCNN.detect_faces(loadImage)

            if detectFace==[]:
                print('hi')
                # return 'Face not detected'

            bounding_box = detectFace[0]['box']

            x3=bounding_box[0]
            y3=bounding_box[1]
            w=x3 + bounding_box[2]
            w1=x3 + bounding_box[2]/2
            h=y3 + bounding_box[3]

            xCoordinate = []
            yCoordinate = []
            for face in facesDetect:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                print(x1,y1,x2,y2)

                landmarks = modelPredictor(loadImage, face)
                # print(landmarks)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    # print(x)
                    y = landmarks.part(n).y

                    # cv2.circle(loadImage, (x, y), 4, (255, 0, 0), -1)
                    xCoordinate.append(x)
                    yCoordinate.append(y)

                x = None
                if yCoordinate[24] > yCoordinate[19]:
                    x = yCoordinate[19]

                else:
                    x = yCoordinate[24]
                print('b[19]', yCoordinate[19])
                print('b[24]', yCoordinate[24])

                y = None
                if xCoordinate[17] < xCoordinate[0]:
                    y = yCoordinate[0]
                elif xCoordinate[17] < xCoordinate[1]:
                    y = yCoordinate[1]
                # elif xCoordinate[17] < xCoordinate[2]:
                #     y = yCoordinate[2]
                else:
                    y = yCoordinate[2]

                z = None
                if xCoordinate[26] > xCoordinate[16]:
                    z = yCoordinate[16]
                elif xCoordinate[26] > xCoordinate[15]:
                    z = yCoordinate[15]
                # elif xCoordinate[26] > xCoordinate[14]:
                #     z = yCoordinate[14]
                else:
                    z = yCoordinate[14]

                foreheadRegion = loadImage[y3:x, xCoordinate[20]:xCoordinate[23]]
                leftCheek = loadImage[yCoordinate[28]:y, xCoordinate[3]:xCoordinate[39]]
                rightCheek = loadImage[yCoordinate[29]:z, xCoordinate[42]:xCoordinate[13]]
                print(foreheadRegion, leftCheek, rightCheek)

                if foreheadRegion==[] and leftCheek==[] and rightCheek==[]:
                    imagePath = r'C:\Users\Hp\Desktop\im\1000.jpg'
                    cv2.imwrite(imagePath, loadImage)
                    # break

                else:



                    try:
                    
                        x = None
                        if yCoordinate[24] > yCoordinate[19]:
                            x = yCoordinate[19]

                        else:
                            x = yCoordinate[24]
                        print('b[19]', yCoordinate[19])
                        print('b[24]', yCoordinate[24])
                        foreheadRegion=loadImage[y3:x, xCoordinate[20]:xCoordinate[23]]




                        grayImage = cv2.cvtColor(foreheadRegion, cv2.COLOR_BGR2GRAY)
                        blurrImage = cv2.GaussianBlur(grayImage,(7,7),0)
                        edgeImage = cv2.Canny(blurrImage, 15, 200, 1)
                        dilateImage=cv2.dilate(edgeImage, None, iterations=2)
                        thresh_gray = cv2.morphologyEx(dilateImage, cv2.MORPH_CLOSE,
                                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
                        foreheadContours, key = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for cnt in foreheadContours:
                            # print('cnt', cnt[0])
                            #
                            # print('cn1', cnt[1])
                            # print('cn2', cnt[2])

                            # # print(cnt)
                            # area = cv2.contourArea(cnt)
                            # print('area1',area)
                            # area = cv2.contourArea(cnt)

                            cv2.drawContours(foreheadRegion, [cnt], -1, (0, 0, 255), 2)
                        # cv2.rectangle(cnt, (x, y), (w, h), (255, 155, 255), 2)
                    # print('12', cnt[0] + cnt[1] + cnt[2])
                    except:
                        # print('abc',foreheadRegion)
                        print('ROI not found1212')



                    try:
                        y = None
                        if xCoordinate[17] < xCoordinate[0]:
                            y = yCoordinate[0]
                        elif xCoordinate[17] < xCoordinate[1]:
                            y = yCoordinate[1]
                        # elif xCoordinate[17] < xCoordinate[2]:
                        #     y = yCoordinate[2]
                        else:
                            y = yCoordinate[2]

                        leftCheek = loadImage[yCoordinate[28]:y, xCoordinate[3]:xCoordinate[39]]

                        grayImage = cv2.cvtColor(leftCheek, cv2.COLOR_BGR2GRAY)
                        blurrImage = cv2.GaussianBlur(grayImage,(7,7),0)
                        edgeImage = cv2.Canny(blurrImage, 15, 200, 2)
                        dilateImage=cv2.dilate(edgeImage, None, iterations=2)

                        leftCheekContours, key = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in leftCheekContours:
                            area = cv2.contourArea(cnt)
                            print('area2', area)
                            cv2.drawContours(leftCheek, [cnt], -1, (0, 0, 255), 2)
                    except:
                        print('ROI not found')

                    try:
                        z = None
                        if xCoordinate[26] > xCoordinate[16]:
                            z = yCoordinate[16]
                        elif xCoordinate[26] > xCoordinate[15]:
                            z = yCoordinate[15]
                        # elif xCoordinate[26] > xCoordinate[14]:
                        #     z = yCoordinate[14]
                        else:
                            z = yCoordinate[14]

                        rightCheek = loadImage[yCoordinate[29]:z, xCoordinate[42]:xCoordinate[13]]

                        grayImage = cv2.cvtColor(rightCheek, cv2.COLOR_BGR2GRAY)
                        blurrImage = cv2.GaussianBlur(grayImage, (7, 7), 0)
                        edgeImage = cv2.Canny(blurrImage, 15, 200, 2)
                        dilateImage = cv2.dilate(edgeImage, None, iterations=2)
                        # thresh_gray = cv2.morphologyEx(dilateImage, cv2.MORPH_CLOSE,
                        #                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
                        # roi3 = loadImage[b[57]:b[10], a[40]:a[45]]
                        #
                        # gray3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2GRAY)
                        # imgBlurr3 = cv2.GaussianBlur(gray3, (7, 7), 0)
                        # edge3 = cv2.Canny(imgBlurr3, 15, 200, 2)
                        # dilate3 = cv2.dilate(edge3, None, iterations=2)

                        rightCheekContours, key = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in rightCheekContours:
                            area = cv2.contourArea(cnt)
                            print('area3', area)

                            cv2.drawContours(rightCheek, [cnt], -1, (0, 0, 255), 2)


                    #
                    except:
                        # print('right', rightCheek)
                        print('ROI not found11')

                    # if foreheadContours==[] and

                    path=image
                    time = datetime.datetime.now()
                    imagePath = path +  '_wrinkle.jpg'
                    cv2.imwrite(imagePath, loadImage)

        except:
            path=image
            time = datetime.datetime.now()
            imagePath = path +  '_wrinkle.jpg'
            cv2.imwrite(imagePath, loadImage)

        return imagePath