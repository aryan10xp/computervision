# Import library
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from flask.views import MethodView
#import ktrain
import argparse
from flask import request,Response,json
import tensorflow as tf
from tensorflow import keras
from OilyPatchDetection import OilyImagePatch
from DetectDarkcirclePatch import DetectDarkCirclePatch
from WrinklePatchDetection import DetectWrinklepatch


    
# EyeColorDetection class
class EyeColorDetection(MethodView):

    eyeColors = None
    eyeColorRange = None

    def __init__(self):
        self.detector =MTCNN() # Face detected
        self.jsonRegister = request.get_json()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-f", type=argparse.FileType())
    

    def initialize(self):
        # define HSV color ranges for eyes colors
        self.eyeColors = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
        self.eyeColorRange = {
            self.eyeColors[0]: ((166, 21, 50), (240, 100, 85)),
            self.eyeColors[1]: ((166, 2, 25), (300, 20, 75)),
            self.eyeColors[2]: ((2, 20, 20), (40, 100, 60)),
            self.eyeColors[3]: ((20, 3, 30), (65, 60, 60)),
            self.eyeColors[4]: ((0, 10, 5), (40, 40, 25)),
            self.eyeColors[5]: ((60, 21, 50), (165, 100, 85)),
            self.eyeColors[6]: ((60, 2, 25), (165, 20, 65))}


    def check_color(self,hsv, color):
        if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and (hsv[1] <= color[1][1]) and (
                hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
            return True
        else:
            return False


    # define eye color category rules in HSV space
    def find_class(self,hsv):
        color_id = 7
        for i in range(len(self.eyeColors) - 1):
            if self.check_color(hsv, self.eyeColorRange[self.eyeColors[i]]) == True:
                color_id = i

        return color_id


    def detectEyeColor(self):
    
        try:
            jsonRegister = request.get_json()
            loadImage = jsonRegister[r'path']
            
            image = cv2.imread(loadImage, cv2.IMREAD_COLOR)
            imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, w = image.shape[0:2]
            imgMask = np.zeros((image.shape[0], image.shape[1], 1))

            result = self.detector.detect_faces(image)
            if result == []:

                return 'Face not detected'
                    

            bounding_box = result[0]['box']
            left_eye = result[0]['keypoints']['left_eye']
            right_eye = result[0]['keypoints']['right_eye']

            eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            eye_radius = eye_distance / 15  # approximate

            cv2.circle(imgMask, left_eye, int(eye_radius), (255, 255, 255), -1)
            cv2.circle(imgMask, right_eye, int(eye_radius), (255, 255, 255), -1)

            cv2.rectangle(image,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (255, 155, 255),
                          2)

            cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
            cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

            eye_class = np.zeros(len(self.eyeColors), np.float)

            for y in range(0, h):
                for x in range(0, w):
                    if imgMask[y, x] != 0:
                        eye_class[self.find_class(imgHSV[y, x])] += 1

            self.main_color_index = np.argmax(eye_class[:len(eye_class) - 1])

            return  self.eyeColors[self.main_color_index]
            
        except:
        
            return 'Image path not valid'
            
    def percentageColorType(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            percentageColorType = None
            if self.eyeColors[self.main_color_index]:
                
                percentageColorType = 25

            return percentageColorType

        except:

            return 'Image path not valid'
            
    
        

# SkinTypeDetection class   
class SkinTypeDetection(EyeColorDetection,MethodView):


    def detectSkin(self):
        
        model = tf.keras.models.load_model(r'D:\Face Mapping\api\Model\Normal Oily Skin Model')
        
        try:
            
            self.loadImage = self.jsonRegister['path']
            image = cv2.imread(self.loadImage)
            #self.result = self.detector.detect_faces(image)
            #if self.result == []:

                #return 'Face not detected'
            classNames=['Normal', 'Oily']
            imgHeight = 180
            imgWidth = 180
            img = keras.preprocessing.image.load_img(
                self.loadImage, target_size=(imgHeight, imgWidth))
            imgArray = keras.preprocessing.image.img_to_array(img)
            imgArray = tf.expand_dims(imgArray, 0) # Create a batch

            predictions = model.predict(imgArray)
            score = tf.nn.softmax(predictions[0])
            self.predictionSkin=classNames[np.argmax(score)]
                
            return self.predictionSkin

        except:

            return 'Image path not valid'
            
    def percentageSkinType(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            percentageSkinType = None
            if self.predictionSkin == 'Normal':
                
                percentageSkinType = 25

            else:
                percentageSkinType = 50

            return percentageSkinType

        except:

            return 'Image path not valid'
            
    def oilyPatch(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            oilyPatch = None
            if self.predictionSkin == 'Oily':
                oilyPatch = OilyImagePatch()
                
                imagePath=oilyPatch.oilyPatch()
                
                return imagePath
                
            else:
            
                return ''
                

        except:

            return 'Image path not valid'
        

# WrinkleDetection class
class WrinkleDetection(EyeColorDetection,MethodView):


    def detectWrinkle(self):
        
        model = tf.keras.models.load_model(r'D:\Face Mapping\api\Model\Wrinkle Detection Model')
        
        try:
            
            self.loadImage = self.jsonRegister['path']
            image = cv2.imread(self.loadImage)
            #self.result = self.detector.detect_faces(image)
            #if self.result == []:

                #return 'Face not detected'
            classNames=['Low Wrinkle', 'Moderate Wrinkle']
            imgHeight = 180
            imgWidth = 180
            img = keras.preprocessing.image.load_img(
                self.loadImage, target_size=(imgHeight, imgWidth))
            imgArray = keras.preprocessing.image.img_to_array(img)
            imgArray = tf.expand_dims(imgArray, 0) # Create a batch

            predictions = model.predict(imgArray)
            score = tf.nn.softmax(predictions[0])
            self.predictionWrinkle=classNames[np.argmax(score)]
                
            return self.predictionWrinkle

        except:

            return 'Image path not valid'
            
    def percentageWrinkle(self):
    
        try:
            
            #if self.result == []:

                #return 'Face not detected'
            percentageWrinkle = None
            if self.predictionWrinkle == 'Low Wrinkle':
                percentageWrinkle = 25

            else:
                percentageWrinkle = 50

            return percentageWrinkle

        except:
        
            return 'Image path not valid'

    def wrinklePatch(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            wrinklePatch = None
            if self.predictionWrinkle == 'Moderate Wrinkle':
                wrinklePatch = DetectWrinklepatch()
                print('wrinkle patch')
                imagePath=wrinklePatch.wrinklePatch()
                
                return imagePath
                
            else:
            
                return ''
                

        except:

            return 'Image path not valid'

class DarkCircleDetection(EyeColorDetection,MethodView):



    def detectDarkCircle(self):
        
        model = tf.keras.models.load_model(r'D:\Face Mapping\api\Model\Dark circle tensorflow model')

        try:
            
            self.loadImage = self.jsonRegister['path']
            image = cv2.imread(self.loadImage)
            #self.result = self.detector.detect_faces(image)
            #if self.result == []:
            
             #   return 'Face not detected'

            classNames=['High dark circle', 'Low dark circle']
            
            imgHeight = 180
            imgWidth = 180
            img = keras.preprocessing.image.load_img(
                self.loadImage, target_size=(imgHeight, imgWidth))
            imgArray = keras.preprocessing.image.img_to_array(img)
            imgArray = tf.expand_dims(imgArray, 0) # Create a batch

            predictions = model.predict(imgArray)
            score = tf.nn.softmax(predictions[0])
            self.prediction=classNames[np.argmax(score)]
                
            return self.prediction

        except:

            return 'Image path not valid'
            
    def percentageDarkCircle(self):
        
        try:
            
            #if self.result == []:

                #return 'Face not detected'
                
            percentageDarkCircle = None
            if self.prediction == 'Low dark circle':
                
                percentageDarkCircle = 25

            else:
                percentageDarkCircle = 50

            return percentageDarkCircle

        except:

            return 'Image path not valid'
            
    def darkcirclePatch(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            darkcirclePatch = None
            if self.prediction == 'High dark circle':
                darkcirclePatch = DetectDarkCirclePatch()
                
                imagePath=darkcirclePatch.darkcirclePatch()
                
                return imagePath
                
            else:
            
                return ''
                

        except:

            return 'Image path not valid'
            
class RednessDetection(EyeColorDetection,MethodView):
    
    def detectRedness(self):
    
        model = tf.keras.models.load_model(r'D:\Face Mapping\api\Model\Redness skin model')

        try:
            
            self.loadImage = self.jsonRegister['path']
            image = cv2.imread(self.loadImage)
            #self.result = self.detector.detect_faces(image)
            #if self.result == []:
            
             #   return 'Face not detected'

            classNames=['Low Redness', 'Moderate Redness']
            
            imgHeight = 180
            imgWidth = 180
            img = keras.preprocessing.image.load_img(
                self.loadImage, target_size=(imgHeight, imgWidth))
            imgArray = keras.preprocessing.image.img_to_array(img)
            imgArray = tf.expand_dims(imgArray, 0) # Create a batch

            predictions = model.predict(imgArray)
            score = tf.nn.softmax(predictions[0])
            self.prediction=classNames[np.argmax(score)]
                
            return self.prediction

        except:

            return 'Image path not valid'
        
        
        
        
    def percentageRedness(self):
        
        try:
        
            #if self.result == []:

                #return 'Face not detected'
            
            percentageRedness=None
            if self.prediction == 'Low Redness':
                percentageRedness = 25
            else:
                percentageRedness = 50
                
            return percentageRedness

        except:
        
            return 'Image path not valid'

    # Import library
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN

from flask.views import MethodView
#import ktrain
import argparse
from flask import request,Response,json
import tensorflow as tf
from tensorflow import keras
from OilyPatchDetection import OilyImagePatch
from DetectDarkcirclePatch import DetectDarkCirclePatch
from WrinklePatchDetection import DetectWrinklepatch


    
# EyeColorDetection class
class EyeColorDetection(MethodView):

    eyeColors = None
    eyeColorRange = None

    def __init__(self):
        self.detector =MTCNN() # Face detected
        self.jsonRegister = request.get_json()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-f", type=argparse.FileType())
    

    def initialize(self):
        # define HSV color ranges for eyes colors
        self.eyeColors = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
        self.eyeColorRange = {
            self.eyeColors[0]: ((166, 21, 50), (240, 100, 85)),
            self.eyeColors[1]: ((166, 2, 25), (300, 20, 75)),
            self.eyeColors[2]: ((2, 20, 20), (40, 100, 60)),
            self.eyeColors[3]: ((20, 3, 30), (65, 60, 60)),
            self.eyeColors[4]: ((0, 10, 5), (40, 40, 25)),
            self.eyeColors[5]: ((60, 21, 50), (165, 100, 85)),
            self.eyeColors[6]: ((60, 2, 25), (165, 20, 65))}


    def check_color(self,hsv, color):
        if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and (hsv[1] <= color[1][1]) and (
                hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
            return True
        else:
            return False


    # define eye color category rules in HSV space
    def find_class(self,hsv):
        color_id = 7
        for i in range(len(self.eyeColors) - 1):
            if self.check_color(hsv, self.eyeColorRange[self.eyeColors[i]]) == True:
                color_id = i

        return color_id


    def detectEyeColor(self):
    
        try:
            jsonRegister = request.get_json()
            loadImage = jsonRegister[r'path']
            
            image = cv2.imread(loadImage, cv2.IMREAD_COLOR)
            imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, w = image.shape[0:2]
            imgMask = np.zeros((image.shape[0], image.shape[1], 1))

            result = self.detector.detect_faces(image)
            if result == []:

                return 'Face not detected'
                    

            bounding_box = result[0]['box']
            left_eye = result[0]['keypoints']['left_eye']
            right_eye = result[0]['keypoints']['right_eye']

            eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            eye_radius = eye_distance / 15  # approximate

            cv2.circle(imgMask, left_eye, int(eye_radius), (255, 255, 255), -1)
            cv2.circle(imgMask, right_eye, int(eye_radius), (255, 255, 255), -1)

            cv2.rectangle(image,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (255, 155, 255),
                          2)

            cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
            cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

            eye_class = np.zeros(len(self.eyeColors), np.float)

            for y in range(0, h):
                for x in range(0, w):
                    if imgMask[y, x] != 0:
                        eye_class[self.find_class(imgHSV[y, x])] += 1

            self.main_color_index = np.argmax(eye_class[:len(eye_class) - 1])

            return  self.eyeColors[self.main_color_index]
            
        except:
        
            return 'Image path not valid'
            
    def percentageColorType(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            percentageColorType = None
            if self.eyeColors[self.main_color_index]:
                
                percentageColorType = 25

            return percentageColorType

        except:

            return 'Image path not valid'
            
    
        

# SkinTypeDetection class   
class SkinTypeDetection(EyeColorDetection,MethodView):


    def detectSkin(self):
        
        model = tf.keras.models.load_model(r'D:\Face Mapping\api\Model\Normal Oily Skin Model')
        
        try:
            
            self.loadImage = self.jsonRegister['path']
            image = cv2.imread(self.loadImage)
            #self.result = self.detector.detect_faces(image)
            #if self.result == []:

                #return 'Face not detected'
            classNames=['Normal', 'Oily']
            imgHeight = 180
            imgWidth = 180
            img = keras.preprocessing.image.load_img(
                self.loadImage, target_size=(imgHeight, imgWidth))
            imgArray = keras.preprocessing.image.img_to_array(img)
            imgArray = tf.expand_dims(imgArray, 0) # Create a batch

            predictions = model.predict(imgArray)
            score = tf.nn.softmax(predictions[0])
            self.predictionSkin=classNames[np.argmax(score)]
                
            return self.predictionSkin

        except:

            return 'Image path not valid'
            
    def percentageSkinType(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            percentageSkinType = None
            if self.predictionSkin == 'Normal':
                
                percentageSkinType = 25

            else:
                percentageSkinType = 50

            return percentageSkinType

        except:

            return 'Image path not valid'
            
    def oilyPatch(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            oilyPatch = None
            if self.predictionSkin == 'Oily':
                oilyPatch = OilyImagePatch()
                
                imagePath=oilyPatch.oilyPatch()
                
                return imagePath
                
            else:
            
                return ''
                

        except:

            return 'Image path not valid'
        

# WrinkleDetection class
class WrinkleDetection(EyeColorDetection,MethodView):


    def detectWrinkle(self):
        
        model = tf.keras.models.load_model(r'D:\Face Mapping\api\Model\Wrinkle Detection Model')
        
        try:
            
            self.loadImage = self.jsonRegister['path']
            image = cv2.imread(self.loadImage)
            #self.result = self.detector.detect_faces(image)
            #if self.result == []:

                #return 'Face not detected'
            classNames=['Low Wrinkle', 'Moderate Wrinkle']
            imgHeight = 180
            imgWidth = 180
            img = keras.preprocessing.image.load_img(
                self.loadImage, target_size=(imgHeight, imgWidth))
            imgArray = keras.preprocessing.image.img_to_array(img)
            imgArray = tf.expand_dims(imgArray, 0) # Create a batch

            predictions = model.predict(imgArray)
            score = tf.nn.softmax(predictions[0])
            self.predictionWrinkle=classNames[np.argmax(score)]
                
            return self.predictionWrinkle

        except:

            return 'Image path not valid'
            
    def percentageWrinkle(self):
    
        try:
            
            #if self.result == []:

                #return 'Face not detected'
            percentageWrinkle = None
            if self.predictionWrinkle == 'Low Wrinkle':
                percentageWrinkle = 25

            else:
                percentageWrinkle = 50

            return percentageWrinkle

        except:
        
            return 'Image path not valid'

    def wrinklePatch(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            wrinklePatch = None
            if self.predictionWrinkle == 'Moderate Wrinkle':
                wrinklePatch = DetectWrinklepatch()
                print('wrinkle patch')
                imagePath=wrinklePatch.wrinklePatch()
                
                return imagePath
                
            else:
            
                return ''
                

        except:

            return 'Image path not valid'

class DarkCircleDetection(EyeColorDetection,MethodView):



    def detectDarkCircle(self):
        
        model = tf.keras.models.load_model(r'D:\Face Mapping\api\Model\Dark circle tensorflow model')

        try:
            
            self.loadImage = self.jsonRegister['path']
            image = cv2.imread(self.loadImage)
            #self.result = self.detector.detect_faces(image)
            #if self.result == []:
            
             #   return 'Face not detected'

            classNames=['High dark circle', 'Low dark circle']
            
            imgHeight = 180
            imgWidth = 180
            img = keras.preprocessing.image.load_img(
                self.loadImage, target_size=(imgHeight, imgWidth))
            imgArray = keras.preprocessing.image.img_to_array(img)
            imgArray = tf.expand_dims(imgArray, 0) # Create a batch

            predictions = model.predict(imgArray)
            score = tf.nn.softmax(predictions[0])
            self.prediction=classNames[np.argmax(score)]
                
            return self.prediction

        except:

            return 'Image path not valid'
            
    def percentageDarkCircle(self):
        
        try:
            
            #if self.result == []:

                #return 'Face not detected'
                
            percentageDarkCircle = None
            if self.prediction == 'Low dark circle':
                
                percentageDarkCircle = 25

            else:
                percentageDarkCircle = 50

            return percentageDarkCircle

        except:

            return 'Image path not valid'
            
    def darkcirclePatch(self):
        
        try:
            #if self.result == []:

                #return 'Face not detected'
                
            darkcirclePatch = None
            if self.prediction == 'High dark circle':
                darkcirclePatch = DetectDarkCirclePatch()
                
                imagePath=darkcirclePatch.darkcirclePatch()
                
                return imagePath
                
            else:
            
                return ''
                

        except:

            return 'Image path not valid'
            
class RednessDetection(EyeColorDetection,MethodView):
    
    def detectRedness(self):
    
        model = tf.keras.models.load_model(r'D:\Face Mapping\api\Model\Redness skin model')

        try:
            
            self.loadImage = self.jsonRegister['path']
            image = cv2.imread(self.loadImage)
            #self.result = self.detector.detect_faces(image)
            #if self.result == []:
            
             #   return 'Face not detected'

            classNames=['Low Redness', 'Moderate Redness']
            
            imgHeight = 180
            imgWidth = 180
            img = keras.preprocessing.image.load_img(
                self.loadImage, target_size=(imgHeight, imgWidth))
            imgArray = keras.preprocessing.image.img_to_array(img)
            imgArray = tf.expand_dims(imgArray, 0) # Create a batch

            predictions = model.predict(imgArray)
            score = tf.nn.softmax(predictions[0])
            self.prediction=classNames[np.argmax(score)]
                
            return self.prediction

        except:

            return 'Image path not valid'
        
        
        
        
    def percentageRedness(self):
        
        try:
        
            #if self.result == []:

                #return 'Face not detected'
            
            percentageRedness=None
            if self.prediction == 'Low Redness':
                percentageRedness = 25
            else:
                percentageRedness = 50
                
            return percentageRedness

        except:
        
            return 'Image path not valid'

    