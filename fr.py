#face recognition
import cv2 as cv

face_cap = cv.CascadeClassifier("C:\\Users\\aryan\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

video_cap =cv.VideoCapture(0)
while True:
    ret, video_data = video_cap.read()
    col=cv.cvtColor(video_data,cv.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv.CASCADE_SCALE_IMAGE 
    )
    for(x,y,w,h) in faces:
        cv.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow("video live",video_data)
    if cv.waitKey(10) == ord("a"):
        break

video_cap.release()



