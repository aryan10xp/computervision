#object detection and tracking
import cv2 as cv

cap=cv.VideoCapture("highway.mp4")

count_line_position = 200

car_count = 0 
prev_car_count = 0

object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    #print(height, width)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    roi=frame[180:380,160:600]

    mask = object_detector.apply(roi)
    _, mask=cv.threshold(mask, 254,255,cv.THRESH_BINARY)
    contours, _ =cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.line(frame,(500,count_line_position),(150,count_line_position),(255,127,0),3)

    for cnt in contours:
        area=cv.contourArea(cnt)
        if area > 100:
            #cv.drawContours(roi,[cnt], -1,(0,255,0),2)
            x,y,w,h=cv.boundingRect(cnt)
            rec=cv.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)

            center_x = x + w // 2
            center_y = y + h //2

            cv.circle(roi,(center_x,center_y),4,(0,0,255),-1)
        
    cv.putText(frame,'Car Count: '+ str(car_count),(10,30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    if car_count != prev_car_count:
        print('Car Count: ', car_count)
    
    prev_car_count = car_count

    cv.imshow("roi",roi)
    cv.imshow("Frame",frame)
    cv.imshow("mask",mask)

    key = cv.waitKey(30)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()