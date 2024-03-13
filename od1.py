import cv2 as cv
import numpy as np

cap=cv.VideoCapture("highway.mp4")

min_width_rect = 80
min_height_rect = 80

count_line_position = 200

algo=cv.createBackgroundSubtractorMOG2()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy
detect = []
offset = 6
counter = 0

while True:
    ret, frame = cap.read()
    grey = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey,(3,3),5)
    img_sub = algo.apply(blur)
    dilat = cv.dilate(img_sub,np.ones((5,5)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    dilatada = cv.morphologyEx(dilat,cv.MORPH_CLOSE,kernel)
    dilatada = cv.morphologyEx(dilatada,cv.MORPH_CLOSE, kernel)
    countershape, _ = cv.findContours(dilatada, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    cv.line(frame,(500,count_line_position),(150,count_line_position),(255,127,0),3)

    cv.imshow("detector",dilatada)
    cv.imshow("Frame",frame)
    cv.imshow("GREY",grey)


    for (i,c) in enumerate(countershape):
        (x,y,w,h) = cv.boundingRect(c)
        validate_counter = (w>= min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(frame,"Vehicle"+str(counter),(x,y-20),cv.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv.circle(frame,center,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position + offset) and y>(count_line_position - offset):
                counter+=1
            cv.line(frame,(500,count_line_position),(150,count_line_position),(0,127,255),3)
            detect.remove((x,y))
            print("Vehicle Counter:"+str(counter))

    cv.putText(frame,"Vehicle Counter :"+str(counter),(30,70),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)            

   # cv.imshow("Frame",frame)
    #cv.imshow("GREY",grey)
    
    key= cv.waitKey(30)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()