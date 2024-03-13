import cv2 as cv

cap=cv.VideoCapture('Vids/demovid.mp4')

fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv.VideoWriter('demovid.mp4',fourcc,20.0,(640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can not receive frame, Exiting...")
        break
    frame = cv.flip(frame,0)

    out.write(frame)

    cv.imshow('frame',frame)
    #click on q to end the video
    if cv.waitKey(1) == ord('q'):
        break


cap.release()
out.release()
cv.destroyAllWindows()
    