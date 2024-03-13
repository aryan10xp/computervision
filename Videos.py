import cv2 as cv

cap=cv.VideoCapture(0)
if not cap.isOpened():
    print("Can not open camera")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can not receive frame. Exiting...")
        break
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord("q"):
        break
cap.release()
cv.destroyAllWindows