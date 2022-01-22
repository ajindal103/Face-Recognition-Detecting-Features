import cv2 as cv

capture = cv.VideoCapture(0)
cascade_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    isTrue, frame = capture.read()

    detections = cascade_classifier.detectMultiScale(frame, 1.3, 5)

    if (len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv.rectangle (frame, (x,y), (x+w,y+h),(0,255,0), 2)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()