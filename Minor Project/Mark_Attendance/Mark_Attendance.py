import cv2 as cv
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path = 'images'
images = []
names = []
myList = os.listdir(path)

for i in myList:
    currentImg = cv.imread(f'{path}/{i}')
    images.append(currentImg)
    names.append(os.path.splitext(i)[0])


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


def encodings(images):
    encoding_list = []
    for i in images:
        i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
        encode = fr.face_encodings(i)[0]
        encoding_list.append(encode)
    return encoding_list


encoding_list_known = encodings(images)

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    frame_small = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    frame_small = cv.cvtColor(frame_small, cv.COLOR_BGR2RGB)

    face_loc_frame = fr.face_locations(frame_small)
    encode_frame = fr.face_encodings(frame_small, face_loc_frame)

    for encodeFace, locFace in zip(encode_frame, face_loc_frame):
        matches = fr.compare_faces(encoding_list_known, encodeFace)
        faceDis = fr.face_distance(encoding_list_known, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            y1, x2, y2, x1 = locFace
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y2+37), (x2, y2), (0, 255, 0), -1)
            cv.putText(frame, name, (x1+6, y2+30),
                       cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv.imshow("webcam", frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
