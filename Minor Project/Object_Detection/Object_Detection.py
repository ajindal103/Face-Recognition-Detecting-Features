import cv2 as cv
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.5)

    if (len(ClassIndex) != 0):
        for ClassInd, conf, boxes, in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd <= 80):
                cv.rectangle(frame, boxes, (255, 0, 0), 2)
                cv.putText(frame, classLabels[ClassInd-1], (boxes[0] + 10,
                                                            boxes[1] + 30), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

    cv.imshow('Object Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
