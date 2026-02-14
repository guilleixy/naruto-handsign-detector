import os
import cv2

capture = cv2.VideoCapture(0)

while True:
    success, frame = capture.read()
    if not success:
        break

    

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break