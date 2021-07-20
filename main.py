from DeepProtect import Detector
import cv2

detector = Detector(path_to_model = '/home/student/Projects/BC_Object_Detection/models/best_Artem2.pt')

cap = cv2.VideoCapture('rtsp://admin:camera12345@172.22.103.2')

while True:
    cap.grab()
    _, frame = cap.read()
    res = detector.detect(frame)
    if res[0]['finally'] == True:
        print('FIannaly')
    cv2.imshow('Frame', res[1])
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break