from DeepProtect import Detector
import cv2

detector = Detector(path_to_model = '/home/student/Projects/BC_Object_Detection/best_Artem.pt')

'''cap = cv2.VideoCapture('/home/student/Projects/BC_Object_Detection/dataset/images/train/Misha1_frame_001217.jpg')
_, frame = cap.read()
res = detector.detect(frame)
print(res[0])
cv2.imshow('Frame', res[1])
cv2.waitKey(5)
cap.release()
cv2.destroyAllWindows()'''
cap = cv2.VideoCapture('rtsp://admin:camera12345@172.22.103.2')

while True:
    cap.grab()
    _, frame = cap.read()
    res = detector.detect(frame)
    cv2.imshow('Frame', res[1])
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break