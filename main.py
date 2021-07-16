from DeepProtect import Detector
import cv2

detector = Detector(path_to_model = '/home/student/Projects/BC_Object_Detection/yolov5/runs/train/exp5/weights/best.pt')

cap = cv2.VideoCapture('/home/student/Projects/BC_Object_Detection/dataset/images/train/Misha1_frame_001234.jpg')
_, frame = cap.read()
print(detector.detect(frame))