from DeepProtect import Detector
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#df = pd.read_csv('videos/default_Misha1.txt', delimiter=' ', names=['frame', 'labels'])
#y_true = df['labels'].tolist()

IS_DRAWING = True
detector = Detector(path_to_model = '/home/student/Projects/BC_Object_Detection/models/best_Artem2.pt')

#cap = cv2.VideoCapture('rtsp://admin:camera12345@172.22.103.2')
cap = cv2.VideoCapture('videos/Misha1.mp4')
out = cv2.VideoWriter('videos/out_Misha1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 720))
#_, frame = cap.read()

#y_pred = []

while True:
    _, frame = cap.read()
    if _ == False:
        break
    else:
        # cap.grab()
        orig_img = frame.copy()
        res = detector.detect(frame, isDrawing=IS_DRAWING)
        if isinstance(res, list):
            frame = res[1]
        #y_pred.append(int(output))
        out.write(frame)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

cap.release()
out.release()
cv2.destroyAllWindows()