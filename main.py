from DeepProtect import Detector
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Скачать папку weights с сетевого диска и положить в папку DeepProtect

df = pd.read_csv('../videos/default_Artem4.txt', delimiter=' ', names=['frame', 'labels'])
y_true = df['labels'].tolist()

IS_DRAWING = True
detector = Detector(path_to_model = '/home/student/Projects/BC_Object_Detection/models/best_Ira.pt')

#cap = cv2.VideoCapture('rtsp://admin:camera12345@172.22.103.2')
cap = cv2.VideoCapture('../videos/Artem4.mp4')
out = cv2.VideoWriter('../videos/out_Artem4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 720))
#_, frame = cap.read()

y_pred = []

while True:
    #cap.grab()
    _, frame = cap.read()
    if _ == False:
        break
    else:
        orig_img = frame.copy()
        res = detector.detect(frame, isDrawing=IS_DRAWING)
        if isinstance(res, list):
            frame = res[1]
            status = res[0]['reason']
            completed = res[0]['finally']
        else:
            status = res['reason']
            completed = res[0]['finally']
        y_pred.append(int(completed))

        if completed:
            cv2.putText(frame, 'Dressed', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
        else:
            if status == 1:
                cv2.putText(frame, 'Zero people in frame', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
            elif status == 2:
                cv2.putText(frame, 'More than one people', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
            elif status == 3:
                cv2.putText(frame, 'Not completed', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
            elif status == 4:
                cv2.putText(frame, 'Undressed', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)

        out.write(frame)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

print('y_pred = ', y_pred)
print('y_true = ', y_true)
cap.release()
out.release()
cv2.destroyAllWindows()