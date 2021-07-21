from DeepProtect import Detector
import cv2
IS_DRAWING = True
detector = Detector(path_to_model = '/home/student/Projects/BC_Object_Detection/models/best_Artem2.pt')

#cap = cv2.VideoCapture('rtsp://admin:camera12345@172.22.103.2')
cap = cv2.VideoCapture('videos/Misha1.mp4')
#out = cv2.VideoWriter('videos/outpy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 720))
_, frame = cap.read()

while _:
    cap.grab()
    orig_img = frame.copy()
    res = detector.detect(frame, isDrawing=IS_DRAWING)
    if isinstance(res, list):
        print(res[0])
        frame = res[1]
    else:
        print(res)
    #out.write(res[1])
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    _, frame = cap.read()
cap.release()
# out.release()
cv2.destroyAllWindows()

# frame = cv2.imread('videos/frame_all_wear.jpg')
# #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# res = detector.detect(frame)
# print(res[0])
# cv2.imshow('Frame', res[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()