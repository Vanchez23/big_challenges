import cv2
import mediapipe as mp
import traceback

WIDTH = 1280
HEIGHT = 720

class PoseEstimator():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.points = [0, 12, 11, 24, 23, 20, 21, 19, 18, 16, 22, 15, 17, 25, 26]

    def detect(self, image, isDrawing=True):
        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity = 2) as pose:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            try:
                coords = results.pose_landmarks.landmark
            except AttributeError as e:
                #print(e)
                # print('error')
                return None
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
            if isDrawing:
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        ret = {'shield': [(coords[0].x * WIDTH, coords[0].y * HEIGHT),
                          (coords[9].x * WIDTH, coords[9].y * HEIGHT)],

               'jacket': [(coords[12].x * WIDTH, coords[12].y * HEIGHT),
                          (coords[23].x * WIDTH, coords[23].y * HEIGHT)],

               'left_glove': [(coords[21].x * WIDTH, coords[21].y * HEIGHT),
                              (coords[19].x * WIDTH, coords[19].y * HEIGHT),
                              (coords[15].x * WIDTH, coords[15].y * HEIGHT),
                              (coords[17].x * WIDTH, coords[17].y * HEIGHT)],

               'right_glove': [(coords[22].x * WIDTH, coords[22].y * HEIGHT),
                                       (coords[20].x * WIDTH, coords[20].y * HEIGHT),
                                       (coords[16].x * WIDTH, coords[16].y * HEIGHT),
                                       (coords[18].x * WIDTH, coords[18].y * HEIGHT)],

               'pants': [(coords[25].x * WIDTH, coords[25].y * HEIGHT),
                         (coords[26].x * WIDTH, coords[26].y * HEIGHT)]}
        if isDrawing:
            return [ret, image]
        else:
            return ret


#0,12,11,24,23,20,21,19,18,16,22,15,17,25,26
#нос,плечи, таз,кисти,колени