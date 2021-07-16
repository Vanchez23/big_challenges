import cv2
import mediapipe as mp
from DeepProtect import Wear
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

WIDTH = 640
HEIGHT = 360



# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)
    coords = results.pose_landmarks.landmark
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
    wear = Wear(points = ret)
    wear.point_check(box = )



    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

