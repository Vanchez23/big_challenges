from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import mediapipe as mp

global pose_estimation, switch
switch=1
pose_estimation=0



#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
API_CAMERA = 'rtsp://admin:camera12345@172.22.103.2/ch1-s1?tcp'

camera = cv2.VideoCapture(API_CAMERA)

def gen_frames():  # generate frame by frame from camera
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            success, frame = camera.read()
            if success:
                if(pose_estimation):
                    # success, frame = camera.read()
                    # Flip the image horizontally for a later selfie-view display, and convert
                    # the BGR image to RGB.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    frame.flags.writeable = False
                    results = pose.process(frame)

                    # Draw the pose annotation on the image.
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                try:
                    ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    pass

            else:
                pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if  request.form.get('pose_estimation') == 'Pose estimation':
            global pose_estimation
            pose_estimation=not pose_estimation


        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                print(switch)
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(API_CAMERA)
                switch = 1

    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')



if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     