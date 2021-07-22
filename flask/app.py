from flask import Flask, render_template, Response, request
import cv2
import time

import datetime, time
import os, sys
import numpy as np
from threading import Thread
import sys
sys.path.insert(0,'..')
import mediapipe as mp
from DeepProtect import Detector


global pose_estimation, switch
switch=1
pose_estimation=0



#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
API_CAMERA = 'rtsp://admin:camera12345@172.22.103.2'
detector = Detector(path_to_model = '/home/student/model/best.pt')

camera = cv2.VideoCapture(API_CAMERA)

def gen_frames():  # generate frame by frame from camera
    while camera.isOpened():
        success, frame = camera.read()
        # frame = cv2.resize(frame,(640,370))
        if success:
            if(pose_estimation):
                camera.grab()
                res = detector.detect(frame, isDrawing=True)
                if isinstance(res, list):
                    frame = res[1]
                else:
                    if res == {'num_people': 1, 'all_wear': True, 'finally': True}:
                        print('green')

            try:
                # ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(e)



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
                # print(switch)
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