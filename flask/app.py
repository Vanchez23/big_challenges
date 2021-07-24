from flask import Flask, render_template, Response, request
import cv2
import rtsp
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import sys
sys.path.insert(0,'..')
import mediapipe as mp
from DeepProtect import Detector
from PIL import Image, ImageDraw, ImageFont

pose_estimation=0
show_boxes = 0
reason = 0
REASON = {0:'Всё хорошо',
          1:'Нет людей в кадре',
          2:'Больше чем 1 человек',
          3:'Не хватает элемента спецодежды',
          4:'Одежда не на человек'}

# REASON = {0: 'It s okay',
#           1: 'There are no people in the frame',
#           2: 'More than 1 person',
#           3: 'A piece of workwear is missing',
#           4: 'Clothes not per person'}
res_ = 0
#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
detector = Detector(path_to_model = '/home/student/model/best.pt')
client = rtsp.Client(rtsp_server_uri = 'rtsp://admin:camera12345@172.22.103.2', verbose=True)


def gen_frames():  # generate frame by frame from camera
    global res_
    while client.isOpened():
        image = client.read(raw=True)
        res = detector.detect(image, isDrawing= show_boxes)
        frame = res[1] if show_boxes else image
        h, w, c = frame.shape

        top_pad = np.ones((80, w, c), dtype=np.uint8) * 200
        low_pad = np.ones((50, w, c), dtype=np.uint8) * 200
        frame = np.vstack([top_pad, frame, low_pad])

        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        f_font = ImageFont.truetype("FreeMono.ttf", 60, encoding='UTF-8')
        finally_ = res[0].get('finally')  # статус

        if finally_:
            draw.text((w // 3, 5), 'Комплектно', font=f_font, fill="green")
        else:
            draw.text((w // 3, 5), 'Некомплектно', font=f_font, fill="blue")

        frame = np.array(frame)

        if(reason):
            res = detector.detect(image, isDrawing=show_boxes)
            frame = res[1] if show_boxes else image
            reason_ = res[0].get('reason')

            finally_ = res[0].get('finally') # статус

            res_ =  REASON.get(reason_)

            h, w, c = frame.shape
            top_pad = np.ones((80, w, c), dtype=np.uint8) * 200
            low_pad = np.ones((50, w, c), dtype=np.uint8) * 200
            frame = np.vstack([top_pad, frame, low_pad])


            frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame)
            font = ImageFont.truetype("FreeMono.ttf", 30, encoding='UTF-8')
            f_font = ImageFont.truetype("FreeMono.ttf", 60, encoding='UTF-8')


            if finally_:
                draw.text((w//3, 10), 'Комплектно', font=f_font, fill="green")
            else:
                draw.text((w//3, 10), 'Некомплектно', font=f_font, fill="blue")

            x = (w - len(res_) * 10) // 2 - w // 12
            draw.text((x, h+85), res_, font=font, fill="black")
            frame = np.array(frame)

        try:
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
    global pose_estimation, show_boxes
    if request.method == 'POST':
        if  request.form.get('show_boxes') == 'Отображение боксов/ключевых точек':
            global show_boxes
            show_boxes=not show_boxes

        elif  request.form.get('reason') == 'Подробный отчёт':
            global reason
            reason=not reason


    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html', num = show_boxes,reason = reason)



if __name__ == '__main__':
    app.run()