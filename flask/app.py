import sys
sys.path.append('.')

import numpy as np
import cv2
import rtsp
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, request

from DeepProtect import Detector

max_ticks = 25
cur_tick = 0
pose_estimation = 0
show_boxes = 0
reason = 0
REASON = {
    0: 'Всё хорошо',
    1: 'Нет людей в кадре',
    2: 'Больше чем 1 человек',
    3: 'Не хватает элемента спецодежды',
    4: 'Одежда не на человек',
}


res_ = 0

app = Flask(__name__, template_folder='./templates')
detector = Detector(path_to_model='/home/student/model/best.pt')
client = rtsp.Client(rtsp_server_uri='rtsp://admin:camera12345@172.22.103.2', verbose=True)


def gen_frames():
    global res_, max_ticks, cur_tick
    while client.isOpened():
        image = client.read(raw=True)
        res = detector.detect(image, isDrawing=show_boxes)
        frame = res[1] if show_boxes else image
        h, w, c = frame.shape

        top_pad = np.ones((80, w, c), dtype=np.uint8) * 200
        low_pad = np.ones((50, w, c), dtype=np.uint8) * 200
        frame = np.vstack([top_pad, frame, low_pad])

        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)

        if reason:
            reason_key = res[0].get('reason')
            res_ = REASON.get(reason_key)
            font = ImageFont.truetype("FreeMono.ttf", 30, encoding='UTF-8')
            x = (w - len(res_) * 10) // 2 - w // 12
            draw.text((x, h + 85), res_, font=font, fill="black")

        f_font = ImageFont.truetype("FreeMono.ttf", 60, encoding='UTF-8')
        finally_ = res[0].get('finally')  # статус
        cur_tick = 1 if finally_ else cur_tick

        if 0 < cur_tick <= max_ticks:
            draw.text((w // 3, 5), 'Комплектно', font=f_font, fill="green")
            cur_tick += 1
        else:
            draw.text((w // 3, 5), 'Некомплектно', font=f_font, fill="blue")
            cur_tick = 0

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


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global pose_estimation, show_boxes
    if request.method == 'POST':
        if request.form.get('show_boxes') == 'Отображение боксов/ключевых точек':
            global show_boxes
            show_boxes = not show_boxes

        elif request.form.get('reason') == 'Подробный отчёт':
            global reason
            reason = not reason

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html', num=show_boxes, reason=reason)


if __name__ == '__main__':
    app.run()