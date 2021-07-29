import sys
sys.path.append('.')

import numpy as np
import cv2
import rtsp
import yaml
import copy
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, Response, request

from DeepProtect import Detector

app = Flask(__name__)

def gen_frames():
    while client.isOpened():
        image = client.read(raw=True)
        res = detector.detect(image, isDrawing=cfg['APP']['SHOW_BOXES'])
        frame = res[1] if cfg['APP']['SHOW_BOXES'] else image
        h, w, c = frame.shape

        top_pad = np.ones((80, w, c), dtype=np.uint8) * 200
        low_pad = np.ones((50, w, c), dtype=np.uint8) * 200
        frame = np.vstack([top_pad, frame, low_pad])

        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)

        if cfg['APP']['SHOW_REASON']:
            reason_key = res[0].get('reason')
            reason = cfg['APP']['REASON_INFO'].get(reason_key)
            font = ImageFont.truetype("FreeMono.ttf", 30, encoding='UTF-8')
            x = (w - len(reason) * 10) // 2 - w // 12
            draw.text((x, h + 85), reason, font=font, fill="black")

        f_font = ImageFont.truetype("FreeMono.ttf", 60, encoding='UTF-8')
        finally_ = res[0].get('finally')  # статус
        cfg['APP']['CUR_TICK'] = 1 if finally_ else cfg['APP']['CUR_TICK']

        if 0 < cfg['APP']['CUR_TICK'] <= cfg['APP']['MAX_TICKS']:
            draw.text((w // 3, 5), 'Комплектно', font=f_font, fill="green")
            cfg['APP']['CUR_TICK'] += 1
        else:
            draw.text((w // 3, 5), 'Некомплектно', font=f_font, fill="blue")
            cfg['APP']['CUR_TICK'] = 0

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
    if request.method == 'POST':
        if request.form.get('show_boxes') == 'Отображение боксов/ключевых точек':
            cfg['APP']['SHOW_BOXES'] = not cfg['APP']['SHOW_BOXES']

        elif request.form.get('reason') == 'Подробный отчёт':
            cfg['APP']['SHOW_REASON'] = not cfg['APP']['SHOW_REASON']

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html', num=cfg['APP']['SHOW_BOXES'], reason=cfg['APP']['SHOW_REASON'])

def main(cfg_path):

    global detector, client, cfg
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    app.template_folder = cfg['APP']['TEMPLATE_FOLDER']
    detector = Detector(path_to_model=cfg['SYSTEM']['CLOTHES_MODEL_WEIGHTS_PATH'],
                        hrnet_config_path=cfg['SYSTEM']['KEYP_MODEL_CFG_PATH'])
    client = rtsp.Client(rtsp_server_uri=cfg['APP']['RTSP_URI'], verbose=True)

    cfg['APP']['CUR_TICK'] = 0
    app.run()

if __name__ == '__main__':
    import sys
    config_name = 'flask_app/app_config.yaml'
    argv = sys.argv
    if len(argv) <= 2:
        config_name = argv[1]
    main(config_name)
