from flask import Flask, Response
import cv2
app = Flask(__name__)
video = cv2.VideoCapture(0)
@app.route('/')
def index():
    return "Default Message"
def gen(video):
    while True:
        success, image = video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    # mp_drawing = mp.solutions.drawing_utils
    # mp_pose = mp.solutions.pose
    # with mp_pose.Pose( min_detection_confidence=0.5,
    # min_tracking_confidence=0.5) as pose:
    #     while video.isOpened():
    #         success, image = video.read()
    #         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    #         image.flags.writeable = False
    #         results = pose.process(image)
    #
    #         # Draw the pose annotation on the image.
    #         image.flags.writeable = True
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         mp_drawing.draw_landmarks(
    #             image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    #
    #         success, image = video.read()
    #         ret, jpeg = cv2.imencode('.jpg', image)
    #         frame = jpeg.tobytes()
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run()
