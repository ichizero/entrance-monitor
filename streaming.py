"""
Streams video with pose estimation.

Author: @ichizero
"""

import argparse
from flask import Flask, render_template, Response
from ip_camera import IPCamera

app = Flask(__name__)

g_url = ""
g_faces_path = ""

@app.route('/')
def index():
    """ Render Template """
    return render_template('index.html')


def recognize(camera):
    """ Recognize face """
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    """ Feeds video """
    return Response(recognize(IPCamera(url=g_url, faces_path=g_faces_path)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Recognition')
    parser.add_argument('--url', help='stream url')
    parser.add_argument('--face', help='face dir path')
    args = parser.parse_args()

    g_url = args.url
    g_faces_path = args.face

    app.run(host='0.0.0.0', port=5001, debug=True)
