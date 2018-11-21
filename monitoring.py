""" """
import os
from datetime import datetime
import argparse
import cv2

from face_recognizer import FaceRecognizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='facerec video')
    parser.add_argument('--url', help='stream url')
    parser.add_argument('--face', help='face dir path')
    parser.add_argument('--tol', type=float, default=0.50, help='tolerance of recognition')
    args = parser.parse_args()

    # Open the input movie file
    cam = cv2.VideoCapture(args.video)

    video_fps = cam.get(cv2.CAP_PROP_FPS)
    image_x = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_y = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("[log] video info: {}fps {} x {}".format(video_fps, image_x, image_y))

    face_recognizer = FaceRecognizer(args.face, args.tol)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        res_img = face_recognizer.recognize(frame)

        for name in face_recognizer.face_names:
            print("[detected] {date} {name}"
                  .format(date=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                          name=face_recognizer.face_names))

    cam.release()
    print("[log] exit")
