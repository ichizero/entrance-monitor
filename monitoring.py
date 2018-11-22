""" """
import os
from datetime import datetime
import argparse
import cv2

from face_recognizer import FaceRecognizer
from slack_notifier import SlackNotifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='monitoring')
    parser.add_argument('--url', type=str, help='stream url')
    parser.add_argument('--face', type=str, help='face dir path')
    parser.add_argument('--tol', type=float, default=0.50, help='tolerance of recognition')
    parser.add_argument('--slack', type=str, help='slack webhook url')
    args = parser.parse_args()

    # Open the input movie file
    cam = cv2.VideoCapture(args.url)

    video_fps = cam.get(cv2.CAP_PROP_FPS)
    image_x = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_y = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("[log] video info: {:.2f}fps {}x{}".format(video_fps, image_x, image_y))

    face_recognizer = FaceRecognizer(args.face, args.tol)
    notifier = SlackNotifier(args.slack)

    detected_dict = {}
    recognized_dict = {}
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[log] failed to read frame")
            continue

        frame_time = datetime.now()

        res_img = face_recognizer.recognize(frame)

        if face_recognizer.face_names:
            print("[detected] {date} {name}"
                  .format(date=frame_time.strftime("%Y/%m/%d %H:%M:%S"),
                          name=face_recognizer.face_names))

        for name in face_recognizer.face_names:
            if name is None:
                continue

            if name in detected_dict:
                detected_time, count = detected_dict[name]
                if (frame_time - detected_time).seconds < 1:
                    count += 1
                    detected_dict[name] = (detected_time, count)
                    if count == 5:
                        if name in recognized_dict:
                            recognized_time = recognized_dict[name]
                            if (frame_time - recognized_time).seconds > 120:
                                notifier.notify("{date} {name} さんが入室しました。"
                                                .format(date=frame_time.strftime("%Y/%m/%d %H:%M"),
                                                        name=name))
                                print("[recognized] {date} {name}"
                                    .format(date=frame_time.strftime("%Y/%m/%d %H:%M"),
                                            name=name))
                            

                        else:
                            notifier.notify("{date} {name} さんが入室しました。"
                                            .format(date=frame_time.strftime("%Y/%m/%d %H:%M"),
                                                    name=name))
                            print("[recognized] {date} {name}"
                                .format(date=frame_time.strftime("%Y/%m/%d %H:%M"),
                                        name=name))
                        recognized_dict[name] = frame_time
                else:
                    detected_dict[name] = (frame_time, 1)
            else:
                detected_dict[name] = (frame_time, 1)
            

    cam.release()
    print("[log] exit")
