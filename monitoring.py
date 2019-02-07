""" Monitoring the entrance """
import sys
import os
from datetime import datetime
import argparse
import cv2

from face_recognizer_knn import FaceRecognizer
from slack_notifier import SlackNotifier
from line_notifier import LineNotifier
from data_store import DataStore
from people import cilab_people


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='monitoring')
    parser.add_argument('--url', type=str, default=os.environ['CAM_URL'], help='stream url')
    parser.add_argument('--face', type=str, default="./faces/trained_knn_model.clf", help='face dir path')
    parser.add_argument('--tol', type=float, default=0.4, help='tolerance of recognition')
    parser.add_argument('--slack', type=str, default=os.environ['SLACK_WEBHOOK_URL'], help='slack webhook url')
    parser.add_argument('--db', action='store_false')
    # parser.add_argument('--lt', type=str, default=os.environ['LINE_ACCESS_TOKEN'], help='line access token')
    # parser.add_argument('--lu', type=str, default=os.environ['LINE_USER_ID'], help='line user id')
    args = parser.parse_args()

    # Open video stream
    cam = cv2.VideoCapture(args.url)

    video_fps = cam.get(cv2.CAP_PROP_FPS)
    image_x = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_y = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("[log] Video info: {:.2f}fps {}x{}".format(video_fps, image_x, image_y))

    # Initialize
    face_recognizer = FaceRecognizer(args.face, args.tol)
    notifier = SlackNotifier(args.slack)
    # notifier = LineNotifier(args.lt, args.lu)
    data_store = DataStore()

    init_time = datetime(2000, 1, 1)
    detected_dict = dict(zip(cilab_people.keys(), [(init_time, 0) for i in range(len(cilab_people))]))
    recognized_dict = dict(zip(cilab_people.keys(), [init_time for i in range(len(cilab_people))]))

    read_error_count = (init_time, 0)

    # Monitoring
    print("[log] Start monitoring...")
    while True:
        frame_time = datetime.now()

        ret, img = cam.read()
        if not ret:
            error_time, error_count = read_error_count
            if (frame_time - error_time).seconds < 10:
                error_count += 1
                if error_count > 10:
                    cam.release()
                    error_message = "[log] Failed to read frames."
                    # notifier.send(error_message)
                    print(error_message)
                    sys.exit(1)
            else:
                error_time = frame_time
                error_count = 1
            read_error_count = (error_time, error_count)
            print("[error] Failed to read")
            continue

        face_names, pred_names, dists = face_recognizer.recognize(img)

        if face_names:
            detected_msg = ""
            for name, dist in zip(pred_names, dists):
                detected_msg += " {name} {dist:.2}".format(name=name, dist=dist[0])
            print("[detected] {date} {msg} "
                  .format(date=frame_time.strftime("%Y/%m/%d %H:%M:%S"),
                          msg=detected_msg))

        for name in face_names:
            if name is None:
                continue

            if name not in detected_dict:
                detected_dict[name] = (init_time, 0)

            detected_time, count = detected_dict[name]
            if (frame_time - detected_time).seconds < 1:
                count += 1
                detected_dict[name] = (detected_time, count)
                if count == 5:
                    recognized_time = recognized_dict[name]
                    if (frame_time - recognized_time).seconds > 120:
                        data_store.add(frame_time, name)
                        if name in cilab_people:
                            notifier.notify(frame_time, cilab_people[name])
                        else:
                            notifier.notify(frame_time, name)
                        print("[recognized] {date} {name}".format(date=frame_time.strftime("%H:%M"), name=name))
                    recognized_dict[name] = frame_time
            else:
                detected_dict[name] = (frame_time, 1)

    cam.release()
    print("[log] Exit")
