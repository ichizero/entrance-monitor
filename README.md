# Entrance Monitor
Monitors labolatory entrance & notifies it.

## Environment
- Python 3
- dlib
- face_recognition
- Cloud Firestore
- DialogFlow

## Usage
```bash
$ docker-compose up -d
```

## Files
```bash
.
├── README.md
├── build_base_image        # to build face_recognition image with cuda driver
│   └── Dockerfile
├── data_store.py           # stores detected info to Cloud Firestore
├── docker-compose.yml      # service info
├── faces
│   ├── personA             # contains person image as .jpg .png
│   ├── personB
│   ├── ...
│   └── personN
├── face_recognizer_knn.py  # face recognizer
├── facerec_from_video.py   # recognize face with video
├── for_streaming           # streaming recognized webcam
│   ├── face_recognizer.py
│   ├── ip_camera.py
│   ├── streaming.py
│   └── templates
│       └── index.html
├── knn_train.py            # trains knn model with faces folder
├── line_notifier.py        # line notifier
├── monitoring.py           # main
├── people.py               # out of git, cilab people info
├── secret.env              # out of git, camera url & firestore , slack, line token
└── slack_notifier.py       # slack notofier
```

## How to add person
1. Writes person name in people.py.
1. Adds user to Cloud Firestore.
1. train knn model
```bash
$ docker exec -it face-recognition /bin/bash
$ python knn_train.py
```
