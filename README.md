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
$ docker-compose restart
```

## env file
### secret.env
- LINE_ACCESS_TOKEN
- LINE_USER_ID
- SLACK_WEBHOOK_URL
- CAM_URL
- FIRESTORE_PROJECT_ID
- FIRESTORE_KEY_ID
- FIRESTORE_KEY
- FIRESTORE_CLIENT_EMAIL
- FIRESTORE_CLIENT_ID
- FIRESTORE_CERT_URL

### proxy.env
http_proxy
https_proxy
