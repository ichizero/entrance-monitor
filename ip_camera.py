"""
VideoCamera

Author: @ichizero
"""


import face_recognition
import cv2

from face_recognizer import FaceRecognizer


class IPCamera:
    """
    Face recognition for IPCamera
    """

    def __init__(self, url, faces_path):
        self.url = url

        # load stream video
        self.camera = cv2.VideoCapture(self.url)
        if self.camera.isOpened() is False:
            print("Error: Failed to open the video source.")

        self.video_frame = self.camera.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.image_x = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_y = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.face_recognizer = FaceRecognizer(faces_path)

    def __del__(self):
        self.camera.release()

    def get_frame(self):
        """ Returns estimated image """
        _, frame = self.camera.read()

        res_img = self.face_recognizer.recognize(frame)

        _, jpeg = cv2.imencode('.jpg', res_img)
        return jpeg.tobytes()
