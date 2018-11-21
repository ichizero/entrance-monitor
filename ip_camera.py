"""
VideoCamera

Author: @ichizero
"""


import face_recognition
import cv2

from known_faces import load_faces


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

        self.known_faces, self.known_names = load_faces(faces_path)

    def __del__(self):
        self.camera.release()

    def get_frame(self):
        """ Returns estimated image """
        _, frame = self.camera.read()

        face_locations = []
        face_encodings = []
        face_names = []

        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.50)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            name = None
            for i, is_match in enumerate(match):
                if is_match:
                    name = self.known_names[i]

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
