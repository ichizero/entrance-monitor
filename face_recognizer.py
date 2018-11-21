from pathlib import Path
import face_recognition
import cv2


class FaceRecognizer:
    def __init__(self, faces_path, tolerance=0.50):
        self.faces_path = faces_path
        self.tolerance = tolerance
        self.known_faces, self.known_names = self._load_faces()

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []

    def recognize(self, frame):
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="cnn")
        self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)

        face_names = []
        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.50)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            name = None
            for i, is_match in enumerate(match):
                if is_match:
                    name = self.known_names[i]

            face_names.append(name)

        self.face_names = face_names

        # Label the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        return frame

    def _load_faces(self):
        """ Loads known faces. """
        faces_path = Path(self.faces_path)

        img_path_list = list(faces_path.glob("*.png"))
        img_path_list.extend(list(faces_path.glob("*.jpg")))

        known_faces = []
        known_names = []

        for img_path in img_path_list:
            face_image = face_recognition.load_image_file(img_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_faces.append(face_encoding)
            known_names.append(img_path.stem)

        return (known_faces, known_names)