""" known_faces.py """
from pathlib import Path
import face_recognition


def load_faces(img_path):
    """ Loads known faces. """
    faces_path = Path(img_path)

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
