import math
import pickle
from sklearn import neighbors
import face_recognition
import cv2


class FaceRecognizer:
    def __init__(self, model_path, distance_threshold=0.4):
        with open(model_path, 'rb') as f:
            self.knn_clf = pickle.load(f)

        self.distance_threshold = distance_threshold

        self.face_locations = []
        self.face_names = []

    def recognize(self, frame):
        """
        Recognizes faces in given image using a trained KNN classifier
        """
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = self.knn_clf.kneighbors(face_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= self.distance_threshold for i in range(len(self.face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        face_names = []
        for pred_name, is_match in zip(self.knn_clf.predict(face_encodings), are_matches):
            if is_match:
                face_names.append(pred_name)
            else:
                face_names.append(None)
        self.face_names = face_names
