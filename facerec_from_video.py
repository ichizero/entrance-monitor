""" It based on https://github.com/ageitgey/face_recognition"""
import os
import argparse
import tqdm
import face_recognition
import cv2

from known_faces import load_faces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='facerec video')
    parser.add_argument('--video', help='video file path')
    parser.add_argument('--face', help='face dir path')
    args = parser.parse_args()

    # Open the input movie file
    input_movie = cv2.VideoCapture(args.video)

    video_frame = input_movie.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = input_movie.get(cv2.CAP_PROP_FPS)
    image_x = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_y = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an output movie file
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    output_file_name = os.path.splitext(args.video)[0] + '_result.mp4'
    output_movie = cv2.VideoWriter(output_file_name, fourcc, video_fps, (image_x, image_y))

    # progress
    progress = tqdm.tqdm(total=video_frame)
    progress.set_description("{:12}".format("Initialize"))

    # Load some sample pictures and learn how to recognize them.
    known_faces, known_names = load_faces(args.face)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    progress.set_description("{:12}".format("Processing"))
    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            name = None
            if match:
                name = known_names[match[0]]

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

        # Write the resulting image to the output video file
        progress.update(1)
        progress.set_postfix(timecode="{:.2f}s".format(frame_number / video_fps))
        output_movie.write(frame)

    # All done!
    input_movie.release()
    output_movie.release()
    progress.set_description("{:12}".format("Finished"))
    progress.close()
    print("Saved " + output_file_name)
