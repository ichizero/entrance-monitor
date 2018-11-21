""" It based on https://github.com/ageitgey/face_recognition"""
import os
import argparse
import tqdm
import cv2

from face_recognizer import FaceRecognizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='facerec video')
    parser.add_argument('--video', help='video file path')
    parser.add_argument('--face', help='face dir path')
    parser.add_argument('--tol', type=float, default=0.50, help='tolerance of recognition')
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

    face_recognizer = FaceRecognizer(args.face, args.tol)

    frame_number = 0

    progress.set_description("{:12}".format("Processing"))
    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        res_img = face_recognizer.recognize(frame)

        # Write the resulting image to the output video file
        progress.update(1)
        progress.set_postfix(timecode="{:.2f}s".format(frame_number / video_fps))
        output_movie.write(res_img)

    # All done!
    input_movie.release()
    output_movie.release()
    progress.set_description("{:12}".format("Finished"))
    progress.close()
    print("Saved " + output_file_name)
