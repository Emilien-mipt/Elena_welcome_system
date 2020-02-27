import json
import os
import subprocess
from time import time

import cv2
# VisionLabs
import FaceEngine as fe
import numpy as np

from extract_descriptors import Database_creator
from recognize import Recognizer

# PATHS
luna_sdk_path = "/home/emin/Documents/luna-sdk_ub1804_rel_v.3.8.8"
data_path = luna_sdk_path + "/data"
conf_path = data_path + "/faceengine.conf"
face_image_path = "../pics/Robotics_Lab/"
video_path = "../videos/Robotics_Lab/"

N_FRAMES = 5 # Process only every N_FRAMES frame

database = Database_creator()
recognizer = Recognizer(threshold = 0.8)

def main():
    # Get image names and sort them
    image_names = os.listdir(face_image_path)
    image_names.sort()

    # Get video names and sort them
    video_names = os.listdir(video_path)
    video_names.sort()

    # Get names of the people from uploaded images and their face encodings
    known_face_names = database.get_known_names(image_names)
    # Create arrays to remember the faces, that were already recognised by the camera
    known_face_names_flags = recognizer.define_known_flags_array(image_names)

    assert(len(known_face_names) == len(known_face_names_flags))

    start_time = time()
    # Load dictionary with descriptors
    descriptors_dict = database.get_descriptors(image_names)
    #print(descriptors_dict)
    print("Time for database creation: {:.4f}".format(time() - start_time))

    video_capture = cv2.VideoCapture(0)
    process = True
    frame = np.zeros((480, 640, 3))
    count_frames = 0

    while process:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        count_frames += 1
        if not ret:
            print("Could not read frame!")
            process = False
            break

        # convert frame to numpy array
        np_image = np.asarray(frame)
        # create FaceEngine img
        image = fe.Image()
        # push np img to FE img
        image.setData(np_image, fe.FormatType.R8G8B8)
        if not image.isValid():
            continue

        if count_frames%N_FRAMES == 0:
            face_names, best_match_indexes = recognizer.recognize(
                image, known_face_names, descriptors_dict
            )
            print(face_names)
            print()
            recognizer.play_video(
                known_face_names, best_match_indexes, video_path
            )
            count_frames = 0

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
