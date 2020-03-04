import json
import os
import subprocess
from time import time
import time as tsl

import cv2

# VisionLabs
import FaceEngine as fe
import numpy as np
import copy
from extract_descriptors import Database_creator
from recognize import Recognizer
import threading
import glob

RATIO = 2.

# PATHS
luna_sdk_path = "/home/emin/Documents/luna-sdk_ub1804_rel_v.3.8.8"
data_path = luna_sdk_path + "/data"
conf_path = data_path + "/faceengine.conf"

# Paths to pics and videos from the dataset
# face_image_path = "../pics/Robotics_Lab/"
face_image_path = "../pics/Government/"
face_names_path = "../pics/"
# video_path = "../videos/Robotics_Lab/"
video_path = "../videos/Government/"
video_jokes_path = "../videos/Jokes/*.*"
# video_phrases_path = "../videos/Phrases"
import csv

N_FRAMES = 5  # Process only every N_FRAMES frame

database = Database_creator()
recognizer = Recognizer(threshold=0.9)

time_define = True
face_list = []
# -------------------------------------------------------
timer_time = 2
timer_is_running = False


def planing(
    face_names, video_path, joke_files=list, face_list=list
):
    global timer_is_running
    recognizer.planing(
        face_names, video_path, joke_files, face_list
    )
    timer_is_running = False


# -------------------------------------------------------
def openPersonNames(file):
    with open(file, "r") as fp:
        reader = csv.reader(fp)
        data_read = [row[0] for row in reader]
        return data_read


def main():
    global timer_is_running
    global time_define
    global timer_time
    global face_list
    timer_is_running = False
    # Get image names and sort them
    image_names = os.listdir(face_image_path)
    joke_files = glob.glob(video_jokes_path)
    # Get video names and sort them
    video_names = os.listdir(video_path)
    face_list = openPersonNames(os.path.join(face_names_path, "names.txt"))
    print("Number of images: {}".format(len(image_names)))
    print("Number of videos: {}".format(len(video_names)))

    # Get names of the people from uploaded images and their face encodings
    known_face_names = database.get_known_names(image_names)

    sorted_face_list = sorted(face_list[1:])
    sorted_known_names = sorted(known_face_names)
    for entry in face_list[1:]:
        if  not entry in known_face_names:
            pass
    print("-----------------------------TEST------------------------", sorted_face_list == sorted_known_names)

    start_time = time()
    # Load dictionary with descriptors
    descriptors_dict = database.get_descriptors(image_names)
    # print(descriptors_dict)
    print("Time for database creation: {:.4f}".format(time() - start_time))

    video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture("/tmp/video1")
    process = True
    frame = np.zeros((480, 640, 3))
    count_frames = 0

    while process:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        tsl.sleep(0.1)
        count_frames += 1
        if not ret:
            print("Could not read frame!")
            process = False
            break

        # Recognize and find faces and locations
        (face_names, boxes) = recognizer.recognize(frame, descriptors_dict)
        # Draw boxes
        recognizer.draw_bounding_boxes(frame, face_names, boxes)
        if len(face_names) == 0:
            pass
        else:
            # Process every N_FRAMES frame
            if time_define == True:
                if timer_is_running == False:
                    timer_is_running = True
                    t = threading.Timer(
                        timer_time,
                        planing,
                        (
                            copy.deepcopy(face_names),
                            video_path,
                            joke_files,
                            face_list,
                        ),
                    )
                    t.start()
            else:
                if count_frames % N_FRAMES == 0:
                    recognizer.planing(
                        face_names,
                        video_path,
                        joke_files,
                        face_list,
                    )
                    count_frames = 0
        
        small_frame = cv2.resize(frame, (0, 0), fx=1.0 / RATIO, fy=1 / RATIO)
        cv2.imshow("frame", small_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
