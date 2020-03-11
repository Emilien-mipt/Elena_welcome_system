import argparse
import copy
import os
import threading
from time import time

import cv2
import numpy as np

from extract_descriptors import Database_creator
from recognize import Recognizer

DELAY_TIME = 1.0

database = Database_creator()
recognizer = Recognizer(threshold=0.9)
lock = threading.RLock()

def play_video(face_names, video_path):
    """Enable video playing and disable timer."""
    global timer_is_running
    recognizer.play_video(face_names, video_path)
    with lock:
        timer_is_running = False


def main():
    # Parse paths to pics and corresponding videos with greetings to create a database
    face_image_path = p.path_photos
    video_path = p.path_videos

    # Global variable to check whether timer is running or not
    global timer_is_running
    timer_is_running = False

    # Get image names
    image_names = os.listdir(face_image_path)

    # Get paths to images
    paths_to_images = [os.path.join(os.path.abspath(face_image_path), name) for name in image_names]

    # Get names of the people from uploaded images and their face encodings
    known_face_names = database.get_known_names(image_names)

    # Get video names
    video_names = os.listdir(video_path)

    start_time = time()

    # Create the dictionary with descriptors
    descriptors_dict = database.get_descriptors(paths_to_images)

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

        # Recognize and find faces and locations
        (face_names, boxes) = recognizer.recognize(frame, descriptors_dict)
        # Draw boxes
        recognizer.draw_bounding_boxes(frame, face_names, boxes)

        if len(face_names) == 0:
            pass
        else:
            with lock:
                if timer_is_running is False:
                    timer_is_running = True
                    # Play video after some time
                    t = threading.Timer(
                        DELAY_TIME,
                        play_video,
                        (copy.deepcopy(face_names), video_path,),
                    )
                    t.start()

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process paths to photos of people and corresponding videos"
    )
    parser.add_argument(
        "--photos",
        dest="path_photos",
        default="../pics/Robotics_Lab/",
        help="Path to photos for descriptor extraction and formation of the database",
    )
    parser.add_argument(
        "--videos",
        dest="path_videos",
        default="../videos/Robotics_Lab/",
        help="Path to videos with greetings",
    )
    p = parser.parse_args()
    main()
