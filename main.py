import os
import subprocess

import cv2
import face_recognition
import numpy as np

"""
This is a demo of running face recognition on live video from your webcam.
It's a little more complicated than the other example,
but it includes some basic performance tweaks to make things run a lot faster:
    1. Process each video frame at 1/4 resolution (though still display it at full resolution)
    2. Only detect faces in every other frame of video.
PLEASE NOTE: This example requires OpenCV (the `cv2` library)
to be installed only to read from your webcam.
OpenCV is *not* required to use the face_recognition library.
It's only required if you want to run this specific demo.
If you have trouble installing it, try any of the other demos that don't require it instead.
"""

PIC_PATH = "./pics/Robotics_Lab/"
VID_PATH = "./videos/Robotics_Lab/"
RATIO = 2.0  # Ratio by which the frame is resized to make the program work faster

REAL_SIM = True

# Get a reference to cameras
if REAL_SIM:
    video_capture = cv2.VideoCapture(2)  # RealSim camera
else:
    video_capture = cv2.VideoCapture(0)  # Default webcam

image_names = os.listdir(PIC_PATH)
image_names.sort()
N = len(image_names)  # Number of people

video_names = os.listdir(VID_PATH)
video_names.sort()

known_face_encodings = []
known_face_names = [0 for i in range(N)]

# Load a sample picture and learn how to recognize it.
for i in range(N):
    path_to_image = os.path.join(PIC_PATH, image_names[i])
    image = face_recognition.load_image_file(path_to_image)
    known_face_encodings.append(face_recognition.face_encodings(image)[0])
    name = os.path.basename(path_to_image)
    known_face_names[i] = os.path.splitext(name)[0]

print("Names: ", known_face_names)
print("known face encodings: ", len(known_face_encodings))
# Create arrays to remember the faces, that were already recognised by the camera
known_face_names_flags = [False for i in range(N)]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1.0 / RATIO, fy=1 / RATIO)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name == "Unknown":
                    break
                if not known_face_names_flags[best_match_index]:
                    p = subprocess.Popen(
                        [
                            "/usr/bin/ffplay",
                            "-autoexit",
                            os.path.join(VID_PATH, name + ".mp4"),
                        ]
                    )
                    known_face_names_flags[best_match_index] = True
            face_names.append(name)
    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= int(RATIO)
        right *= int(RATIO)
        bottom *= int(RATIO)
        left *= int(RATIO)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the camera
video_capture.release()
cv2.destroyAllWindows()
