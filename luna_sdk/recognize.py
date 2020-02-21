import cv2
import subprocess
import os
# VisionLabs
import FaceEngine as fe
import numpy as np
from PIL import Image as PILImage

from time import time

# PATHS
luna_sdk_path = "/home/emin/Documents/luna-sdk_ub1804_rel_v.3.8.8"
data_path = luna_sdk_path + "/data"
conf_path = data_path + "/faceengine.conf"

# Create face engine
faceEngine = fe.createFaceEngine(data_path, conf_path)


def detect_faces(_image_det):
    max_detections = 10
    detector_type = fe.DetectionType(1)
    detector = faceEngine.createDetector(fe.FACE_DET_DEFAULT)
    errors, detector_result = detector.detect(
        [_image_det], [_image_det.getRect()], max_detections, detector_type
    )
    return detector_result


def warp_faces(warper, image_det, _detection, landmarks):
    _transformation = warper.createTransformation(_detection, landmarks)
    warp_result = warper.warp(image_det, _transformation)
    if warp_result[0].isError:
        print("Failed image warping.")
        return None
    _warp_image = warp_result[1]
    return _warp_image


# sorting face by square
def sort_by_square(face):
    return face.detection.rect.height * face.detection.rect.width


# identify person, get fe.image, person_id, return identification result bool
def compare_descriptors(
    image,
    image_extractor,
    matcher,
    image_descriptor,
    loaded_descriptor,
    loaded_descriptor_value,
):
    # create extractor and mathcer
    print("Comparing descriptors...")
    # create descriptors
    # extract descriptor from videostream
    ext = image_extractor.extractFromWarpedImage(image, image_descriptor)
    # extract descriptor from dictionary value
    loaded_descriptor.load(loaded_descriptor_value, 1)
    result = matcher.match(image_descriptor, loaded_descriptor)
    return result.value.similarity


# Recognize function get ros msg image and person_id return recognition result
def recognize(frame, known_names, descriptor_dictionary, threshold=0.8):
    face_names = []
    best_indexes = []
    # unpack detector result
    faces = detect_faces(frame)[0]
    faces.sort(key=sort_by_square)

    warper = faceEngine.createWarper()

    extractor = faceEngine.createExtractor()
    matcher = faceEngine.createMatcher()

    image_descriptor = faceEngine.createDescriptor()
    loaded_descriptor = faceEngine.createDescriptor()

    
    start_time = time()
    for face in faces:
        # Default name
        name = "Unknown"

        detection = face.detection
        # check detector result is valid
        if (detection.rect.height < 1) and (detection.rect.width < 1):
            return

        # warp image
        warped_result = warp_faces(warper, face.img, detection, face.landmarks5_opt.value())

        for key, value in descriptor_dictionary.items():
            # identify person
            print("Check descriptor {}".format(key))

            similarity = compare_descriptors(
                warped_result,
                extractor,
                matcher,
                image_descriptor,
                loaded_descriptor,
                value
            )
            print(similarity)
            if similarity > threshold:
                name = key
                best_indexes.append(known_names.index(name))
                break
        face_names.append(name)

    print("elapsed time: {:.4f}".format(time() - start_time))
    return (face_names, best_indexes)


def play_video(known_face_names_flags, known_face_names, best_indexes, vid_path):
    """Play videos corresponding to recognised people."""
    for index in best_indexes:
        if not known_face_names_flags[index]:
            name = known_face_names[index]
            p = subprocess.Popen(
                ["/usr/bin/ffplay", "-autoexit", os.path.join(vid_path, name + ".mp4")]
            )
            known_face_names_flags[index] = True