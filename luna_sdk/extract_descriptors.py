import json
import os

# VisionLabs
import FaceEngine as fe
import numpy as np
from PIL import Image as PILImage

# PATHS
luna_sdk_path = "/home/emin/Documents/luna-sdk_ub1804_rel_v.3.8.8"
# package_path = os.path.abspath("./")
data_path = luna_sdk_path + "/data"
conf_path = data_path + "/faceengine.conf"
face_image_path = "../pics/Robotics_Lab/"

faceEngine = fe.createFaceEngine(data_path, conf_path)


def get_known_names(image_names):
    known_face_names = []
    N = len(image_names)
    for i in range(N):
        # Get names
        name = os.path.splitext(image_names[i])[0]
        known_face_names.append(name)
    assert(len(image_names) == len(known_face_names))
    print("Names in the database: {}".format(known_face_names))
    return known_face_names


def detect_face(_image_det):
    detector_type = fe.DetectionType(1)
    detector = faceEngine.createDetector(fe.FACE_DET_DEFAULT)
    errors, detector_result = detector.detectOne(
        _image_det, _image_det.getRect(), detector_type
    )
    return detector_result


def warp_faces(image_det, _detection, landmarks):
    warper = faceEngine.createWarper()
    _transformation = warper.createTransformation(_detection, landmarks)
    warp_result = warper.warp(image_det, _transformation)
    if warp_result[0].isError:
        print("Failed image warping.")
        return None
    _warp_image = warp_result[1]
    return _warp_image


def extract_from_photo(image_name):
    # create extractor and matcher
    extractor = faceEngine.createExtractor()
    descriptor = faceEngine.createDescriptor()
    # load image from file
    path_to_image = os.path.join(face_image_path, image_name)
    img = PILImage.open(path_to_image)
    img.load()
    # convert image to numpy array
    data = np.asarray(img)
    np_image = np.asarray(data)
    # create FaceEngine image
    image = fe.Image()
    # push np img to FE img
    err = image.setData(np_image, fe.FormatType.R8G8B8)
    # check image is valid
    if not image.isValid():
        print("Image error = ", err)
    # unpack detector result
    face = detect_face(image)
    detection = face.detection
    # check detector result is valid
    if (detection.rect.height < 1) and (detection.rect.width < 1):
        print("no face")
        return
    # warp image
    warped_result = warp_faces(face.img, detection, face.landmarks5_opt.value())
    # extract descrptor
    ext = extractor.extractFromWarpedImage(warped_result, descriptor)
    err, desc = descriptor.save()
    return desc


def get_descriptors(image_names, known_names):
    people_descriptors = {}
    N = len(image_names)
    print("Extracting the descriptors from the database...")
    for i in range(N):
        people_descriptors[known_names[i]] = extract_from_photo(image_names[i])
    assert(len(people_descriptors) == len(image_names))
    print("Extracted the descriptors sucessfuly.")
    return people_descriptors

