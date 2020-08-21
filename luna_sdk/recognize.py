import os
import subprocess
from time import time

import cv2

# VisionLabs
import FaceEngine as fe
import numpy as np

# PATHS
LUNA_SDK_PATH = "/home/emin/Documents/luna-sdk_ub1804_rel_v.3.8.8"
DATA_PATH = LUNA_SDK_PATH + "/data"
CONF_PATH = DATA_PATH + "/faceengine.conf"


class Recognizer:
    def __init__(self, threshold):
        """Initialize the value for threshold, objects for Face engine module and set for storing the people, who are already recognized by the system."""
        self.threshold = threshold
        # Create face engine
        self.faceEngine = fe.createFaceEngine(DATA_PATH, CONF_PATH)
        self.warper = self.faceEngine.createWarper()
        # Create extractor and descriptor for detected faces in the frame
        self.image_extractor = self.faceEngine.createExtractor()
        self.image_descriptor = self.faceEngine.createDescriptor()
        # Create matcher
        self.matcher = self.faceEngine.createMatcher()
        # Create descriptor to load the descriptors from the database
        self.loaded_descriptor = self.faceEngine.createDescriptor()
        # Create the set to store the names of people who are already recognized by the system
        self.recognized_people = set()

    def _detect_faces(self, _image_det):
        max_detections = 10
        detector_type = fe.DetectionType(1)
        detector = self.faceEngine.createDetector(fe.FACE_DET_DEFAULT)
        errors, detector_result = detector.detect(
            [_image_det], [_image_det.getRect()], max_detections, detector_type
        )
        return detector_result

    def _warp_faces(self, warper, image_det, _detection, landmarks):
        _transformation = warper.createTransformation(_detection, landmarks)
        warp_result = self.warper.warp(image_det, _transformation)
        if warp_result[0].isError:
            print("Failed image warping.")
            return None
        _warp_image = warp_result[1]
        return _warp_image

    def _sort_by_square(self, face):
        """Sort face by square."""
        return face.detection.rect.height * face.detection.rect.width

    def _compare_descriptors(self, loaded_descriptor_value):
        """Compare the descriptors of detected people and people from the database and return similarity in the form of the float."""
        print("Comparing descriptors...")
        # extract descriptor from dictionary value
        self.loaded_descriptor.load(loaded_descriptor_value, 1)
        result = self.matcher.match(self.image_descriptor, self.loaded_descriptor)
        return result.value.similarity

    def recognize(self, frame, descriptor_dictionary):
        """Recognize and return recognition result."""
        # convert frame to numpy array
        np_image = np.asarray(frame)
        # create FaceEngine img
        image = fe.Image()
        # push np img to FE img
        image.setData(np_image, fe.FormatType.R8G8B8)
        if not image.isValid():
            print("Image in FE format is not valid")

        face_names = []
        boxes = []

        # unpack detector result
        faces = self._detect_faces(image)[0]
        faces.sort(key=self._sort_by_square)

        start_time = time()

        for face in faces:
            # Default name
            name = "Unknown"

            detection = face.detection
            # check detector result is valid
            if (detection.rect.height < 1) and (detection.rect.width < 1):
                return

            # warp image
            warped_result = self._warp_faces(
                self.warper, face.img, detection, face.landmarks5_opt.value()
            )
            # extract descriptor from videostream
            ext = self.image_extractor.extractFromWarpedImage(
                warped_result, self.image_descriptor
            )
            for key, value in descriptor_dictionary.items():
                # identify person
                print("Check descriptor: {}".format(key))

                similarity = self._compare_descriptors(value)
                print(similarity)
                if similarity > self.threshold:
                    name = key
                    break
            face_names.append(name)
            box = face.detection.rect
            boxes.append(box)
            print(face_names)
        print("elapsed time: {:.4f}".format(time() - start_time))
        print()
        return (face_names, boxes)

    def play_video(self, face_names, vid_path):
        """Play videos corresponding to recognised people."""
        for name in face_names:
            if name not in self.recognized_people:
                self.recognized_people.add(name)
                p = subprocess.Popen(
                    [
                        "/usr/bin/ffplay",
                        "-autoexit",
                        "-vf",
                        "colorkey=green:0.3:0.2,fade=in:0:15",
                        os.path.join(vid_path, name + '.mp4'),
                    ]
                )
                p.wait()

    def draw_bounding_boxes(self, frame, face_names, boxes):
        """Draw boxes around recognized people."""
        for name, box in zip(face_names, boxes):
            print(box.x, box.y, box.x + box.width, box.y + box.height)
            cv2.rectangle(
                frame,
                (int(box.x), int(box.y)),
                (int(box.x + box.width), int(box.y + box.height)),
                (0, 0, 255),
                3,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                frame, name, (int(box.x), int(box.y - 10)), font, 1, (25, 240, 25), 2
            )
