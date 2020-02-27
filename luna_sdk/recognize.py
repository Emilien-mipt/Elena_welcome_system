import os
import subprocess
from time import time

# VisionLabs
import FaceEngine as fe

# PATHS
luna_sdk_path = "/home/emin/Documents/luna-sdk_ub1804_rel_v.3.8.8"
data_path = luna_sdk_path + "/data"
conf_path = data_path + "/faceengine.conf"


class Recognizer:
    def __init__(self, threshold):
        self.threshold = threshold
        # Create face engine
        self.faceEngine = fe.createFaceEngine(data_path, conf_path)
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

    # sorting face by square
    def _sort_by_square(self, face):
        return face.detection.rect.height * face.detection.rect.width

    # identify person, get fe.image, person_id, return identification result bool
    def _compare_descriptors(self, loaded_descriptor_value):
        # create extractor and mathcer
        print("Comparing descriptors...")
        # create descriptors
        # extract descriptor from dictionary value
        self.loaded_descriptor.load(loaded_descriptor_value, 1)
        result = self.matcher.match(self.image_descriptor, self.loaded_descriptor)
        return result.value.similarity

    # Recognize and return recognition result
    def recognize(self, frame, descriptor_dictionary):
        face_names = []

        # unpack detector result
        faces = self._detect_faces(frame)[0]
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
            ext = self.image_extractor.extractFromWarpedImage(warped_result, self.image_descriptor)
            for key, value in descriptor_dictionary.items():
                # identify person
                print("Check descriptor: {}".format(key))

                similarity = self._compare_descriptors(value)
                print(similarity)
                if similarity > self.threshold:
                    name = key
                    break
            face_names.append(name)
            print(face_names)            
        print("elapsed time: {:.4f}".format(time() - start_time))
        print()
        return face_names

    def play_video(self, face_names, vid_path):
        """Play videos corresponding to recognised people."""
        for name in face_names:
            if name not in self.recognized_people:
                self.recognized_people.add(name)
                p = subprocess.Popen(
                    [
                        "/usr/bin/ffplay",
                        "-autoexit",
                        os.path.join(vid_path, name + ".mp4"),
                    ]
                )
                p.wait()
