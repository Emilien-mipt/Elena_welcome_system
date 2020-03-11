import os

# VisionLabs
import FaceEngine as fe
import numpy as np
from PIL import Image as PILImage

# PATHS
LUNA_SDK_PATH = "/home/emin/Documents/luna-sdk_ub1804_rel_v.3.8.8"
DATA_PATH = LUNA_SDK_PATH + "/data"
CONF_PATH = DATA_PATH + "/faceengine.conf"

face_image_path = "../pics/Robotics_Lab/"

# The class that creates the database with descriptors from the pictures in the database
class Database_creator:
    def __init__(self):
        self.faceEngine = fe.createFaceEngine(DATA_PATH, CONF_PATH)
        # Create warper
        self.warper = self.faceEngine.createWarper()
        # Create extractor and descriptor
        self.extractor = self.faceEngine.createExtractor()
        self.descriptor = self.faceEngine.createDescriptor()

        self.known_face_names = []
        self.people_descriptors = {}

    def get_known_names(self, image_names):
        """Form the list of names of people from the dataset."""
        N = len(image_names)
        for i in range(N):
            # Get names
            name = os.path.splitext(image_names[i])[0]
            self.known_face_names.append(name)
        assert len(image_names) == len(self.known_face_names)
        print("Names in the database: {}".format(self.known_face_names))
        print("Number of people in the dataset: {}".format(N))
        return self.known_face_names

    def _detect_face(self, _image_det):
        detector_type = fe.DetectionType(1)
        detector = self.faceEngine.createDetector(fe.FACE_DET_DEFAULT)
        errors, detector_result = detector.detectOne(
            _image_det, _image_det.getRect(), detector_type
        )
        return detector_result

    def _warp_faces(self, image_det, _detection, landmarks):
        _transformation = self.warper.createTransformation(_detection, landmarks)
        warp_result = self.warper.warp(image_det, _transformation)
        if warp_result[0].isError:
            print("Failed image warping.")
            return None
        _warp_image = warp_result[1]
        return _warp_image

    def _extract_from_photo(self, image_name):
        """Extract the descriptor from single photo."""
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
        face = self._detect_face(image)
        detection = face.detection
        # check detector result is valid
        if (detection.rect.height < 1) and (detection.rect.width < 1):
            print("no face")
            return
        # warp image
        warped_result = self._warp_faces(
            face.img, detection, face.landmarks5_opt.value()
        )
        # extract descrptor
        ext = self.extractor.extractFromWarpedImage(warped_result, self.descriptor)
        err, desc = self.descriptor.save()
        return desc

    def get_descriptors(self, image_names):
        """Create the dictionary with descriptors of people from the database."""
        N = len(image_names)
        print("Extracting the descriptors from the database...")
        for i in range(N):
            self.people_descriptors[
                self.known_face_names[i]
            ] = self._extract_from_photo(image_names[i])
        assert len(self.people_descriptors) == len(image_names)
        print("Extracted the descriptors sucessfuly.")
        return self.people_descriptors
