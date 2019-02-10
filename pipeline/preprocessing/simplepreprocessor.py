# import packages
import cv2

class SimplePreprocessor:

    def __init__(self, width, height, inter = cv2.INTER_AREA):
        # store the target image width, height and interpolation
        # method for resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size
        # ignore the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)
