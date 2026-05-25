import cv2

class FeatureExtractor:
    def __init__(self, nfeatures = 1000):
        self.orb = cv2.ORB_create(nfeatures)

    def extract(self, image):
        kp, des = self.orb.detectAndCompute(image, None)
        return kp, des