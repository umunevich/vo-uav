import cv2

class FeatureMatcher:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    def match(self, des1, des2):
        matches = self.bf.match(des1, des2)
        return sorted(matches, key = lambda x: x.distance)