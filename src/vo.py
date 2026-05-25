import numpy as np
import cv2

class VisualOdometry:
    def __init__(self, extractor, matcher, K):
        self.extractor = extractor
        self.matcher = matcher
        self.K = K  # Матриця внутрішніх параметрів камери
        
        self.prev_kp = None
        self.prev_des = None
        
        # Поточна позиція БПЛА в просторі
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

    def process_frame(self, img):
        kp, des = self.extractor.extract(img)
        if self.prev_kp is None:
            self.prev_kp, self.prev_des = kp, des
            return self.cur_t

        matches = self.matcher.match(self.prev_des, des)
        
        # Витягуємо координати точок, що збіглися
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        # Обчислення Essential Matrix (ядро VO)
        # 
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Розклад матриці на R та t
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        # Оновлення глобальної траєкторії
        self.cur_t = self.cur_t + self.cur_R.dot(t)
        self.cur_R = R.dot(self.cur_R)

        self.prev_kp, self.prev_des = kp, des
        return self.cur_t