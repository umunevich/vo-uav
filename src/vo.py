import numpy as np
import cv2

class VisualOdometry:
    def __init__(self, K):
        self.K = K
        
        # Поточна глобальна позиція
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # Параметри для Оптичного потоку Лукаса-Канаде
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Параметри для пошуку сильних кутів (алгоритм Shi-Tomasi замість ORB)
        self.feature_params = dict(maxCorners=2000,
                                   qualityLevel=0.01,
                                   minDistance=10,
                                   blockSize=3)

        # Дані Ключового кадру (Keyframe)
        self.kf_frame = None
        self.kf_pts = None     # Оригінальні координати точок на Ключовому кадрі
        self.kf_R = np.eye(3)
        self.kf_t = np.zeros((3, 1))

        # Дані попереднього кадру (для покрокового трекінгу)
        self.prev_frame = None
        self.prev_pts = None   # Поточні координати цих точок
        
    def set_new_keyframe(self, img):
        """Створює новий якір: знаходить свіжі точки для відстеження"""
        self.kf_frame = img.copy()
        self.prev_frame = img.copy()
        
        # Використовуємо детектор кутів Shi-Tomasi (він кращий для трекінгу, ніж ORB)
        self.kf_pts = cv2.goodFeaturesToTrack(img, mask=None, **self.feature_params)
        self.prev_pts = self.kf_pts
        
        self.kf_R = self.cur_R.copy()
        self.kf_t = self.cur_t.copy()

    def process_frame(self, img):
        # 1. Якщо це перший кадр або ми втратили забагато точок — робимо новий Keyframe
        if self.kf_frame is None or self.kf_pts is None or len(self.kf_pts) < 40:
            self.set_new_keyframe(img)
            return self.cur_t

        # 2. Магія Оптичного потоку: шукаємо, куди перемістилися точки з попереднього кадру
        cur_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, img, self.prev_pts, None, **self.lk_params)

        # 3. Фільтруємо точки: залишаємо тільки ті, які алгоритм успішно знайшов (status == 1)
        good_new = cur_pts[st == 1]
        good_kf = self.kf_pts[st == 1] # Обов'язково фільтруємо і точки Keyframe!

        # Якщо після відсіювання залишилося мало точок — оновлюємо якір
        if len(good_new) < 40:
            self.set_new_keyframe(img)
            return self.cur_t

        # 4. Рахуємо матрицю E між Ключовим кадром (good_kf) та Поточним (good_new)
        E, mask = cv2.findEssentialMat(good_kf, good_new, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Захист від битих матриць
        if E is not None and (E.shape == (3, 3) or E.shape[0] % 3 == 0):
            if E.shape[0] > 3:
                E = E[0:3, 0:3]
                
            _, R_rel, t_rel, mask = cv2.recoverPose(E, good_kf, good_new, self.K)

            # Захист від "зависання"
            distance_moved = np.linalg.norm(t_rel)
            if distance_moved > 0.05: 
                absolute_scale = 1.0 
                
                # Оновлюємо глобальні координати
                self.cur_t = self.kf_t + absolute_scale * self.kf_R.dot(t_rel)
                self.cur_R = R_rel.dot(self.kf_R)

                # Якщо відлетіли далеко від якоря — оновлюємо його
                if distance_moved > 1.5: 
                    self.set_new_keyframe(img)
                    return self.cur_t

        # 5. Оновлюємо стан для наступного кадру
        self.prev_frame = img.copy()
        
        # Важливо: повертаємо масивам правильну форму (N, 1, 2), якої вимагає OpenCV
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.kf_pts = good_kf.reshape(-1, 1, 2)

        return self.cur_t