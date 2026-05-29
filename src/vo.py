import numpy as np
import cv2

class VisualOdometry:
    def __init__(self, K):
        self.K = K
        
        # Поточна глобальна позиція камери в просторі
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # ОПТИМІЗАЦІЯ 1: Розширюємо параметри Лукаса-Канаде для кращого трекінгу швидких рухів
        self.lk_params = dict(winSize=(31, 31),       # Збільшено з (21,21) для компенсації розмиття (motion blur)
                              maxLevel=4,             # Збільшено з 3 для кращої обробки великих зсувів пікселів
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Параметри для пошуку сильних кутів (алгоритм Shi-Tomasi)
        self.feature_params = dict(maxCorners=2000,
                                   qualityLevel=0.01,
                                   minDistance=10,
                                   blockSize=3)

        # Дані Ключового кадру (Keyframe) - наш "якір"
        self.kf_frame = None
        self.kf_pts = None     
        self.kf_R = np.eye(3)
        self.kf_t = np.zeros((3, 1))

        # Дані попереднього кадру (для покрокового трекінгу Optical Flow)
        self.prev_frame = None
        self.prev_pts = None   
        
    def set_new_keyframe(self, img):
        """Створює новий якір: знаходить свіжі стабільні точки для відстеження"""
        self.kf_frame = img.copy()
        self.prev_frame = img.copy()
        
        # Виявлення сильних кутових ознак, за які легко зачепитися
        self.kf_pts = cv2.goodFeaturesToTrack(img, mask=None, **self.feature_params)
        self.prev_pts = self.kf_pts
        
        self.kf_R = self.cur_R.copy()
        self.kf_t = self.cur_t.copy()

    def process_frame(self, img):
        # 1. Якщо це перший кадр або ми втратили критичну кількість точок — оновлюємо якір
        if self.kf_frame is None or self.kf_pts is None or len(self.kf_pts) < 40:
            self.set_new_keyframe(img)
            return self.cur_t

        # 2. Обчислюємо Оптичний потік Лукаса-Канаде між попереднім та поточним кадрами
        cur_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, img, self.prev_pts, None, **self.lk_params)

        # 3. Фільтруємо точки, залишаючи лише ті, що успішно пройшли трекінг (status == 1)
        good_new = cur_pts[st == 1]
        good_kf = self.kf_pts[st == 1]

        # Якщо точок залишилося замало — скидаємо якір на поточний кадр
        if len(good_new) < 40:
            self.set_new_keyframe(img)
            return self.cur_t

        # 4. Розраховуємо істотну матрицю E між базовим Ключовим кадром та поточним зображенням
        E, mask = cv2.findEssentialMat(good_kf, good_new, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Захист від нелінійних вихідних форматів матриць у OpenCV
        if E is not None and (E.shape == (3, 3) or E.shape[0] % 3 == 0):
            if E.shape[0] > 3:
                E = E[0:3, 0:3]
                
            _, R_rel, t_rel, mask = cv2.recoverPose(E, good_kf, good_new, self.K)

            # ОПТИМІЗАЦІЯ 2: Підвищуємо фільтр "стояння на місці" для відсіювання піксельного шуму
            distance_moved = np.linalg.norm(t_rel)
            if distance_moved > 0.15: # Збільшено поріг з 0.05
                
                # ОПТИМІЗАЦІЯ 3: Жорстка нормалізація масштабу (Scale Fix)
                # Примусово зводимо довжину вектора трансляції до одиниці
                t_normalized = t_rel / distance_moved
                
                absolute_scale = 1.0 # Цей коефіцієнт тепер лінійно задає крок на графіку
                
                # Оновлюємо глобальні координати відштовхуючись від Ключового кадру
                self.cur_t = self.kf_t + absolute_scale * self.kf_R.dot(t_normalized)
                self.cur_R = R_rel.dot(self.kf_R)

                # Якщо відійшли занадто далеко від поточного "якоря" — фіксуємо новий Ключовий кадр
                if distance_moved > 1.5: 
                    self.set_new_keyframe(img)
                    return self.cur_t

        # 5. Зберігаємо стан кадрів та масивів для наступної ітерації циклу
        self.prev_frame = img.copy()
        
        # Гарантуємо правильну розмірність (N, 1, 2) для методів OpenCV на наступному кроці
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.kf_pts = good_kf.reshape(-1, 1, 2)

        return self.cur_t