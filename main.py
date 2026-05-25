import cv2
from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.matcher import FeatureMatcher
from src.vo import VisualOdometry

loader = DataLoader("data/mav0/cam0/data/*.png")
extractor = FeatureExtractor()
matcher = FeatureMatcher()

vo = VisualOdometry(extractor, matcher)

# перший кадр
first_img = cv2.imread(loader.get_frame(0), 0)
vo.process_first_frame(first_img)

# цикл
for i in range(1, 1000):
    img = cv2.imread(loader.get_frame(i), 0)

    output = vo.process_frame(img)

    cv2.imshow("VO", output)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()