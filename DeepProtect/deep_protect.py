from people_detection import PeopleDetector
from special_wear_detector import WearDetector
from pose_estimation import PoseEstimator
from check_complit import CheckComplite

class Detector():
    def __init__(self, image_size = 640, size_model = 2, cam_num = 0, path_to_model = ''):
        self.people_detector = PeopleDetector()
        self.wear_detector = WearDetector(path = path_to_model)
        self.pose_estimation = PoseEstimator(size_model = size_model)
        self.check_complit = CheckComplite()

    def detect(self, img):
        config = {'num_people': 0, 'all_wear': False, 'finally': False}
        config['num_people'] = self.people_detector.detect(img)
        if config['num_people'] == 1:
            wear = self.wear_detector.detect(img)
            if wear[0]:
                config['all_wear'] = True
                pose = self.pose_estimation.detect(img)
                if self.check_complit.detect(wear[1], pose):
                    config['finnaly'] = True
        return config