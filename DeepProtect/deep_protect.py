from .people_detection import PeopleDetector
from .special_wear_detector import WearDetector
from .pose_estimation import PoseEstimator
from .check_complit import CheckComplite
import cv2

class Detector():
    def __init__(self, image_size = 640, size_model = 2, cam_num = 0, path_to_model = ''):
        self.people_detector = PeopleDetector()
        self.wear_detector = WearDetector(path = path_to_model)
        self.pose_estimation = PoseEstimator()
        self.check_complit = CheckComplite()

    def detect(self, orig_img, isDrawing = True):
        img = orig_img
        config = {'num_people': 0, 'all_wear': False, 'finally': False}
        config['num_people'] = self.people_detector.detect(orig_img)
        if config['num_people'] == 1:
            wear = self.wear_detector.detect(orig_img, isDrawing = isDrawing)
            if isDrawing:
                img = wear[2]
                #print(img)

            if wear[0]:
                config['all_wear'] = True
                pose = self.pose_estimation.detect(orig_img)
                if pose != None:
                    if isDrawing:
                        img = pose[1]
                        pose = pose[0]
                    if self.check_complit.detect(img, wear[1], pose):
                        config['finally'] = True
                        cv2.putText(img, 'Dressed', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
                    else:
                        cv2.putText(img, 'Undressed', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
            else:
                cv2.putText(img, 'Not complete', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
        elif config['num_people'] == 0:
            cv2.putText(img, 'Zero people', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
        else:
            cv2.putText(img, 'More than one people', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
        if isDrawing:
            return [config, img]
        else:
            return config