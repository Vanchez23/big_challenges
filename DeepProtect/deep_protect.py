from .people_detection import PeopleDetector
from .special_wear_detector import WearDetector
from .pose_estimation import PoseEstimator
from .check_complit import CheckComplite

class Detector():
    def __init__(self, image_size = 640, size_model = 2, cam_num = 0, path_to_model = '', hrnet_config_path=''):
        self.people_detector = PeopleDetector()
        self.wear_detector = WearDetector(path = path_to_model)
        self.pose_estimation = PoseEstimator(hrnet_config_path)
        self.check_complit = CheckComplite()

    def detect(self, orig_img, isDrawing = True):
        img = orig_img
        config = {'num_people': 0, 'all_wear': False, 'finally': False, 'reason': 0}
        config['num_people'], bbox = self.people_detector.detect(orig_img)
        # if bbox is not None:
        #     cv2.rectangle(orig_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2, cv2.LINE_AA)
        if config['num_people'] == 1:
            wear = self.wear_detector.detect(orig_img, isDrawing = isDrawing)
            if isDrawing:
                img = wear[2]
                #print(img)

            if wear[0]:
                config['all_wear'] = True
                pose = self.pose_estimation.detect(orig_img, bbox = bbox, isDrawing = isDrawing)
                if pose != None:
                    # if isDrawing:
                    #     img = pose[1]
                    #     pose = pose[0]
                    if self.check_complit.detect(img, wear[1], pose):
                        config['finally'] = True
                        #cv2.putText(img, 'Dressed', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
                    else:
                        config['reason'] = 4
                        #cv2.putText(img, 'Undressed', (50, 50), 0, 1, [225, 255, 255], thickness=3, lineType=cv2.LINE_AA)
            else:
                config['reason'] = 3
        elif config['num_people'] == 0:
            config['reason'] = 1
        else:
            config['reason'] = 2


        return [config, img]