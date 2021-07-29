import numpy as np
import yaml
from typing import List, Dict

from DeepProtect.hrnet import HRNetModel
from DeepProtect.hrnet.utils.utils_hrnet import draw_joints, draw_points, COLORS, LINK_PAIRS, POINT_COLOR

class PoseEstimator():
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.model = HRNetModel(self.cfg)

    def keypoints_detect(self, img: np.ndarray, bbox: List,
                         flip: bool = False) -> Dict[str, Dict]:
        coords, confs = self.model(img, bbox)

        keypoints = dict()
        for pred, conf, idx in zip(coords, confs, range(
                len(self.cfg['KEYPOINTS_NAMES']))):
            pred = list(pred)
            label = self.cfg['KEYPOINTS_NAMES'][idx]
            if flip:
                if label.startswith('l_'):
                    label = label.replace('l_', 'r_')
                elif label.startswith('r_'):
                    label = label.replace('r_', 'l_')
            keypoints[idx] = {'point': (int(pred[0]), int(pred[1])),
                              'conf': conf,
                              'label': label}
        return keypoints

    def detect(self, image, bbox, isDrawing = True):
        coords = self.keypoints_detect(image, bbox=bbox)

        if isDrawing:
            draw_joints(image, coords, LINK_PAIRS, COLORS)
            draw_points(image, coords, POINT_COLOR, False, 2)

        ret = {'shield': [coords[0]['point']],

               'jacket': [coords[5]['point'],
                          coords[6]['point']],

               'left_glove': [coords[9]['point']],

               'right_glove': [coords[10]['point']],

               'pants': [coords[13]['point'],
                         coords[14]['point']]}
        return ret

#0,12,11,24,23,20,21,19,18,16,22,15,17,25,26
#нос,плечи, таз,кисти,колени