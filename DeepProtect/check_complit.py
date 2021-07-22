import numpy as np
import pandas as pd
import cv2
from math import hypot


class Wear:
    def padding(self, box, padx, pady):
        box['xmin'] -= padx
        box['xmax'] += padx

        box['ymin'] -= pady
        box['ymax'] += pady
        return box

    def distance(self, p1, p2):
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def point_check(self, box, points, name = '', num = 1):
        local_popo = points[name]

        #dist = self.distance(local_popo[0], local_popo[1])
        #print('Dist', dist)
        df = pd.DataFrame(box)
        mas = [False] * len(local_popo)
        popusk = []
        for ind, coord in enumerate(local_popo):
            for ind_box, row in enumerate(box):
                if 'glove' in name and row['label'] == 'gloves':
                    if ind_box in popusk:
                        continue
                    if (coord[0] >= row['x1'] and coord[0] <= row['x2'] and
                        coord[1] >= row['y1'] and coord[1] <= row['y2']):
                        mas[ind] = True
                        popusk.append(ind_box)
                    # else:
                    #     jacket_df = df[df['label'] == 'jacket']
                    #     boxA = (row['x1'].tolist(), row['y1'].tolist(), row['x2'].tolist(), row['y2'].tolist())
                    #     print(boxA)
                    #     boxB = (jacket_df['x1'].tolist(), jacket_df['y1'].tolist(), jacket_df['x2'].tolist(), jacket_df['y2'].tolist())
                    #     print(boxB)
                    #     print(self.bb_intersection_over_union(boxA, boxB))
                    #print(self.distance(coord, (row['x1'] + (row['x2'] - row['x1']) / 2, row['y1'] + (row['y2'] - row['y1']) / 2)), dist)
                    # elif self.distance(coord, (row['x1'] + (row['x2'] - row['x1']) / 2, row['y1'] + (row['y2'] - row['y1']) / 2)) <= dist:
                    #     print('Norm distance')
                elif row['label'] == name:
                    if (coord[0] >= row['x1'] and coord[0] <= row['x2'] and
                        coord[1] >= row['y1'] and coord[1] <= row['y2']):
                        mas[ind] = True

        return sum(mas) >= num

class CheckComplite:
    def __init__(self):
        self.wear = Wear()
    def detect(self, img, box, pose):
        shield_check = self.wear.point_check(box, pose, name = 'shield', num=1)
        jacket_check = self.wear.point_check(box, pose, name='jacket', num=1)
        left_glove_check = self.wear.point_check(box, pose, name='left_glove', num=1)
        right_glove_check = self.wear.point_check(box, pose, name='right_glove', num=1)
        pants_check = self.wear.point_check(box, pose, name='pants', num=1)
        #print(shield_check, jacket_check, pants_check, right_glove_check, left_glove_check)

        if shield_check and jacket_check and pants_check and right_glove_check and left_glove_check:
            return True
        else:
            return False