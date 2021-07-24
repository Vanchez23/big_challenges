import numpy as np
import pandas as pd
import cv2
from math import hypot


class Wear:
    def padding(self, box, padx, pady):
        box['x1'] -= padx
        box['x2'] += padx

        box['y1'] -= pady
        box['y2'] += pady
        return box

    def distance(self, p1, p2):
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    def box_intersection(self, box_glove, box_jacket):
        xA = max(box_glove[0], box_jacket[0])
        yA = max(box_glove[1], box_jacket[1])
        xB = min(box_glove[2], box_jacket[2])
        yB = min(box_glove[3], box_jacket[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box_glove_Area = (box_glove[2] - box_glove[0] + 1) * (box_glove[3] - box_glove[1] + 1)
        return interArea / box_glove_Area

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
                    row = self.padding(row, 20, 20)
                    if (coord[0] >= row['x1'] and coord[0] <= row['x2'] and
                        coord[1] >= row['y1'] and coord[1] <= row['y2']):
                        mas[ind] = True
                        popusk.append(ind_box)
                    else:
                        jacket_df = df[df['label'] == 'jacket'].iloc[0]
                        boxA = [row['x1'].item(), row['y1'].item(), row['x2'].item(), row['y2'].item()]
                        boxB = [jacket_df['x1'].item(), jacket_df['y1'].item(), jacket_df['x2'].item(), jacket_df['y2'].item()]
                        sq = self.box_intersection(boxA, boxB) * 100
                        if sq > 40:
                            mas[ind] = True
                    #     print('Square checked')
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