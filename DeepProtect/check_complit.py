import numpy as np
import pandas as pd
import cv2

class Wear:
    def padding(self, box, padx, pady):
        box['xmin'] -= padx
        box['xmax'] += padx

        box['ymin'] -= pady
        box['ymax'] += pady
        return box

    def point_check(self, box, points, name = '', num = 1):
        local_popo = points[name]
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