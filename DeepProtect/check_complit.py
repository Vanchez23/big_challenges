import numpy as np

class Wear:
    def __init__(self, points):
        self.popo = points

    def padding(self, box, padx, pady):
        box['xmin'] -= padx
        box['xmax'] += padx

        box['ymin'] -= pady
        box['ymax'] += pady
        return box

    def point_check(self, box, name):
        local_popo = self.popo[name]
        mas = [False] * len(local_popo)
        for ind, coord in enumerate(local_popo):
            for _, row in box.iterrows():
                if (coord.x >= row['xmin'] and coord.x <= row['xmax'] and
                    coord.y >= row['ymin'] and coord.y <= row['ymax']):
                    mas[ind] = True
        return sum(mas) >= 1

    def any_two_points(self, box, name):
        local_popo = self.popo[name]
        mas = [False] * len(local_popo)
        for ind, coord in enumerate(local_popo):
