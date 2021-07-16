import torch

class WearDetector():
    def __init__(self, path):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, classes = 5)
        self.model.load_state_dict(torch.load(path)['model'].state_dict())

    def detect(self, orig_img):
        df = self.model(orig_img).pandas().xyxy[0]
        return [df['class'].unique().shape[0] == 5, df]