import torch

class PeopleDetector():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, classes=80)

    def detect(self, orig_img):
        df = self.model(orig_img).pandas().xyxy[0]
        df_person = df[df['name'] == 'person']
        if df_person.shape[0] == 1:
            df_person = df_person.iloc[0]
            return [1, (df_person['xmin'], df_person['ymin'], df_person['xmax'], df_person['xmax'])]
        else:
            return [df_person.shape[0], None]