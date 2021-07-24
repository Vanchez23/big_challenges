import torch
import cv2
import numpy as np
from typing import Tuple, List
from .utils import letterbox, non_max_suppression, scale_coords
from random import randrange
import pandas as pd

class WearDetector():
    def __init__(self, path):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', autoshape=False, classes=5)
        self.model.load_state_dict(torch.load(path)['model'].state_dict())

    def plot_one_box(self, x, im, color=(128, 128, 128), label=None, line_thickness=3):
        # Plots one bounding box on image 'im' using OpenCV
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
        tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def detect(self, img, isDrawing = True):
        pred, tensor_shape, orig_shape, bbox = self.preprocess(img)
        list_of_boxes = self.postprocess(pred, tensor_shape, orig_shape, bbox)
        if len(list_of_boxes) == 0:
            if isDrawing:
                return [0, list_of_boxes, img]
            else:
                return [0, list_of_boxes]

        df = pd.DataFrame(list_of_boxes)
        mas = [False] * 4
        mas[0] = df[df['label'] == 'shield'].shape[0] == 1
        mas[1] = df[df['label'] == 'jacket'].shape[0] == 1
        mas[2] = df[df['label'] == 'pants'].shape[0] == 1
        mas[3] = df[df['label'] == 'gloves'].shape[0] == 2

        if isDrawing:
            for el in list_of_boxes:
                self.plot_one_box((el['x1'], el['y1'], el['x2'], el['y2']), img, label=el['label'], color=(255, 0, 0), line_thickness=2)

            return [sum(mas) == 4, list_of_boxes, img]
        else:
            return [sum(mas) == 4, list_of_boxes]


        #df = self.model(img).pandas().xyxy[0]
        #return [df['class'].unique().shape[0] == 5, df]

    def preprocess(self, img):
        orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox(orig_img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).to('cuda').float()
        img_tensor /= 255

        img_tensor = img_tensor.unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            results = self.model(img_tensor)

        return results[0], list(img_tensor.shape), list(orig_img.shape), [0, 0, img.shape[1], img.shape[0]]

    def postprocess(self, pred: (Tuple, List, np.ndarray), tensor_shape: (List, np.ndarray), orig_shape: List, bbox: List,
                    normalize_output: bool = False, conf_thres: float = 0.5, iou_thres: float = 0.3,
                    agnostic: bool = True, device='cuda',
                    classes=['gloves', 'pants', 'jacket', 'helmet', 'shield']) -> List:
        if isinstance(pred, (Tuple, List, np.ndarray)):
            pred = pred[0]

        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            agnostic=agnostic)

        new_preds = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    tensor_shape[2:], det[:, :4], orig_shape).round()
                det[:,
                :4] += torch.tensor([bbox[0],
                                     bbox[1],
                                     bbox[0],
                                     bbox[1]]).to(device)
                for *xyxy, conf, cls in reversed(det):
                    for i in range(len(xyxy)):
                        xyxy[i] = xyxy[i].cpu().numpy()
                    if normalize_output:
                        xyxy[0] /= orig_shape[1]
                        xyxy[1] /= orig_shape[0]
                        xyxy[2] /= orig_shape[1]
                        xyxy[3] /= orig_shape[0]
                    new_preds.append({'conf': conf.to('cpu').numpy(),
                                      'label': classes[int(cls)],
                                      'x1': xyxy[0],
                                      'y1': xyxy[1],
                                      'x2': xyxy[2],
                                      'y2': xyxy[3]})
                    #plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=2)

        return new_preds