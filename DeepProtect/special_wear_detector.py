import torch
import cv2
import numpy as np
from typing import Tuple, List
from .utils import letterbox, non_max_suppression, scale_coords

class WearDetector():
    def __init__(self, path):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, classes = 5)
        self.model.load_state_dict(torch.load(path)['model'].state_dict())

    def detect(self, img):
        pred, tensor_shape, orig_shape, bbox = self.preprocess(img)
        list_of_boxes = self.postprocess(pred, tensor_shape, orig_shape, bbox)
        num = 0

        for el in list_of_boxes:
            if (el['label'] == 'shield' or
                    el['label'] == 'helmet' or
                    el['label'] == 'jacket' or
                    el['label'] == 'pants' or
                    el['label'] == 'gloves'):
                num += 1
        return [num >= 3, list_of_boxes]


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
                    normalize_output: bool = False, conf_thres: float = 0.3, iou_thres: float = 0.3,
                    agnostic: bool = True, device='cuda',
                    classes=['gloves', 'pants', 'jacket', 'helmet', 'shield']) -> List:
        """
        Обработка предсказаний модели и отбор детекций.

        Args:
            pred: предсказания модели.
            tensor_shape: размер входного тензора.
            orig_shape: оригинальный размер изображения.
            bbox: координаты bounding box.

        Returns:
            Очищенные предсказания.
        """
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

        return new_preds