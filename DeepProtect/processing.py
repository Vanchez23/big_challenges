import cv2
import torch

def preprocess(path_to_img):
    orig_img = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
    img = letterbox(orig_img)[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).to('cuda').float()
    img_tensor /=255
    img_tensor = img_tensor.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        results = model(img_tensor)
    return results[0], list(img_tensor.shape), list(orig_img.shape), [0, 0, img.shape[1], img.shape[0]]

    def postprocess(pred: (Tuple, List, np.ndarray), tensor_shape: (List, np.ndarray), orig_shape: List, bbox: List,
                    normalize_output: bool = False, conf_thres: float = 0.3, iou_thres: float = 0.3,
                    agnostic: bool = True, device='cuda',
                    classes=['gloves', 'pants', 'jacket', 'helmet', 'shield', 'person']) -> List:
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