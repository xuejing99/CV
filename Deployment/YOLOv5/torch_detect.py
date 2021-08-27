import time
import numpy as np
import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

def preprocess(img):
    img0 = img.copy()
    img = letterbox(img, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    img = torch.from_numpy(img).to(torch.device("cpu"))
    img = img.float() if torch.device("cpu").type == "cpu" else img.half()
    img /= 255.0
    img = img[np.newaxis, :]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img0, img


def load_model(weights):
    # model = attempt_load(weights, map_location=torch.device("cuda:0"))
    model = attempt_load(weights, map_location=torch.device("cpu"))
    model.eval()
    return model


def output(pred, labels, threshold, iou_threshold):
    pred = pred.float()
    pred = non_max_suppression(pred, threshold, iou_threshold)
    boxes = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            for *x, conf, cls_id in det:
                lbl = labels[int(cls_id)]
                x1, y1 = int(x[0]), int(x[1])
                x2, y2 = int(x[2]), int(x[3])
                boxes.append((x1, y1, x2, y2, lbl, conf))
    return boxes


if __name__ == "__main__":
    labels = open("data/label.txt", 'r').readlines()
    model = load_model('./weights/yolov5s.pt')

    repeat = 100
    start = time.time()
    for i in range(repeat):
        im0, img = preprocess(cv2.imread("data/images/zidane.jpg"))
        pred = model(img)[0]
        result = output(pred, labels, 0.25, 0.45)
    end = time.time()

    print("repeat {} times costs: ".format(repeat), (end-start)*1000/repeat, ' ms.')
    print(result)



