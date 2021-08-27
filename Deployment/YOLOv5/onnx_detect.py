import cv2
import onnxruntime as ort
import numpy as np
import time
import torch

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


def load_model(weights):
    print(ort.get_device())
    # model = onnx.load(weights)
    # label = model.names
    ort_session = ort.InferenceSession(weights)
    return ort_session


def preprocess(img):
    img0 = img.copy()
    img = letterbox(img, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    img = img[np.newaxis, :]
    return img0, img


def output(pred, labels, threshold, iou_threshold):
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
    ort_session = load_model('weights/yolov5s.onnx')

    repeat = 100
    start = time.time()
    for i in range(repeat):
        im0, img = preprocess(cv2.imread("data/images/zidane.jpg"))
        inputs = {'images': img}
        pred = ort_session.run(None, inputs)[0]
        result = output(torch.from_numpy(pred), labels, 0.25, 0.45)
    end = time.time()

    print("repeat {} times, average costs: ".format(repeat), (end - start) * 1000 / repeat, ' ms.')
    print(result)
