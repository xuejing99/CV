import numpy as np
import cv2
import time
from openvino.inference_engine import IENetwork, IECore
import torch

import pdb

from utils.general import non_max_suppression, scale_coords

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def preprocess(img):
    img0 = img.copy()
    img = letterbox(img, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    img = img[np.newaxis, :]
    return img0, img
    

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
    
    
def load_model(weights):
    ie = IECore()
    net = ie.read_network(model=weights)
    net.batch_size = 1
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)
    exec_net = ie.load_network(network=net, num_requests=2, device_name="CPU")
    return net, exec_net, input_blob, out_blob


class Detect():
    def __init__(self):
        self.grid = {'Conv_251': np.array(0), 'Conv_559': np.array(0), 'Conv_867': np.array(0)}
        self.stride = {'Conv_251': 8, 'Conv_559': 16, 'Conv_867': 32}
        anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 
                   59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
        anchors = np.array(anchors).reshape(3,1,-1,1,1,2)
        layer = ['Conv_251', 'Conv_559', 'Conv_867']
        self.anchor_grid = {}
        for i in range(len(layer)):
            self.anchor_grid[layer[i]] = anchors[i]  
    
    def forward(self, blob, layer_name):
        bs, _, ny, nx = blob.shape
        blob = blob.reshape(bs, 3, 85, ny, nx).transpose(0,1,3,4,2)
        if self.grid[layer_name].shape[2:4] != blob.shape[2:4]:
            xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
            self.grid[layer_name] = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2))
        y = 1.0/(1.0+np.exp(-blob)) 
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[layer_name]) * self.stride[layer_name]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[layer_name]
        return y.reshape(bs, -1, 85)
    
    def detect(self, output):
        objects = list()
        for layer_name, out_blob in output.items():
            objects.append(self.forward(out_blob.buffer, layer_name))
            # objects.append(self.forward(out_blob, layer_name))
        return np.concatenate(objects, 1)


def output(pred, labels, threshold, iou_threshold):
    # pdb.set_trace()
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
    net, exec_net, input_blob, out_blob = load_model("weights/yolov5s-test.xml")
    detector = Detect()
    repeat = 1000
    start = time.time()

    for i in range(repeat):
        # t0 = time.time()
        im0, img = preprocess(cv2.imread("data/images/zidane.jpg"))
        # print("read image: ", time.time()-t0)
        
        """
        res = exec_net.infer(inputs={input_blob: img})
        pred = detector.detect(res)
        result = output(torch.from_numpy(pred), labels, 0.25, 0.45)
        """
        t1 = time.time()
        exec_net.start_async(request_id=0, inputs={input_blob: img})
        # print("start_async", time.time()-t1)
        if exec_net.requests[0].wait(-1) == 0:     
            res = exec_net.requests[0].output_blobs
            # print(res['output'].buffer)
            # print("get_output: ", time.time()-t1)
            # t2 = time.time()
            # pred = detector.detect(res)
            # print("detect forward: ", time.time()-t2)
            pred = res['output'].buffer
            # t3 = time.time()
            result = output(torch.from_numpy(pred), labels, 0.25, 0.45)
            # print("NMS: ", time.time()-t3)
        
    end = time.time()
    print("repeat {} times costs: ".format(repeat), (end - start) * 1000 / repeat, ' ms.')
    print(result)
