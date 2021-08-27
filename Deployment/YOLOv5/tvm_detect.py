import numpy as np
import time
import cv2
import torch
import tvm
from tvm.contrib import graph_executor
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
# from tvm.contrib.debugger import debug_executor as graph_executor


def load_model(weights, target):
    lib = tvm.runtime.load_module(weights)
    # dev = tvm.device(str(target), 0)
    # dev = tvm.cuda(0)
    dev = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](dev))
    return module


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
    weights = "weights/yolov5s.so"
    # weights = "weights/yolov5s_optimizer-sim.so"
    # weights = "weights/yolov5s_optimizer-gpu.so"
    print(weights )

    target = "llvm"
    input_name = "images"
    labels = open("data/label.txt", 'r').readlines()
    module = load_model(weights, target)
    # tvm.cuda(0).sync()

    repeat = 100
    start = time.time()
    for i in range(repeat):
        im0, img = preprocess(cv2.imread("data/images/zidane.jpg"))
        # ss = time.time()
        # module.set_input(input_name, tvm.nd.array(img, tvm.cuda(0)))
        module.set_input(input_name, img)
        module.run()
        # print("inference: ", time.time()-ss)
        # tt = time.time()
        tvm_output = module.get_output(0, tvm.nd.empty((1, 15120, 85))).numpy()
        # print(tvm_output.shape)
        # print("get uotput: ", time.time()-tt)
        result = output(torch.from_numpy(tvm_output), labels, 0.25, 0.45)
    end = time.time()

    print("repeat {} times costs: ".format(repeat), (end - start) * 1000 / repeat, ' ms.')
    print(result)

