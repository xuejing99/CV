import numpy as np
import cv2
import time
from openvino.inference_engine import IENetwork, IECore


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
    

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

    
def load_model(weights):
    print(weights)
    ie = IECore()
    net = ie.read_network(model=weights+'.xml',  weights=weights+'.bin')
    net.batch_size = 1
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)
    exec_net = ie.load_network(network=net, num_requests=2, device_name="CPU")
    return net, exec_net, input_blob, out_blob


def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
    
    
def non_max_suppression(pred, threshold, iou_threshold):
    max_nms, time_limit, max_wh, max_det = 30000, 10.0, 4096, 1000
    nc = pred.shape[2] - 5
    xc = pred[..., 4] > 0.25
    t = time.time()
    output = [np.zeros((0, 6))] * pred.shape[0]
    for xi, x in enumerate(pred):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        index = x[:, 5:].argmax(1).reshape((-1,1))
        conf = x[:, 5:].max(1).reshape((-1,1))
        x = np.concatenate((box, conf, index), 1)[conf.reshape(-1) > 0.25]
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence
        
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = np.array(nms(boxes, scores, 0.45))
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output
    

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])  # y1, y2

        
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
    net, exec_net, input_blob, out_blob = load_model("weights/yolov5s-test")
    repeat = 1000
    start = time.time()

    for i in range(repeat):
        im0, img = preprocess(cv2.imread("data/images/zidane.jpg"))
        exec_net.start_async(request_id=0, inputs={input_blob: img})
        if exec_net.requests[0].wait(-1) == 0:     
            res = exec_net.requests[0].output_blobs
            pred = res['output'].buffer
            result = output(pred, labels, 0.25, 0.45)

    end = time.time()
    print("repeat {} times costs: ".format(repeat), (end - start) * 1000 / repeat, ' ms.')
    print(result)
