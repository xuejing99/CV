import torch
import onnx
from onnxsim import simplify

from models.experimental import attempt_load


def load_torch_model(weights):
    model = attempt_load(weights, map_location=torch.device("cpu"))
    # model = attempt_load(weights, map_location=torch.device("cuda:0"))
    model.eval()
    return model


def export_onnx(model, img, save_path):
    torch.onnx.export(model, img, save_path, verbose=False, opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=False,
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes=None)

def simplify(weights, output_path):
    onnx_model = onnx.load(weights)
    model_simp, check = simplify(onnx_mode, output_path)
    onnx.save(model_simp, output_path)

if __name__ == "__main__":
    model = load_torch_model('./weights/yolov5s.pt')
    # img = torch.zeros(1, 3, *[384, 640]).to(torch.device("cuda:0"))
    img = torch.zeros(1, 3, *[384, 640]).to(torch.device("cpu"))
    export_onnx(model, img, './weights/yolov5s.onnx')
    simplify('./weights/yolov5s.onnx', './weights/yolov5s-sim.onnx')
    print("well done!")

