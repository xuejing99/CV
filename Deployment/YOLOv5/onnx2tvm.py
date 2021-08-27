import onnx
import tvm.relay as relay
import tvm


def load_onnx_model(model_path):
    onnx_model = onnx.load(model_path)
    return onnx_model


def export_tvm(onnx_model, target, shape_dict, save_path):
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # mod= relay.quantize.quantize(mod, params)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    lib.export_library(save_path)


if __name__ == "__main__":
    onnx_model = load_onnx_model('weights/yolov5s_gpu.onnx')
    # target = "llvm"
    target = 'cuda'
    input_name = "images"
    shape_dict = {input_name: [1, 3, 384, 640]}
    save_path = "weights/yolov5s_gpu.so"
    export_tvm(onnx_model, target, shape_dict, save_path)
