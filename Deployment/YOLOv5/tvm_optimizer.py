import onnx
import tvm
from tvm import autotvm
import tvm.relay as relay
from tvm.autotvm.tuner import XGBTuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner


def load_model(weights, target, shape_dict):
    onnx_model = onnx.load(weights)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                  relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    return mod, params, tasks


def create_tvm_runner(number=10, repeat=10, timeout=1000, min_repeat_ms=0):
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
    )
    return runner


def tuning_option_setting(runner):
    tuning_option = {
        "tuner": "random",
        "trials": 10,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": "weights/yolov5-autotuning-gpu.json",
        "tuning_graph_records": "weights/yolov5-graph-autotuning-gpu.log",
    }
    return tuning_option


def tuning_model(tasks, tuning_option):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = GridSearchTuner(task)
        # tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )


def tune_graph(graph, shape_dict, tuning_option, target, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, shape_dict, tuning_option["tuning_records"], target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(tuning_option["tuning_graph_records"])


def save_tvm_model(tuning_option, mod, target, params, save_path):
    with autotvm.apply_history_best(tuning_option["tuning_records"]):
    # with autotvm.apply_graph_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
            lib = relay.build(mod, target=target, params=params)
    lib.export_library(save_path)


if __name__ == "__main__":
    # weights, target = "weights/yolov5s-sim.onnx", "llvm --mcpu=skylake-avx512  -libs=cblas"
    weights, target = "weights/yolov5s_gpu.onnx", 'cuda' 
    input_name = "images"
    save_path = "weights/yolov5s_optimizer-gpu.so"
    shape_dict = {input_name: [1, 3, 384, 640]}
    mod, params, tasks = load_model(weights, target, shape_dict)

    runner = create_tvm_runner()
    tuning_option = tuning_option_setting(runner)
    tuning_model(tasks, tuning_option)
    tune_graph(mod["main"], shape_dict, tuning_option, target)

    save_tvm_model(tuning_option, mod, target, params, save_path)
