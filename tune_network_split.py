import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)

def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape


# Define the neural network and compilation target
network = "resnet-18"
batch_size = 4
layout = "NHWC"
#target = tvm.target.Target("cuda")
target = tvm.target.Target("llvm")
dtype = "float32"

mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
#print("================net tvm.lower(mod)=======================")
#print(tvm.lower(mod, params))
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    log_file = "%s-%s-B%d-%s-subtask-%d.json" % (network, layout, batch_size, target.kind.name, idx)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=5,
    )
    #task.tune(tune_option)
    ###resume_search(task, log_file)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    ir_m = tvm.lower(sch, args, simple_mode=True)
    #tvm定义的所有tir在此链接中
    #https://tvm.apache.org/docs/reference/api/python/tir.html
    #tir实现的代码在tvm/python/tvm/tir里面
    #把模型中所有tir打印出来
    print("=====================Tir Code======================")
    #print(ir_m.astext(show_meta_data=False))
    print(ir_m.astext(show_meta_data=True))
    #print("=====================C Code======================")
    #rt_m = tvm.build(ir_m, [a, b, c], target='c', name='mmult')
    #print(rt_m.get_source())

    #test the schedule
    func = tvm.build(sch, args, target)
    assert func

 
'''
with open('model.json', 'w') as f:
    f.write(lib.get_graph_json())
lib.get_lib().save('model.so')

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
'''
