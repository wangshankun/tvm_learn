from __future__ import absolute_import, print_function

import onnx
from tvm.relay.testing import mobilenet, resnet, squeezenet

import os
import numpy as np
import tvm
import vta
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils

from vta.testing import simulator
from vta.top import graph_pack
from tvm.contrib.download import download_testdata

from functools import partial
print_flushed = partial(print, flush=True)

# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()
#print_flushed(env.__dict__)
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network_name = "resnet18-v1-7"
network_onnx = network_name + ".onnx"
input_data_name = "data" 
start_name = "nn.max_pool2d"
stop_name = "nn.global_avg_pool2d"

model_url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/" + network_onnx
model_path = download_testdata(model_url, network_onnx, module="onnx")
onnx_model = onnx.load(model_path)

remote = rpc.LocalSession()

#print_flushed(env.TARGET)
# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

graph_path = "./" + network_name + ".json"
param_path = "./" + network_name + ".params"
so_path = "./" + network_name + ".so"


if not os.path.exists(graph_path):
    # Load pre-configured AutoTVM schedules
    with autotvm.tophub.context(target):
        # Populate the shape and data type dictionary for ImageNet classifier input
        dtype_dict = {"data": "float32"}
        shape_dict = {"data": (env.BATCH, 3, 224, 224)}

        # Get off the shelf gluon model, and convert to relay
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        # Update shape and type dictionary
        shape_dict.update({k: v.shape for k, v in params.items()})
        dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)
            # Perform graph packing and constant folding for VTA target
            assert env.BLOCK_IN == env.BLOCK_OUT
            # do device annotation if target is intelfocl or sim
            relay_prog = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=start_name,
                stop_name=stop_name,
                device_annot=(env.TARGET == "intelfocl"),
            )
          
        with vta.build_config(
            opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}
        ):
            intrp = relay.build_module.build(
                       relay_prog,
                       target=tvm.target.Target(target, host=env.target_host),
                       params=params,
                       mod_name=network_name + "_mod")

    graph = intrp.get_executor_config()
    with open(graph_path, 'w') as fo:
        fo.write(graph)

    with open(param_path, 'wb') as fo:
        fo.write(relay.save_param_dict(params))

    intrp.export_library(so_path)


#########################################################################################

print_flushed("network_name upload file: ", so_path)
remote.upload(so_path)
loaded_lib = remote.load_module(so_path)
loaded_graph = open(graph_path).read()
loaded_params = bytearray(open(param_path, "rb").read())

executor = graph_executor.create(loaded_graph, loaded_lib, ctx)
executor.load_params(loaded_params)

#print(executor.get_input_info())

inp_idx = executor.get_input_index(input_data_name)
assert inp_idx != -1#assert mode input node name exist

inp_dshapes, inp_dtypes = executor.get_input_info()

inp_shape = inp_dshapes[input_data_name]
inp_type  = inp_dtypes[input_data_name]
#data = np.random.rand(env.BATCH, 3, 224, 224).astype("float32")
data = np.random.rand(*inp_shape).astype(inp_type)
executor.set_input('data', tvm.nd.array(data))

simulator.clear_stats()
executor.run()
#res = executor.get_output(0).asnumpy()
###  May simulator RPC error,
###  according to the string "run", it can't get the correct graph entry function,
###  so, use executor.run() replace executor.module.time_evaluator. 
#run_timer = executor.module.time_evaluator("run", ctx, number=1, repeat=1)
sim_stats = simulator.stats()
print_flushed("execution statistics:")
for k, v in sim_stats.items():
    print_flushed("\t{:<16}: {:>16}".format(k, v))

#prof_res = run_timer()
#print_flushed(prof_res)

