from __future__ import absolute_import, print_function

import onnx
from onnx import shape_inference, TensorProto
import onnx.helper as helper

from tvm.relay.testing import densenet, mobilenet, resnet, resnet_3d, squeezenet
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision

import sys
import os
import time
import numpy as np
import tvm
import vta
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils
from vta.testing import simulator
from vta.top import graph_pack
from tvm.contrib.download import download_testdata

# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()
print(env.__dict__)
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network = "resnet18-v1-7"
network_onnx = network + ".onnx"
print(network_onnx)

model_url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/" + network_onnx
model_path = download_testdata(model_url, network_onnx, module="onnx")
onnx_model = onnx.load(model_path)

start_name = "nn.max_pool2d"
stop_name = "nn.global_avg_pool2d"

print(env.TARGET)
print(target.device_name)

remote = rpc.LocalSession()

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

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
        #graph, lib, params = relay.build(
        #    relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params
        #)
        #graph, lib, params = relay.build_module.build(
        #relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params)
        tir = relay.build_module.get_lower_ir(
        relay_prog, target=tvm.target.Target(target, host=env.target_host), params=params)

print(tir)
