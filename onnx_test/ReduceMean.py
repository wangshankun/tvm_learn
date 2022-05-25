import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

import onnxruntime

#https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
#https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/testdata/transform/fusion/bias_softmax_gen.py


shape = [3, 2, 2]
axes = [-2]
keepdims = 1

reduce_mean_node = onnx.helper.make_node(
    'ReduceMean',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
print(reduced)

onnx.save(
    helper.make_model(
    helper.make_graph(
    [reduce_mean_node], "ReduceMeanTest",
    [
        helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2]),
    ],
    [
        helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [3, 1, 2]),
    ],
    [])), r'ReduceMeanTest.onnx')
	
session = onnxruntime.InferenceSession("./ReduceMeanTest.onnx", None)
x = np.random.randn(3, 2, 2).astype(np.float32)
input_0_name = session.get_inputs()[0].name
result = session.run([], {input_0_name:x })
#print(result)
