import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

import onnxruntime

#https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
#https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/testdata/transform/fusion/bias_softmax_gen.py


add = helper.make_node("Add", ["input", "bias"], ["add_out"], "add")
softmax1 =  helper.make_node("Softmax", ["add_out"], ["output"], "softmax", axis=1)

onnx.save(
    helper.make_model(
    helper.make_graph(
    [add, softmax1], "Add_Softmax_Fusion",
    [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 128]),
        helper.make_tensor_value_info('bias', TensorProto.FLOAT, [4, 128]),
    ],
    [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 128]),
    ],
    [])), r'bias_softmax_fusion_simple.onnx')
	
session = onnxruntime.InferenceSession("./bias_softmax_fusion_simple.onnx", None)
x = np.random.randn(4, 128).astype(np.float32)
y = np.random.randn(4, 128).astype(np.float32)
input_0_name = session.get_inputs()[0].name
input_1_name = session.get_inputs()[1].name
result = session.run([], {input_0_name:x, input_1_name:y})
print(result)
