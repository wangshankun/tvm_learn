import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

import itertools
import onnxruntime

from onnx.backend.test.case.node import expect

#https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
#https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/testdata/transform/fusion/bias_softmax_gen.py


shape = (2, 3, 4)
data = np.random.random_sample(shape).astype(np.float32)
permutation = (1, 2, 0)

node = onnx.helper.make_node(
        'Transpose',
        inputs=['data'],
        outputs=['transposed'],
        perm=permutation
    )

transposed = np.transpose(data, permutation)
expect(node, inputs=[data], outputs=[transposed], name='test_transpose')


onnx.save(
    helper.make_model(
    helper.make_graph(
    [node], "TransposeTest",
    [
        helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3, 4]),
    ],
    [
        helper.make_tensor_value_info('transposed', TensorProto.FLOAT, [3, 4, 2]),
    ],
    [])), r'TransposeTest.onnx')
	
session = onnxruntime.InferenceSession("./TransposeTest.onnx", None)
data = np.random.randint(10, size=(2, 3, 4)).astype(np.float32)

input_0_name = session.get_inputs()[0].name
result = session.run([], {input_0_name:data })
print(result[0].shape)


