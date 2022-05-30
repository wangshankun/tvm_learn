from __future__ import absolute_import, print_function
import tvm
import numpy as np
from tvm import relay, te, topi
import tvm.testing

b = te.var("b")
c = te.var("c")
h = te.var("h")

X = te.placeholder([b, c, h])
'''
Y = te.compute(X.shape, lambda i, j, k: X[i, j, k])
s = te.create_schedule(Y.op)
s[Y].transform_layout(lambda i, j, k: [j, k, i])
'''

Y = topi.transpose(X, axes=[1,2,0])
s = te.create_schedule(Y.op)

#####################################################
tgt = tvm.target.Target(target="llvm", host="llvm")
target = 'llvm'
lib_func = tvm.build(s, [X, Y], tgt, name="test")

dev = tvm.device(tgt.kind.name, 0)

B=2
C=3
H=4
a = tvm.nd.array(np.random.rand(B, C, H).astype(X.dtype), dev)
b = tvm.nd.array(np.zeros((C, H, B), dtype=Y.dtype), dev)

lib_func(a, b)

#tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
print("input data")
print(a.numpy())
print("output data")
print(b.numpy())
#https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
c = np.transpose(a.numpy(), (1, 2, 0))
tvm.testing.assert_allclose(c,  b.numpy(),  rtol=1e-5)
print("verfiy data")
print(c)


