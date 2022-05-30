from __future__ import absolute_import, print_function
import tvm
import numpy as np
from tvm import relay, te, topi
import tvm.testing


keepdims = 1
axis = [1, -1]

b = te.var("b")
n = te.var("n")
m = te.var("m")
X = te.placeholder((b, m, n), name="X")
Y = topi.sum(X, axis=axis, keepdims=keepdims)

def _compute_div(Y, *indices):
    filter_index = list(x % len(X.shape) for x in axis)#negative index to  positive index
    assert len(filter_index) == len(set(filter_index))#no duplicate index
        
    reduced_shape = list(X.shape[i] for i in filter_index)
    return te.div(Y[indices], sum(reduced_shape))
    
Z = te.compute(Y.shape, 
               lambda *indices:_compute_div(Y, *indices), 
               name="T_reducemean_div")
s = te.create_schedule(Z.op)
#####################################################
tgt = tvm.target.Target(target="llvm", host="llvm")
target = 'llvm'
lib_func = tvm.build(s, [X, Z], tgt, name="test")

dev = tvm.device(tgt.kind.name, 0)

B=3
N=2
M=2
a = tvm.nd.array(np.random.rand(B, M, N).astype(X.dtype), dev)

if keepdims == 1:
    b = tvm.nd.array(np.zeros((B, 1, 1), dtype=Z.dtype), dev)
else:
    b = tvm.nd.array(np.zeros((B), dtype=Z.dtype), dev)

lib_func(a, b)

#tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
print("input data")
print(a.numpy())
print("output data")
print(b.numpy())
#https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
c = np.mean(a.numpy(), axis=tuple(axis), keepdims=keepdims == 1)
tvm.testing.assert_allclose(c,  b.numpy(),  rtol=1e-5)
print("verfiy data")
print(c)
