###参考topi的softmax实现，完成reducemean的te开发;
###tvm\python\tvm\topi\nn\softmax.py
###


from __future__ import absolute_import, print_function
import tvm
import numpy as np
from tvm import relay, te, topi

attr_axis = 1
attr_keepdims = 1

b = te.var("b")
n = te.var("n")
m = te.var("m")
X = te.placeholder((b, m, n), name="X")

#assert attr_axis < len(X.shape)

k = te.reduce_axis((0, X.shape[attr_axis]), name="k")

def insert_reduce_index(indices, reduce_index):
    return indices[:attr_axis] + (reduce_index,) + indices[attr_axis:]

def get_non_reduce_indices(indices):
    return tuple([var for (i, var) in enumerate(indices) if i != attr_axis])

def _compute_sum(*indices):
    eval_range = insert_reduce_index(indices, k)
    return te.sum(X[eval_range], axis=k)

def _compute_div(Y, *indices):
    return te.div(Y[indices], X.shape[attr_axis])
    
reduced_shape = tuple([dim for (i, dim) in enumerate(X.shape) if i != attr_axis])
print(reduced_shape)
Y = te.compute(reduced_shape, _compute_sum, name="T_reducemean_sum")
Z = te.compute(reduced_shape, 
               lambda *indices:_compute_div(Y, *indices), 
               name="T_reducemean_div")

if attr_keepdims == 1:
    out_shape = reduced_shape[:attr_axis] + (1,) + reduced_shape[attr_axis:]
    print(out_shape)
    Z = topi.reshape(Z, out_shape)

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

if attr_keepdims == 1:
    b = tvm.nd.array(np.zeros((B, 1, N), dtype=Z.dtype), dev)
else:
    b = tvm.nd.array(np.zeros((B, N), dtype=Z.dtype), dev)

lib_func(a, b)

#tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
print("input data")
print(a.numpy())
print("output data")
print(b.numpy())
