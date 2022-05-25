from __future__ import absolute_import, print_function
import tvm
import numpy as np
from tvm import relay, te, topi


#def _compute_max(*indices):
#    out_shape = indices[:axis] + (1,) + indices[axis:]
    

n = te.var("n")
m = te.var("m")
mysum = te.comm_reducer(lambda x, y: x+y,
    lambda t: tvm.tir.const(0, dtype=t), name="mysum")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n, ), lambda i: mysum(A[i, k], axis=k), name="B")
C = topi.reshape(B, (n, 1))#用reshape改一下维度
s = te.create_schedule(C.op)
#####################################################
tgt = tvm.target.Target(target="llvm", host="llvm")
target = 'llvm'
lib_func = tvm.build(s, [A, C], tgt, name="test")

dev = tvm.device(tgt.kind.name, 0)

N=1
M=2
a = tvm.nd.array(np.random.rand(N, M).astype(A.dtype), dev)
b = tvm.nd.array(np.zeros((N, 1), dtype=B.dtype), dev)
lib_func(a, b)

#tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
print(a.numpy())
print(b.numpy())
