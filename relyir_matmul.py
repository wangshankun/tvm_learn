import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
import tvm.testing
import numpy as np

dtype="float32"
m = 1024
k = 1024
n = 1024
b = 1

shape_x = tvm.relay.TensorType((b, m, k), dtype=dtype)
shape_y = tvm.relay.TensorType((b, k, n), dtype=dtype)

x = tvm.relay.var("x", shape_x)
y = tvm.relay.var("y", shape_y)
z = tvm.relay.nn.batch_matmul(x, y)

mod = tvm.ir.IRModule.from_expr(z)

#func = relay.Function(relay.analysis.free_vars(z), z)
#print(func)

target = tvm.target.Target("llvm")
#with tvm.transform.PassContext(opt_level=3):
with tvm.relay.build_config(opt_level=3):
    graph, lib, params = tvm.relay.build(mod, target)

from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
runtime_exec = graph_runtime.create(graph, lib, ctx)

x_np = np.random.uniform(1, 10, size=(b, m, k)).astype(np.float32)
y_np = np.random.uniform(1, 10, size=(b, k, n)).astype(np.float32)
 
x_tvm = tvm.nd.array(x_np, device=ctx)
y_tvm = tvm.nd.array(y_np, device=ctx)

print("[Python] Execute the compiled model")
runtime_exec.set_input(0, x_tvm)
runtime_exec.set_input(1, y_tvm)
runtime_exec.set_input(**params)
runtime_exec.run()

output = runtime_exec.get_output(0).asnumpy()
output = output.astype(np.float32)
print(output)
print("[Python] Done")
