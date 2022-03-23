import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
import tvm.testing
import numpy as np

from tvm.relay.backend import te_compiler, Executor, Runtime
from tvm.relay.backend.te_compiler import TECompiler

dshape = (1, 16, 64, 64)
x = relay.var("x", shape=dshape)
x = relay.add(x, relay.const(1, "float32"))
y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=16)
y1 = relay.add(relay.const(1, "float32"), y)
y = relay.add(y, y1)
z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(1, 1), padding=(0, 0), channels=16)
z = relay.add(z2, z3)
func = relay.Function(relay.analysis.free_vars(z), z)
#print("before pass: ", func)
func = run_opt_pass(func, transform.FuseOps(fuse_opt_level=2))
#print("before pass: ", func)

tec = tvm.relay.backend.te_compiler.get()
xx = tec.items()
print(tec.lower(z))

