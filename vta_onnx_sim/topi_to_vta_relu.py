from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
from tvm import topi
import numpy as np

np.set_printoptions(threshold=999999)

import vta
from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator
import logging

env = vta.get_env()
print(env.__dict__, flush=True)
if env.TARGET in ["sim", "tsim"]:
    remote = rpc.LocalSession()

m = 4
n = 6
# compute
a = te.placeholder((m, n, env.BATCH, env.BLOCK_OUT), name="a", dtype=env.acc_dtype)
a_buf = te.compute(
	(m, n, env.BATCH, env.BLOCK_OUT), lambda *i: a(*i), "a_buf"
)  # DRAM->SRAM

'''
b_buf = te.compute((m, n, env.BATCH, env.BLOCK_OUT), 
                   lambda *i: tvm.te.max(a_buf(*i), tvm.tir.const(0, a_buf.dtype)),
                   name="b_buf")
'''
b_buf = topi.nn.relu(a_buf)

res = te.compute(
    (m, n, env.BATCH, env.BLOCK_OUT),
    lambda *i: b_buf(*i).astype(env.inp_dtype),
    "res_buf",
)  # SRAM->DRAM

# schedule
s = te.create_schedule(res.op)
s[a_buf].set_scope(env.acc_scope)  # SRAM
s[a_buf].pragma(a_buf.op.axis[0], env.dma_copy)  # DRAM->SRAM
s[b_buf].set_scope(env.acc_scope)  # SRAM
s[b_buf].pragma(b_buf.op.axis[0], env.alu)  # compute
s[res].pragma(res.op.axis[0], env.dma_copy)  # SRAM->DRAM
# build
print(vta.lower(s, [a, res], simple_mode=True))

print("===================vta.build=================================")
with vta.build_config(debug_flag=2):
	mod = vta.build(s, [a, res], tvm.target.Target("ext_dev", host=env.target_host))
temp = utils.tempdir()
mod.save(temp.relpath("load_act.o"))
remote.upload(temp.relpath("load_act.o"))
f = remote.load_module("load_act.o")
# verify
dev = remote.ext_dev(0)
a_np = np.random.randint(-128, 127, size=(m, n, env.BATCH, env.BLOCK_OUT)).astype(a.dtype)
res_np = np.clip(a_np, 0, (1 << (env.INP_WIDTH - 1)) - 1).astype(res.dtype)
#res_np = np.clip(a_np, 0, a_max = None)

a_nd = tvm.nd.array(a_np, dev)
res_nd = tvm.nd.array(np.zeros((m, n, env.BATCH, env.BLOCK_OUT)).astype(res.dtype), dev)

if env.TARGET in ["sim", "tsim"]:
	simulator.clear_stats()

print("===================run funtion=================================")
f(a_nd, res_nd)

np.testing.assert_equal(res_np, res_nd.numpy())

if env.TARGET in ["sim", "tsim"]:
	sim_stats = simulator.stats()
	print("Relu execution statistics:")
	for k, v in sim_stats.items():
		print("\t{:<16}: {:>16}".format(k, v))

#print(res_nd.numpy().shape)
#print(res_nd.numpy())
