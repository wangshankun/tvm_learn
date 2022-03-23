import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
from tvm.ir.module import IRModule
from tvm import tir
from tvm.tir import Schedule
from tvm.script import tir as T
import numpy as np

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024
# The default tensor type in tvm
dtype = "float32"

target = "llvm"
#target = "llvm -mcpu=skylake-avx512"
#target = "llvm -mcpu=core-avx2"
dev = tvm.device("cpu", 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

np_repeat = 10
np_runing_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_runing_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

def _schedule_matmul(sch: Schedule):
    #print(sch.mod.functions)
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_tiles = [4, 4, 4, 16]
    j_tiles = [4, 4, 4, 16]
    k_tiles = [128, 8]
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    # Organize the loops into “SSRSRS” 6-level tiles
    sch.reorder(
        i_0, j_0, # S
        i_1, j_1, # S
        k_0,      # R
        i_2, j_2, # S
        k_1,      # R
        i_3, j_3, # S
    )
    #print(sch.mod.script())
    #print(sch.mod)
    return sch

ir_module = MyModule
sch = tir.Schedule(ir_module)
sch = _schedule_matmul(sch)
func = tvm.build(sch.mod, target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
#tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("wsk_SSRSRS_v1: %f" % evaluator(a, b, c).mean)
