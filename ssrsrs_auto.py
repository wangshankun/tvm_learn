import tvm
import tvm.testing
import numpy
import timeit
from tvm.ir.module import IRModule
from tvm.tir import Schedule
from tvm.script import tir as T
import numpy as np
from tvm import te, auto_scheduler, topi

def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul(N, K, M, dtype):
    A = te.placeholder((N, K), name="A", dtype=dtype)
    B = te.placeholder((K, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")

    matmul = te.compute(
        (M, N),
        lambda m, n: te.sum(A[m, k] * B[k, n], axis=k),
        name="matmul")

    return [A, B, matmul]

target = tvm.target.Target("llvm")
N = K = M = 1024
dtype = "float32"
task = tvm.auto_scheduler.SearchTask(func=matmul, args=(N, K, M, dtype), target=target)

# Inspect the computational graph
#print("Computational DAG:")
#print(task.compute_dag)

log_file = "matmul.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=5,
)

task.tune(tune_option)
#resume_search(task, log_file)
# Apply the best schedule
sch, args = task.apply_best(log_file)

#test the schedule
func = tvm.build(sch, args, target)
assert func

dev = tvm.device("cpu", 0)
# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("ssrsrs_auto: %f" % evaluator(a, b, c).mean)
'''
print("=====================C Code======================")
ir_m = tvm.lower(sch, args, simple_mode=True)
rt_m = tvm.build(ir_m, [a, b, c], target='c', name='mmult')
print(rt_m.get_source())
print("=====================Tir Code======================")
print(ir_m.astext(show_meta_data=False))
'''

np_repeat = 100
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
#numpy 和 tvm 结果对比
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-3)
