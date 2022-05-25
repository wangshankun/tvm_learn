from __future__ import absolute_import, print_function

import tvm
import numpy as np
#定义一些变量
n = tvm.var('n')
m = tvm.var('m')
#定义矩阵元素element-wise乘法
A = tvm.placeholder((m,n), name='A')
B = tvm.placeholder((m,n), name='B')
C = tvm.compute((m,n),lambda i,j: A[i,j] * B[i,j], name ='C')
#创建调度
s = tvm.create_schedule([C.op])
#lower会将计算从定义转换为真正的可调用函数。 使用参数`simple_mode = True`，它将返回一个可读的C伪代码，我们在这里使用它来打印计划结果。
print(tvm.lower(s, [A, B, C], simple_mode=True))

