from jittor_core import *
from jittor_core.ops import *
from .misc import *
from . import attention as attention, contrib as contrib, dataset as dataset, init as init, linalg as linalg, lr_scheduler as lr_scheduler, numpy2cupy as numpy2cupy, optim as optim, sparse as sparse
from .compile_extern import cublas as cublas, cudnn as cudnn, cufft as cufft, curand as curand, mkl_ops as mkl_ops, mpi_ops as mpi_ops, world_size as world_size
from .compiler import compile_custom_op as compile_custom_op, compile_custom_ops as compile_custom_ops
from .contrib import concat as concat
from .nn import bmm as bmm, bmm_transpose as bmm_transpose, matmul as matmul
from collections import OrderedDict as OrderedDict
from collections.abc import Mapping as Mapping
from typing import Any, List, Tuple


def safepickle(obj, path) -> None: ...
def safeunpickle(path): ...

class _call_no_record_scope:
    def __enter__(self) -> None: ...
    def __exit__(self, *exc) -> None: ...
    def __call__(self, func): ...

class flag_scope(_call_no_record_scope):
    jt_flags: Any
    def __init__(self, **jt_flags) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc) -> None: ...

class no_grad(flag_scope):
    jt_flags: Any
    def __init__(self, **jt_flags) -> None: ...

class enable_grad(flag_scope):
    jt_flags: Any
    def __init__(self, **jt_flags) -> None: ...

single_log_capture: Any

class log_capture_scope(_call_no_record_scope):
    fs: Any
    def __init__(self, **jt_flags) -> None: ...
    logs: Any
    def __enter__(self): ...
    def __exit__(self, *exc) -> None: ...

class profile_scope(_call_no_record_scope):
    fs: Any
    warmup: Any
    rerun: Any
    def __init__(self, warmup: int = ..., rerun: int = ..., **jt_flags) -> None: ...
    report: Any
    def __enter__(self): ...
    def __exit__(self, *exc) -> None: ...

class __single_process_scope:
    rank: Any
    def __init__(self, rank: int = ...) -> None: ...
    bk_in_mpi: Any
    bk_mpi_state: Any
    def __enter__(self): ...
    def __exit__(self, *exc) -> None: ...

def single_process_scope(rank: int = ...): ...
def clean() -> None: ...
cast = unary

def array(data, dtype: Any | None = ...): ...
def random(shape, dtype: str = ..., type: str = ...): ...
def float_auto(x): ...
def array64(data, dtype: Any | None = ...): ...
def grad(loss, targets, retain_graph: bool = ...): ...
def liveness_info(): ...
def ones(shape, dtype: str = ...): ...
def ones_like(x): ...
def zeros(shape, dtype: str = ...): ...
def full(shape, val, dtype: str = ...): ...
def full_like(x, val, dtype: Any | None = ...): ...
def zeros_like(x, dtype: Any | None = ...): ...

def var(x, dim: Any | None = ..., dims: Any | None = ..., unbiased: bool = ..., keepdims: bool = ...): ...
def std(x): ...
def norm(x, p: int = ..., dim: int = ..., keepdims: bool = ..., eps: float = ..., keepdim: bool = ...): ...
origin_reshape = reshape

def reshape(x, *shape): ...
view = reshape
origin_transpose = transpose

def transpose(x, *dim): ...
permute = transpose
def flatten(input, start_dim: int = ..., end_dim: int = ...): ...
def detach(x): ...
def unsqueeze(x, dim): ...
def squeeze(x, dim): ...
def clamp(x, min_v: Any | None = ..., max_v: Any | None = ...): ...
def type_as(a, b): ...
def masked_fill(x, mask, value): ...
def sqr(x): ...
def pow(x, y): ...
def argmax(x, dim, keepdims: bool = ...): ...
def argmin(x, dim, keepdims: bool = ...): ...
def randn(*size, dtype: str = ..., requires_grad: bool = ...): ...
def rand(*size, dtype: str = ..., requires_grad: bool = ...): ...
def rand_like(x, dtype: Any | None = ...): ...
def randn_like(x, dtype: Any | None = ...): ...
def randint(low, high: Any | None = ..., shape=..., dtype: str = ...): ...
def randint_like(x, low, high: Any | None = ...): ...
def normal(mean, std, size: Any | None = ..., dtype: str = ...): ...
def attrs(var): ...
def fetch(*args) -> None: ...
def display_memory_info() -> None: ...
def load(path: str): ...
def save(params_dict, path: str): ...

class Module:
    def __init__(self, *args, **kw) -> None: ...
    def execute(self, *args, **kw) -> None: ...
    def __call__(self, *args, **kw): ...
    def __name__(self) -> None: ...
    def dfs(self, parents, k, callback, callback_leave: Any | None = ...) -> None: ...
    def parameters(self) -> List: ...
    def state_dict(self, to: Any | None = ...): ...
    def named_parameters(self) -> List[Tuple[str, Var]]: ...
    def load_state_dict(self, params) -> None: ...
    def modules(self) -> List: ...
    def named_modules(self): ...
    def requires_grad_(self, requires_grad: bool = ...): ...
    def __hooked_call__(self, *args, **kw): ...
    __fhook__: Any
    def register_forward_hook(self, func) -> None: ...
    def remove_forward_hook(self) -> None: ...
    __fhook2__: Any
    def register_pre_forward_hook(self, func) -> None: ...
    def remove_pre_forward_hook(self) -> None: ...
    __bihook__: Any
    def register_input_backward_hook(self, func) -> None: ...
    def remove_input_backward_hook(self) -> None: ...
    __bohook__: Any
    def register_output_backward_hook(self, func) -> None: ...
    def remove_output_backward_hook(self) -> None: ...
    def register_backward_hook(self, func): ...
    def remove_backward_hook(self) -> None: ...
    def children(self) -> List: ...
    def extra_repr(self): ...
    def apply(self, func) -> None: ...
    def load_parameters(self, params) -> None: ...
    def save(self, path: str): ...
    def load(self, path: str): ...
    backup_grad_state: Any
    def eval(self) -> None: ...
    def train(self) -> None: ...
    is_train: bool
    def is_training(self) -> bool: ...
    def mpi_param_broadcast(self, root: int = ...) -> None: ...
    def __setattr__(self, key, value) -> None: ...
    def __getattr__(self, key): ...
    def float64(self): ...
    def float16(self): ...
    def half(self): ...
    def float_auto(self): ...

class Function(Module):
    input_mask: Any
    output_mask: Any
    def __call__(self, *args): ...
    def dfs(self, parents, k, callback, callback_leave: Any | None = ...) -> None: ...
    @classmethod
    def apply(cls, *args, **kw): ...

class GradHooker(Function):
    hook: Any
    def __init__(self, hook) -> None: ...
    def execute(self, *args): ...
    def grad(self, *grad_input): ...

def grad_hooker(args, hook): ...
def register_hook(v, hook): ...
def make_module(func, exec_n_args: int = ...): ...
def dirty_fix_pytorch_runtime_error() -> None: ...

class ExitHooks:
    exit_code: Any
    exception: Any
    def __init__(self) -> None: ...
    def hook(self) -> None: ...
    def exit(self, code: int = ...) -> None: ...
    def exc_handler(self, exc_type, exc, *args) -> None: ...

hooks: Any

def jittor_exit() -> None: ...
def vtos(v): ...
def size(v, dim: Any | None = ...): ...
def to_int(v): ...
def to_float(v): ...
def to_bool(v): ...
def format(v, spec): ...
def get_len(var): ...
half = float16

def is_var(v): ...
from typing import List, Tuple, Callable, overload
import numpy
def ternary(cond: Var, x: Var, y: Var):
 ...
@overload
def reindex(x: Var, shape: Tuple[int], indexes: List[str], overflow_value: float=0, overflow_conditions: List[str]={}, extras: List[Var]={}):
	'''Document:
	* 
	    Reindex Operator is a one-to-many map operator.
	    It performs equivalent Python-pseudo implementation below::
	
	        # input is x, output is y
	        n = len(shape)-1
	        m = len(x.shape)-1
	        k = len(overflow_conditions)-1
	        y = np.zeros(shape, x.dtype)
	        for i0 in range(shape[0]): # 1-st loop
	            for i1 in range(shape[1]): # 2-nd loop
	                ...... # many loops
	                for in in range(shape[n]) # n+1 -th loop
	                    if is_overflow(i0,i1,...,in):
	                        y[i0,i1,...,in] = overflow_value
	                    else:
	                        # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
	                        y[i0,i1,...,in] = x[indexes[0],indexes[1],...,indexes[m]]
	
	        # is_overflow is defined as following
	        def is_overflow(i0,i1,...,in):
	            return (
	                indexes[0] < 0 || indexes[0] >= x.shape[0] ||
	                indexes[1] < 0 || indexes[1] >= x.shape[1] ||
	                ......
	                indexes[m] < 0 || indexes[m] >= x.shape[m] ||
	
	                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
	                overflow_conditions[0] ||
	                overflow_conditions[1] ||
	                ......
	                overflow_conditions[k]
	            )
	    ----------------
	    * [in] x:	A input jittor Var
		
	    * [in] shape:	the output shape, a integer array
		
	    * [in] indexes:	array of c++ style integer expression, its length should be the same with the number of dimension of x, some buildin variables it can use are::
	        
	             XDIM, xshape0, ..., xshapen, xstride0, ..., xstriden
	             YDIM, yshape0, ..., yshapem, ystride0, ..., ystridem
	             i0, i1, ..., in
	             @e0(...), @e1(...) for extras input index
	             e0p, e1p , ... for extras input pointer
				 
	    * [in] overflow_value:	overflow value
		
	    * [in] overflow_conditions:	array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes
			
	    * [in] extras: extra var used for index
		
	    ----------------
	    Example
	    Convolution implemented by reindex operation::
	
	        def conv(x, w):
	            N,H,W,C = x.shape
	            Kh, Kw, _C, Kc = w.shape
	            assert C==_C
	            xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
	                'i0', # Nid
	                'i1+i3', # Hid+Khid
	                'i2+i4', # Wid+KWid
	                'i5', # Cid
	            ])
	            ww = w.broadcast_var(xx)
	            yy = xx*ww
	            y = yy.sum([3,4,5]) # Kh, Kw, C
	            return y, yy'''
	...
@overload
def reindex(x: Var, indexes: List[Var], overflow_value: float=0, overflow_conditions: List[str]={}):
	'''Document:
	* Alias x.reindex([i,j,k]) -> 
	        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])'''
	...
def reindex_var(x: Var, indexes: List[Var], overflow_value: float=0, overflow_conditions: List[str]={}):
	'''Document:
	* Alias x.reindex([i,j,k]) -> 
	        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])'''
	...
@overload
def index(shape: Tuple[int], dim: int, dtype: str="int32"):
	'''

与Jittor库中的jt.index(shape: Tuple[int], dtype: str="int32")方法类似，生成输入张量所有可能的索引。

    参数:
        - a (Var): 需要索引的输入张量。 
        - dtype (str): 数据类型，默认为'int32'。

    返回值:
        output (List[jittor.Var]): 所有符合给定形状的索引。

    代码示例:
        >>> import jittor as jt
        >>> a = jt.array([[1, 2], [3, 4]])
        >>> print(jt.index(a))
        (jt.Var([[0 0]
                 [1 1]], dtype=int32),
        jt.Var([[0 1]
                [0 1]], dtype=int32))

'''
	...
@overload
def index(shape: Tuple[int], dtype: str="int32"):
	'''

与Jittor库中的jt.index(shape: Tuple[int], dtype: str="int32")方法类似，生成输入张量所有可能的索引。

    参数:
        - a (Var): 需要索引的输入张量。 
        - dtype (str): 数据类型，默认为'int32'。

    返回值:
        output (List[jittor.Var]): 所有符合给定形状的索引。

    代码示例:
        >>> import jittor as jt
        >>> a = jt.array([[1, 2], [3, 4]])
        >>> print(jt.index(a))
        (jt.Var([[0 0]
                 [1 1]], dtype=int32),
        jt.Var([[0 1]
                [0 1]], dtype=int32))

'''
	...
@overload
def index(a: Var, dim: int, dtype: str="int32"):
	''' 
``a`` 中每个元素在每一维的下标

    参数：
        - a (Var): 一个张量
        - dtype (str，可选): 返回的数据类型。默认值： ``'int32'``

    返回值：
        ``tuple[Var]`` 包含 ``a.ndims`` 个张量，每个张量形状和 ``a`` 一样，第 :math:`k` 个张量下标 :math:`(i_0, ..., i_k, ...)` 处元素值是 :math:`i_k` ，即第 :math:`k` 维的下标

    代码示例：
        >>> x = jt.zeros((2, 3, 4))
        >>> jt.index_var(x)
        (jt.Var([[[0 0 0 0]
                  [0 0 0 0]
                  [0 0 0 0]]

                 [[1 1 1 1]
                  [1 1 1 1]
                  [1 1 1 1]]], dtype=int32),
         jt.Var([[[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]

                 [[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]], dtype=int32),
         jt.Var([[[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]

                 [[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]], dtype=int32))

'''
	...
@overload
def index(a: Var, dtype: str="int32"):
	''' 
``a`` 中每个元素在每一维的下标

    参数：
        - a (Var): 一个张量
        - dtype (str，可选): 返回的数据类型。默认值： ``'int32'``

    返回值：
        ``tuple[Var]`` 包含 ``a.ndims`` 个张量，每个张量形状和 ``a`` 一样，第 :math:`k` 个张量下标 :math:`(i_0, ..., i_k, ...)` 处元素值是 :math:`i_k` ，即第 :math:`k` 维的下标

    代码示例：
        >>> x = jt.zeros((2, 3, 4))
        >>> jt.index_var(x)
        (jt.Var([[[0 0 0 0]
                  [0 0 0 0]
                  [0 0 0 0]]

                 [[1 1 1 1]
                  [1 1 1 1]
                  [1 1 1 1]]], dtype=int32),
         jt.Var([[[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]

                 [[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]], dtype=int32),
         jt.Var([[[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]

                 [[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]], dtype=int32))

'''
	...
@overload
def index_var(a: Var, dim: int, dtype: str="int32"):
	''' 
``a`` 中每个元素在每一维的下标

    参数：
        - a (Var): 一个张量
        - dtype (str，可选): 返回的数据类型。默认值： ``'int32'``

    返回值：
        ``tuple[Var]`` 包含 ``a.ndims`` 个张量，每个张量形状和 ``a`` 一样，第 :math:`k` 个张量下标 :math:`(i_0, ..., i_k, ...)` 处元素值是 :math:`i_k` ，即第 :math:`k` 维的下标

    代码示例：
        >>> x = jt.zeros((2, 3, 4))
        >>> jt.index_var(x)
        (jt.Var([[[0 0 0 0]
                  [0 0 0 0]
                  [0 0 0 0]]

                 [[1 1 1 1]
                  [1 1 1 1]
                  [1 1 1 1]]], dtype=int32),
         jt.Var([[[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]

                 [[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]], dtype=int32),
         jt.Var([[[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]

                 [[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]], dtype=int32))

'''
	...
@overload
def index_var(a: Var, dtype: str="int32"):
	''' 
``a`` 中每个元素在每一维的下标

    参数：
        - a (Var): 一个张量
        - dtype (str，可选): 返回的数据类型。默认值： ``'int32'``

    返回值：
        ``tuple[Var]`` 包含 ``a.ndims`` 个张量，每个张量形状和 ``a`` 一样，第 :math:`k` 个张量下标 :math:`(i_0, ..., i_k, ...)` 处元素值是 :math:`i_k` ，即第 :math:`k` 维的下标

    代码示例：
        >>> x = jt.zeros((2, 3, 4))
        >>> jt.index_var(x)
        (jt.Var([[[0 0 0 0]
                  [0 0 0 0]
                  [0 0 0 0]]

                 [[1 1 1 1]
                  [1 1 1 1]
                  [1 1 1 1]]], dtype=int32),
         jt.Var([[[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]

                 [[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]], dtype=int32),
         jt.Var([[[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]

                 [[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]], dtype=int32))

'''
	...
def binary(x: Var, y: Var, p: str):
 ...
def pow(x: Var, y: Var):
	''' 
对两个张量逐个元素计算乘方，也可以使用 ``**`` 运算符调用

    参数：
        - x (Var): 乘方运算的底数
        - y (Var): 乘方运算的幂次，形状与 ``x`` 相同

    返回值：
        乘方结果，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 3])
        >>> y = jt.array([2, 3, 4])
        >>> jt.pow(x, y)
        jt.Var([ 1  8 81], dtype=int32)
        >>> x ** y
        jt.Var([ 1  8 81], dtype=int32)

'''
	...
def maximum(x: Var, y: Var):
	''' 
对两个张量中对应元素计算最大值

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素中的较大者，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 2, 0])
        >>> jt.maximum(x, y)
        jt.Var([3 2 4], dtype=int32)


'''
	...
def minimum(x: Var, y: Var):
	''' 
对两个张量中对应元素计算最小值

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素中的较小者，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 2, 0])
        >>> jt.minimum(x, y)
        jt.Var([1 2 0], dtype=int32)

'''
	...
def add(x: Var, y: Var):
	''' 
对两个张量中对应元素计算求和，也可以使用 ``+`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相加结果，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.add(x, y)
        jt.Var([4 6 9], dtype=int32)
        >>> x + y
        jt.Var([4 6 9], dtype=int32)


'''
	...
def subtract(x: Var, y: Var):
	''' 
对两个张量中对应元素计算求差，也可以使用 ``-`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相减结果，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.subtract(x, y)
        jt.Var([-2 -2 -1], dtype=int32)
        >>> x - y
        jt.Var([-2 -2 -1], dtype=int32)


'''
	...
def multiply(x: Var, y: Var):
	''' 
对两个张量中对应元素计算求积，也可以使用 ``*`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相乘结果，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.multiply(x, y)
        jt.Var([ 3  8 20], dtype=int32)
        >>> x * y
        jt.Var([ 3  8 20], dtype=int32)


'''
	...
def divide(x: Var, y: Var):
	''' 
对两个张量中对应元素计算求商，也可以使用 ``/`` 运算符调用。

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相除结果，形状与 ``x`` 和 ``y`` 相同

    注意：
        输入数据类型是整数的情况下，会返回浮点数结果

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.divide(x, y)
        jt.Var([0.3333333  0.49999997 0.8       ], dtype=float32)
        >>> x / y
        jt.Var([0.3333333  0.49999997 0.8       ], dtype=float32)

'''
	...
def floor_divide(x: Var, y: Var):
	''' 
对两个张量中对应元素计算求商并向下取整，也可以使用 ``//`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相除向下取整结果，形状与 ``x`` 和 ``y`` 相同，输入数据类型是整数的情况下，会返回整数

    代码示例：
        >>> x = jt.array([3, 4, 5])
        >>> y = jt.array([1, 2, 4])
        >>> jt.floor_divide(x, y)
        jt.Var([3 2 1], dtype=int32)
        >>> x // y
        jt.Var([3 2 1], dtype=int32)

'''
	...
def mod(x: Var, y: Var):
	''' 
对两个张量中对应元素计算求余数，也可以使用 ``%`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相除的余数，形状与 ``x`` 和 ``y`` 相同。非零余数的符号和 ``y`` 中元素的符号相同，该行为与 Python 的 ``x % y`` 运算一致

    代码示例：
        >>> x = jt.array([3, 4, 5, -5, 5])
        >>> y = jt.array([1, 2, 3, 3, -3])
        >>> jt.mod(x, y)
        jt.Var([ 0  0  2 -2  2], dtype=int32)
        >>> x % y
        jt.Var([ 0  0  2 -2  2], dtype=int32)

'''
	...
def less(x: Var, y: Var):
	''' 
两个张量中对应元素比较是否为小于，把结果布尔值放入输出张量的对应位置，也可以使用 ``<`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        输出张量，形状与 ``x`` 和 ``y`` 相同，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素小于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.less(x, y)
        jt.Var([False False  True], dtype=bool)
        >>> x < y
        jt.Var([False False  True], dtype=bool)


'''
	...
def less_equal(x: Var, y: Var):
	''' 
两个张量中对应元素比较是否为小于或等于，把结果布尔值放入输出张量的对应位置，也可以使用 ``<=`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素小于或等于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.less_equal(x, y)
        jt.Var([False  True  True], dtype=bool)
        >>> x <= y
        jt.Var([False  True  True], dtype=bool)

'''
	...
def greater(x: Var, y: Var):
	''' 两个张量中对应元素比较是否为大于，把结果布尔值放入输出张量的对应位置，也可以使用 ``>`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素大于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.greater(x, y)
        jt.Var([ True False False], dtype=bool)
        >>> x > y
        jt.Var([ True False False], dtype=bool)

'''
	...
def greater_equal(x: Var, y: Var):
	''' 
两个张量中对应元素比较是否为大于或等于，把结果布尔值放入输出张量的对应位置，也可以使用 ``>=`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素大于或等于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.greater_equal(x, y)
        jt.Var([ True  True False], dtype=bool)
        >>> x >= y
        jt.Var([ True  True False], dtype=bool)

'''
	...
def equal(x: Var, y: Var):
	''' 
两个张量中对应元素比较是否相等，把结果布尔值放入输出张量的对应位置，也可以使用 ``=`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素等于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.equal(x, y)
        jt.Var([False  True False], dtype=bool)
        >>> x == y
        jt.Var([False  True False], dtype=bool)

'''
	...
def not_equal(x: Var, y: Var):
	''' 
两个张量中对应元素比较是否不相等，把结果布尔值放入输出张量的对应位置，也可以使用 ``!=`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素不等于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.not_equal(x, y)
        jt.Var([ True False  True], dtype=bool)
        >>> x != y
        jt.Var([ True False  True], dtype=bool)


'''
	...
def left_shift(x: Var, y: Var):
	''' 
对两个张量中对应元素计算左移移位运算，也可以使用 ``<<`` 运算符调用

    参数：
        - x (Var): 被左移的张量，元素的数据类型是 ``int32`` 或 ``int64``
        - y (Var): 左移的位数张量，元素数据类型是 ``int32`` 或 ``int64`` ，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，其元素为 ``x`` 和 ``y`` 对应索引位置的元素计算左移操作的结果数

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.left_shift(x, y)
        jt.Var([  8  32 128], dtype=int32)
        >>> x << y
        jt.Var([  8  32 128], dtype=int32)

'''
	...
def right_shift(x: Var, y: Var):
	''' 
对两个张量中对应元素计算右移移位运算，也可以使用 ``>>`` 运算符调用

    参数：
        - x (Var): 被右移的张量，元素的数据类型是 ``int32`` 或 ``int64``
        - y (Var): 右移的位数张量，元素的数据类型是 ``int32`` 或 ``int64`` ，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，其元素为 ``x`` 和 ``y`` 对应索引位置的元素计算右移操作的结果数

    代码示例：
        >>> x = jt.array([123, 345, -456])
        >>> y = jt.array([3, 4, 5])
        >>> jt.right_shift(x, y)
        jt.Var([ 15  21 -15], dtype=int32)
        >>> x >> y
        jt.Var([ 15  21 -15], dtype=int32)

'''
	...
def logical_and(x: Var, y: Var):
	''' 
对两个张量中对应元素计算逻辑与

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
       形状与 ``x`` 和 ``y`` 相同的张量，元素数据类型为 ``bool`` 。其元素为 ``x`` 和 ``y`` 对应索引位置上的元素进行逻辑与操作的结果，即：如果都为非零值或 ``True`` 则输出为 ``True`` ，否则为 ``False`` 。

    代码示例：
        >>> x = jt.array([0, 0, 1, 1])
        >>> y = jt.array([0, 1, 0, 1])
        >>> jt.logical_and(x, y)
        jt.Var([False False False  True], dtype=bool)

'''
	...
def logical_or(x: Var, y: Var):
	''' 
对两个张量中对应元素计算逻辑或

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，元素数据类型为 ``bool`` 。其元素为 ``x`` 和 ``y`` 对应索引位置上的元素进行逻辑或操作的结果，即：如果两者有非零值或 ``True`` 则输出为 ``True`` ，否则为 ``False`` 

    代码示例：
        >>> x = jt.array([0, 0, 1, 1])
        >>> y = jt.array([0, 1, 0, 1])
        >>> jt.logical_or(x, y)
        jt.Var([False  True  True  True], dtype=bool)


'''
	...
def logical_xor(x: Var, y: Var):
	''' 
两个张量的元素级逻辑异或

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
       形状与 ``x`` 和 ``y`` 相同的张量，元素数据类型为 ``bool`` 。其元素为 ``x`` 和 ``y`` 对应索引位置上的元素进行逻辑异或操作的结果，即：如果两者有且仅有一个为非零值或 ``True`` 则输出为 ``True`` ，否则为 ``False`` ，数据类型为 ``bool`` ，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([0, 0, 1, 1])
        >>> y = jt.array([0, 1, 0, 1])
        >>> jt.logical_xor(x, y)
        jt.Var([False  True  True False], dtype=bool)


'''
	...
def bitwise_and(x: Var, y: Var):
	''' 
对两个张量中对应元素计算按位与，也可以使用 ``&`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
       形状与 ``x`` 和 ``y`` 相同的张量。 其元素为 ``x`` 和 ``y`` 对应索引位置上元素的按位与结果。

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.bitwise_and(x, y)
        jt.Var([1 0 4], dtype=int32)
        >>> x & y
        jt.Var([1 0 4], dtype=int32)

'''
	...
def bitwise_or(x: Var, y: Var):
	''' 
两个张量的元素级按位或，也可以使用 ``|`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量。 ``x`` 和 ``y`` 对应索引位置上元素的按位或结果。

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.bitwise_or(x, y)
        jt.Var([3 6 5], dtype=int32)
        >>> x | y
        jt.Var([3 6 5], dtype=int32)


'''
	...
def bitwise_xor(x: Var, y: Var):
	''' 
两个张量的元素级按位异或，也可以使用 ``^`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量。 ``x`` 和 ``y`` 对应索引位置上元素的按位异或结果。

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.bitwise_xor(x, y)
        jt.Var([2 6 1], dtype=int32)
        >>> x ^ y
        jt.Var([2 6 1], dtype=int32)


'''
	...
def tape(x: Var):
 ...
@overload
def where(cond: Var, dtype: str="int32"):
	''' 
对张量中元素计算条件选择

    参数：
        - cond (Var): 选择条件
        - x (Var): 条件我非零值或 ``True`` 时选择的数据，形状与 ``cond`` 一样
        - y (Var): 条件为零或 ``False`` 时选择的数据，形状与 ``cond`` 一样

    返回值：
        形状与输入都一样，每个值如果对应 ``cond`` 中的值为非零或 ``True`` 则返回 ``x`` 中对应的元素，否则返回 ``y`` 中对应的元素

    代码示例：
        >>> cond = jt.array([1, 0, 1])
        >>> x = jt.array([3, 4, 5])
        >>> y = jt.array([9, 8, 7])
        >>> jt.where(cond, x, y)
        jt.Var([3 8 5], dtype=int32)


'''
	...
@overload
def where(cond: Var, x: Var, y: Var):
	''' 
对张量中元素计算条件选择

    参数：
        - cond (Var): 选择条件
        - x (Var): 条件我非零值或 ``True`` 时选择的数据，形状与 ``cond`` 一样
        - y (Var): 条件为零或 ``False`` 时选择的数据，形状与 ``cond`` 一样

    返回值：
        形状与输入都一样，每个值如果对应 ``cond`` 中的值为非零或 ``True`` 则返回 ``x`` 中对应的元素，否则返回 ``y`` 中对应的元素

    代码示例：
        >>> cond = jt.array([1, 0, 1])
        >>> x = jt.array([3, 4, 5])
        >>> y = jt.array([9, 8, 7])
        >>> jt.where(cond, x, y)
        jt.Var([3 8 5], dtype=int32)


'''
	...
def argsort(x: Var, dim: int=-1, descending: bool=False, dtype: str="int32"):
	''' 
返回张量在 ``dim`` 维排序的所用的下标排列和排序后的值

    参数：
        - x (Var): 被排序的数据
        - dim (int): 在哪个维度排序，负数表示从后往前数。默认值：-1
        - descending (bool): 是否降序排列，False表示升序排列。默认值：False
        - dtype (str): 返回的下标的数据类型

    返回值：
        ``tuple[Var, Var]`` ，其中第一个返回张量中每个第 ``dim`` 维的切片是一个 ``[0, x.shape[dim])`` 的排列，表示将 ``x`` 中对应的切片中元素排序的下标顺序，第二个返回张量是排序结果

    代码示例：
        >>> x = jt.array([3, 2, 10, 8])
        >>> jt.argsort(x)
        (jt.Var([1 0 3 2], dtype=int32), jt.Var([ 2  3  8 10], dtype=int32))


'''
	...
def fetch(inputs: List[Var], func: Callable):
 ...
def arg_reduce(x: Var, op: str, dim: int, keepdims: bool):
	'''Document:
	*
	    Returns the indices of the maximum / minimum of the input across a dimension.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] op:      "max" or "min". 
	
	    * [in] dim:     int. Specifies which dimension to be reduced.
	
	    * [in] keepdims: bool. Whether the output has ``dim`` retained or not.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(0, 10, shape=(2, 3))
	        >>> x
	        jt.Var([[4 2 5]
	         [6 7 1]], dtype=int32)
	        >>> jt.arg_reduce(x, 'max', dim=1, keepdims=False)
	        [jt.Var([2 1], dtype=int32), jt.Var([5 7], dtype=int32)]
	        >>> jt.arg_reduce(x, 'min', dim=1, keepdims=False)
	        [jt.Var([1 2], dtype=int32), jt.Var([2 1], dtype=int32)]'''
	...
def random(shape: Tuple[int], dtype: str="float32", type: str="uniform"):
 ...
@overload
def reduce(x: Var, op: str, dim: int, keepdims: bool=False):
 ...
@overload
def reduce(x: Var, op: str, dims: Tuple[int]=(), keepdims: bool=False):
 ...
@overload
def max(x: Var, dim: int, keepdims: bool=False):
	'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
	...
@overload
def max(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
	...
@overload
def max(x: Var, dims_mask: int, keepdims_mask: int):
	'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
	...
@overload
def reduce_maximum(x: Var, dim: int, keepdims: bool=False):
	'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
	...
@overload
def reduce_maximum(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
	...
@overload
def reduce_maximum(x: Var, dims_mask: int, keepdims_mask: int):
	'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
	...
@overload
def min(x: Var, dim: int, keepdims: bool=False):
	'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
	...
@overload
def min(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
	...
@overload
def min(x: Var, dims_mask: int, keepdims_mask: int):
	'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
	...
@overload
def reduce_minimum(x: Var, dim: int, keepdims: bool=False):
	'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
	...
@overload
def reduce_minimum(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
	...
@overload
def reduce_minimum(x: Var, dims_mask: int, keepdims_mask: int):
	'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
	...
@overload
def sum(x: Var, dim: int, keepdims: bool=False):
	''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
	...
@overload
def sum(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
	...
@overload
def sum(x: Var, dims_mask: int, keepdims_mask: int):
	''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
	...
@overload
def reduce_add(x: Var, dim: int, keepdims: bool=False):
	''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
	...
@overload
def reduce_add(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
	...
@overload
def reduce_add(x: Var, dims_mask: int, keepdims_mask: int):
	''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
	...
@overload
def prod(x: Var, dim: int, keepdims: bool=False):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def prod(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def prod(x: Var, dims_mask: int, keepdims_mask: int):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def product(x: Var, dim: int, keepdims: bool=False):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def product(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def product(x: Var, dims_mask: int, keepdims_mask: int):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def reduce_multiply(x: Var, dim: int, keepdims: bool=False):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def reduce_multiply(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def reduce_multiply(x: Var, dims_mask: int, keepdims_mask: int):
	'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
	...
@overload
def reduce_logical_and(x: Var, dim: int, keepdims: bool=False):
	'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
	...
@overload
def reduce_logical_and(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
	...
@overload
def reduce_logical_and(x: Var, dims_mask: int, keepdims_mask: int):
	'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
	...
@overload
def all_(x: Var, dim: int, keepdims: bool=False):
	'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
	...
@overload
def all_(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
	...
@overload
def all_(x: Var, dims_mask: int, keepdims_mask: int):
	'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
	...
@overload
def reduce_logical_or(x: Var, dim: int, keepdims: bool=False):
	'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
	...
@overload
def reduce_logical_or(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
	...
@overload
def reduce_logical_or(x: Var, dims_mask: int, keepdims_mask: int):
	'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
	...
@overload
def any_(x: Var, dim: int, keepdims: bool=False):
	'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
	...
@overload
def any_(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
	...
@overload
def any_(x: Var, dims_mask: int, keepdims_mask: int):
	'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
	...
@overload
def reduce_logical_xor(x: Var, dim: int, keepdims: bool=False):
 ...
@overload
def reduce_logical_xor(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
 ...
@overload
def reduce_logical_xor(x: Var, dims_mask: int, keepdims_mask: int):
 ...
@overload
def reduce_bitwise_and(x: Var, dim: int, keepdims: bool=False):
 ...
@overload
def reduce_bitwise_and(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
 ...
@overload
def reduce_bitwise_and(x: Var, dims_mask: int, keepdims_mask: int):
 ...
@overload
def reduce_bitwise_or(x: Var, dim: int, keepdims: bool=False):
 ...
@overload
def reduce_bitwise_or(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
 ...
@overload
def reduce_bitwise_or(x: Var, dims_mask: int, keepdims_mask: int):
 ...
@overload
def reduce_bitwise_xor(x: Var, dim: int, keepdims: bool=False):
 ...
@overload
def reduce_bitwise_xor(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
 ...
@overload
def reduce_bitwise_xor(x: Var, dims_mask: int, keepdims_mask: int):
 ...
@overload
def mean(x: Var, dim: int, keepdims: bool=False):
	'''
    计算输入张量的均值。

    参数:
        - x (Var): 输入的 :math:`Jittor` 的变量。
        - dims_mask (int): 如果给定，沿着指定的维度缩减。
        - keepdims_mask (bool): 如果为True，则输出结果会保留维度。默认值：False

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[9 4 4]
         [1 9 6]], dtype=int32)
        >>> jt.mean(x)
        jt.Var([5.5000005], dtype=float32)
        >>> x.mean(dim=1)
        jt.Var([5.666667  5.3333335], dtype=float32)
        >>> x.mean(dim=1, keepdims=True)
        jt.Var([[5.666667 ]
         [5.3333335]], dtype=float32)
 
    返回值:
        计算得到的均值(Var)

    

  '''
	...
@overload
def mean(x: Var, dims: Tuple[int]=(), keepdims: bool=False):
	'''
    计算输入张量的均值。

    参数:
        - x (Var): 输入的 :math:`Jittor` 的变量。
        - dims_mask (int): 如果给定，沿着指定的维度缩减。
        - keepdims_mask (bool): 如果为True，则输出结果会保留维度。默认值：False

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[9 4 4]
         [1 9 6]], dtype=int32)
        >>> jt.mean(x)
        jt.Var([5.5000005], dtype=float32)
        >>> x.mean(dim=1)
        jt.Var([5.666667  5.3333335], dtype=float32)
        >>> x.mean(dim=1, keepdims=True)
        jt.Var([[5.666667 ]
         [5.3333335]], dtype=float32)
 
    返回值:
        计算得到的均值(Var)

    

  '''
	...
@overload
def mean(x: Var, dims_mask: int, keepdims_mask: int):
	'''
    计算输入张量的均值。

    参数:
        - x (Var): 输入的 :math:`Jittor` 的变量。
        - dims_mask (int): 如果给定，沿着指定的维度缩减。
        - keepdims_mask (bool): 如果为True，则输出结果会保留维度。默认值：False

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[9 4 4]
         [1 9 6]], dtype=int32)
        >>> jt.mean(x)
        jt.Var([5.5000005], dtype=float32)
        >>> x.mean(dim=1)
        jt.Var([5.666667  5.3333335], dtype=float32)
        >>> x.mean(dim=1, keepdims=True)
        jt.Var([[5.666667 ]
         [5.3333335]], dtype=float32)
 
    返回值:
        计算得到的均值(Var)

    

  '''
	...
def clone(x: Var):
 ...
def unary(x: Var, op: str):
 ...
def cast(x: Var, op: str):
 ...
def int8(x: Var):
	'''  创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``int8`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,3))
        >>> x
        jt.Var([[-0.3373416   0.28905255  1.496737  ]
                [-1.0225579  -1.0431367   0.3423414 ]], dtype=float32)
        >>> x.int8()
        jt.Var([[ 0  0  1]
                [-1 -1  0]], dtype=int8)
        
    返回值：
        返回一个新的张量, 数据类型为 ``int8`` 

    '''
	...
def int16(x: Var):
	'''  创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``int16`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,3))
        >>> x
        jt.Var([[ 1.8623732  -1.060122    0.82836497]
                [-0.17111613  1.0357065  -0.4598468 ]], dtype=float32)
        >>> x.int16()
        jt.Var([[ 1 -1  0]
                [ 0  1  0]], dtype=int16)
        
    返回值：
        返回一个新的张量, 数据类型为 ``int16`` 

    '''
	...
def int32(x: Var):
	'''     创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``int32`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,3))
        >>> x
        jt.Var([[-1.8707825  -0.5307898  -2.2583888 ]
                [-0.1800911   0.10568491  0.0891161 ]], dtype=float32)
        >>> x.int32()
        jt.Var([[-1  0 -2]
                [ 0  0  0]], dtype=int32)

    返回值：
        返回一个新的张量, 数据类型为 ``int32`` 

    '''
	...
def int64(x: Var):
	'''  创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``int64`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,3))
        >>> x
        jt.Var([[-1.4805598  -0.34178254 -0.3623456 ]
                [-0.31025547  0.3924162  -1.5397007 ]], dtype=float32)
        >>> x.int64()
        jt.Var([[-1  0  0]
                [ 0  0 -1]], dtype=int64)

    返回值：
        返回一个新的张量, 数据类型为 ``int64`` 

    '''
	...
def uint8(x: Var):
	''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``uint8`` ，数据范围为 ``[0, 255]``

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[-0.7970826   0.99872327]
                [-0.543484   -2.133479  ]], dtype=float32)
        >>> x.uint8()
        jt.Var([[  0   0]
                [  0 254]], dtype=uint8)

    返回值：
        返回一个新的张量, 数据类型为 ``uint8`` 

    '''
	...
def uint16(x: Var):
	''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``uint16`` ，数据范围为 ``[0, 65535]``

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[-0.7970826   0.99872327]
                [-0.543484   -2.133479  ]], dtype=float32)
        >>> x.uint16()
        jt.Var([[    0     0]
                [    0 65534]], dtype=uint16)

    返回值：
        返回一个新的张量, 数据类型为 ``uint16`` 

    '''
	...
def uint32(x: Var):
	''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``uint32`` ，数据范围为 ``[0, 4294967295]``

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[-0.7970826   0.99872327]
                [-0.543484   -2.133479  ]], dtype=float32)
        >>> x.uint32()
        jt.Var([[         0          0]
                [         0 4294967294]], dtype=uint32)

    返回值：
        返回一个新的张量, 数据类型为 ``uint32`` 

    '''
	...
def uint64(x: Var):
	'''     创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``uint64`` ，数据范围为 ``[0, 18446744073709551615]``

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[-0.7970826   0.99872327]
                [-0.543484   -2.133479  ]], dtype=float32)
        >>> x.uint64()
        jt.Var([[                   0                    0]
                [                   0 18446744073709551614]], dtype=uint64)

    返回值：
        返回一个新的张量, 数据类型为 ``uint64`` 

    '''
	...
def float16(x: Var):
	''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``float16`` ，返回半精度的浮点数，即用16个比特(2个字节)表示一个数。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[ 1.4581492   0.2615918 ]
                [ 0.26567948 -0.34585235]], dtype=float32)
        >>> x.float16()
        jt.Var([[ 1.458   0.2615]
                [ 0.2656 -0.346 ]], dtype=float16)
    返回值：
        返回一个新的张量, 数据类型为 ``float16`` 

    '''
	...
def float32(x: Var):
	'''     创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``float32`` ，返回单精度的浮点数，即用32个比特(4个字节)表示一个数。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.ones((2,2), dtype='int8')
        >>> x
        jt.Var([[1 1]
                [1 1]], dtype=int8)
        >>> x.float32()
        jt.Var([[1. 1.]
                1. 1.]], dtype=float32)
    返回值：
        返回一个新的张量, 数据类型为 ``float32`` 

    '''
	...
def float64(x: Var):
	'''     创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``float64`` ，返回双精度的浮点数，即用64个比特(8个字节)表示一个数。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[ 1.4581492   0.2615918 ]
                [ 0.26567948 -0.34585235]], dtype=float32)
        >>> x.float64()
        jt.Var([[ 1.45814919  0.26159179]
                [ 0.26567948 -0.34585235]], dtype=float64)
    返回值：
        返回一个新的张量, 数据类型为 ``float64`` 

    '''
	...
def abs(x: Var):
	''' 创建一个张量，其将 ``x`` 中的每一个数值转换为其绝对值。

    .. math::
        out_i = |input_i|


    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> jt.abs(jt.float32([-1, 0, 1]))
        jt.Var([1. 0. 1.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置的绝对值
        
 
'''
	...
def negative(x: Var):
	'''     创建一个张量，其将 ``x`` 中的每一个数值转换为其相反数。

    .. math::
        out_i = - input_i


    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> jt. logical_not(jt.float32([-1, 0, 1]))
        jt.Var([1. 0. -1.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置的相反数

    '''
	...
def logical_not(x: Var):
	''' 创建一个张量，其将 ``x`` 中的每一个数值转换为其逻辑非, 数据类型为 ``bool``。 如果输入张量 ``x`` 不是数据类型为 ``bool`` 的张量，则零被视为 ``False`` ，非零被视为 ``True``。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> jt.logical_not(jt.float32([-1, 0, 1]))
        jt.Var([False  True False], dtype=bool)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置的逻辑非、

    '''
	...
def bitwise_not(x: Var):
	''' 创建一个张量，其将 ``x`` 中的每一个数值进行位非运算。 

    参数：
        - x (Var): Var类型的张量，数据类型应为布尔型或整型
        
    代码示例：
        >>> jt.bitwise_not(jt.int32([-2, 3, 1]))
        jt.Var([ 1 -4 -2], dtype=int32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行位非运算后的值

    '''
	...
def log(x: Var):
	'''
    创建一个张量，其将 ``x`` 中的每一个数值的进行自然对数运算。 


    .. math::
        y_i = \ln(x_i)


    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.rand(5) * 5
        >>> x
        jt.Var([2.8649285  0.854602   3.2959728  0.41697088 3.0296502 ], dtype=float32)
        >>> jt.log(x)
        jt.Var([ 1.0525434  -0.15711944  1.1927013  -0.8747389   1.1084472 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行自然对数运算后的值

    注意事项：
        - 自然对数是以常数e（数学常量，大约等于2.71828）为底的对数

    '''
	...
def exp(x: Var):
	'''    创建一个张量，其将 ``x`` 中的每一个数值的进行指数运算。 

    .. math::
        y_i = e^{x_i}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.rand(5) * 5
        >>> x
        jt.Var([2.8649285  0.854602   3.2959728  0.41697088 3.0296502 ], dtype=float32)
        >>> jt.exp(x)
        jt.Var([17.547798   2.3504386 27.003672   1.5173583 20.689995 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行指数运算后的值

    '''
	...
def sqrt(x: Var):
	'''  创建一个张量，其将 ``x`` 中的每一个数值进行平方根运算。 

    .. math::
        out_i = \sqrt{input_i}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 1.1675875  -0.36471805  2.0246403   1.1496693  -0.09650089], dtype=float32)
        >>> jt.sqrt(x)
        jt.Var([1.0805496       nan 1.4228985 1.0722263       nan], dtype=float32)


    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行平方根运算后的值

    注意事项：
        如果输入元素是负数，返回 ``nan`` 


    
'''
	...
def round(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为离其最近的整数（四舍五入）。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 1.1675875  -0.36471805  2.0246403   1.1496693  -0.09650089], dtype=float32)
        >>> jt.round(x)
        jt.Var([ 1. -0.  2.  1. -0.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置取最近整数的值

    '''
	...
def floor(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为向下取整离其最近的整数。
   如：``floor(3.7)`` 的结果是3, ``floor(-3.7)`` 的结果是-4。



    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 1.1675875  -0.36471805  2.0246403   1.1496693  -0.09650089], dtype=float32)
        >>> jt.round(x)
        jt.Var([ 1. -1.  2.  1. -1.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置向下取整离其最近的整数

    '''
	...
def ceil(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为向上取整离其最近的整数。
   如：``ceil(3.7)`` 的结果是4, ``ceil(-3.7)`` 的结果是-3。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.rand(5) * 5
        >>> x
        jt.Var([ 1.1675875  -0.36471805  2.0246403   1.1496693  -0.09650089], dtype=float32)
        >>> jt.ceil(x)
        jt.Var([ 2. -0.  3.  2. -0.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置向上取整离其最近的整数

    '''
	...
def round_int(x: Var):
	''' 创建一个张量, 其将 ``x`` 中的每一个数值转换为离其最近的整数（四舍五入）, 数据类型为 ``int32`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 2.00422    -2.081075   -0.7260027   0.46558696 -0.1570169 ], dtype=float32)
        >>> jt.round_int(x)
        jt.Var([ 2 -2 -1  0  0], dtype=int32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置取最近整数的值, 数据类型为 ``int32`` 

    '''
	...
def floor_int(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为向下取整离其最近的整数。
   如：``floor_int(3.7)`` 的结果是3, ``floor_int(-3.7)`` 的结果是-4。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 2.00422    -2.081075   -0.7260027   0.46558696 -0.1570169 ], dtype=float32)
        >>> jt.floor_int(x)
        jt.Var([ 2 -3 -1  0 -1], dtype=int32)

    返回值：
        返回一个新的张量，其数值是 ``x`` 中对应位置向下取整离其最近的整数，数据类型为 ``int32`` 

    '''
	...
def ceil_int(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为向上取整离其最近的整数。
   如：``ceil_int(3.7)`` 的结果是4, ``ceil_int(-3.7)`` 的结果是-3。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 2.00422    -2.081075   -0.7260027   0.46558696 -0.1570169 ], dtype=float32)
        >>> jt.round_int(x)
        jt.Var([ 3 -2  0  1  0], dtype=int32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应向上取整离其最近的整数, 数据类型为 ``int32`` 

    
'''
	...
def sin(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行正弦运算。

    .. math::
        y_i = \sin(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-1.6485653  -0.10077649 -0.7546536  -0.02543143  1.0830703 ], dtype=float32)
        >>> jt.sin(x)
        jt.Var([-0.9969775  -0.10060599 -0.68503636 -0.02542868  0.88340074], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行正弦运算的结果

    注意事项：
        - 正弦函数的定义域是实数集，值域是 ``[-1, 1]`` 
        - 支持使用 ``x.sin()`` 进行调用

    
'''
	...
def asin(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反正弦运算。

    .. math::
        y_i = \sin^{-1}(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-1.6485653  -0.10077649 -0.7546536  -0.02543143  1.0830703 ], dtype=float32)
        >>> jt.asin(x)
        jt.Var([        nan -0.10094785 -0.85512596 -0.02543417         nan], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反正弦运算的结果

    注意事项：
        - 反正弦函数的定义域是 ``[-1, 1]`` ，值域是实数集， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.asin()`` 进行调用

    '''
	...
def arcsin(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反正弦运算。

    .. math::
        y_i = \sin^{-1}(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-1.6485653  -0.10077649 -0.7546536  -0.02543143  1.0830703 ], dtype=float32)
        >>> jt.asin(x)
        jt.Var([        nan -0.10094785 -0.85512596 -0.02543417         nan], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反正弦运算的结果

    注意事项：
        - 反正弦函数的定义域是 ``[-1, 1]`` ，值域是实数集， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.asin()`` 进行调用

    '''
	...
def sinh(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行双曲正弦运算。

    .. math::
        y_i = \sinh(x_i) = \frac{e^x - e^{-x}}{2}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 0.50530916 -1.0814334   0.0039318   1.1737247  -1.4199417 ], dtype=float32)
        >>> jt.sin(x)
        jt.Var([ 0.5270894  -1.3048973   0.00393181  1.4624014  -1.9475756 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行双曲正弦运算的结果

    注意事项：
        - 正弦函数的定义域和值域是实数集
        - 支持使用 ``x.sinh()`` 进行调用

    
'''
	...
def asinh(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲正弦运算。

    .. math::
        y_i = \sinh^{-1}(x_i) = \ln\left(x_i + \sqrt{x_i^2 + 1}\right)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 0.50530916 -1.0814334   0.0039318   1.1737247  -1.4199417 ], dtype=float32)
        >>> jt.asin(x)
        jt.Var([ 0.48595542 -0.9377998   0.00393179  0.99904275 -1.1495185 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲正弦运算的结果

    注意事项：
        - 反正弦函数的定义域和值域是实数集
        - 支持使用 ``x.asinh()`` 进行调用

    '''
	...
def arcsinh(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲正弦运算。

    .. math::
        y_i = \sinh^{-1}(x_i) = \ln\left(x_i + \sqrt{x_i^2 + 1}\right)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 0.50530916 -1.0814334   0.0039318   1.1737247  -1.4199417 ], dtype=float32)
        >>> jt.asin(x)
        jt.Var([ 0.48595542 -0.9377998   0.00393179  0.99904275 -1.1495185 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲正弦运算的结果

    注意事项：
        - 反正弦函数的定义域和值域是实数集
        - 支持使用 ``x.asinh()`` 进行调用

    '''
	...
def tan(x: Var):
	''' 创建一个张量, 其将 ``x`` 中的每一个数值进行正切运算，即正弦值除以余弦值。

    .. math::
        y_i = \tan(x_i) = \frac{\sin(x)}{\cos(x)}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-0.1146331  -0.22897322 -1.1813502   0.7146521  -0.02658333], dtype=float32)
        >>> jt.tan(x)
        jt.Var([-0.11513788 -0.23306055 -2.4366024   0.8676503  -0.02658959], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行正切运算的结果

    注意事项：
        - 该函数的值域为实数集，定义域为

        .. math:: 
            x \in \mathbb{R} \setminus \left\{ \left(2k+1\right)\frac{\pi}{2} \mid k \in \mathbb{Z} \right\}


        - 支持使用 ``x.tan()`` 进行调用

    '''
	...
def atan(x: Var):
	''' 创建一个张量, 其将 ``x`` 中的每一个数值进行反正切运算。

.. math::
    y_i = \tan^{-1}(x_i) 

参数：
    - x (Var): Var类型的张量
    
代码示例：
    >>> x = jt.randn(5)
    >>> x
    jt.Var([-0.1146331  -0.22897322 -1.1813502   0.7146521  -0.02658333], dtype=float32)
    >>> jt.atan(x)
    jt.Var([-0.1141349  -0.22509298 -0.86834407  0.6204921  -0.02657707], dtype=float32)

返回值：
    返回一个新的张量, 其数值是 ``x`` 中对应位置进行反正切运算的结果

注意事项：
    - 反正弦函数的定义域为实数集，值域为

    .. math:: 
        \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)

    - 支持使用 ``x.tan()`` 进行调用


'''
	...
def arctan(x: Var):
	''' 创建一个张量, 其将 ``x`` 中的每一个数值进行反正切运算。

.. math::
    y_i = \tan^{-1}(x_i) 

参数：
    - x (Var): Var类型的张量
    
代码示例：
    >>> x = jt.randn(5)
    >>> x
    jt.Var([-0.1146331  -0.22897322 -1.1813502   0.7146521  -0.02658333], dtype=float32)
    >>> jt.atan(x)
    jt.Var([-0.1141349  -0.22509298 -0.86834407  0.6204921  -0.02657707], dtype=float32)

返回值：
    返回一个新的张量, 其数值是 ``x`` 中对应位置进行反正切运算的结果

注意事项：
    - 反正弦函数的定义域为实数集，值域为

    .. math:: 
        \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)

    - 支持使用 ``x.tan()`` 进行调用


'''
	...
def tanh(x: Var):
	'''  创建一个张量, 其将 ``x`` 中的每一个数值进行双曲正切运算。

    .. math::
	    \tanh(x) = \frac{{e^{x} - e^{-x}}}{{e^{x} + e^{-x}}}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 0.561267    1.7155168  -0.53718305  1.6382825   0.5129911 ], dtype=float32)
        >>> jt.tanh(x)
        jt.Var([ 0.508917    0.93732095 -0.4908527   0.9272321   0.47227246], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行双曲正切运算的结果

    注意事项：
        - 双曲正切运算的定义域为所有实数，值域为 (-1, 1)
        - 支持使用 ``x.tanh()`` 进行调用

    '''
	...
def atanh(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲正切运算。

    .. math::
        atanh(x) = \frac{1}{2} * log(\frac{1 + x}{1 - x})

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 0.561267    1.7155168  -0.53718305  1.6382825   0.5129911 ], dtype=float32)
        >>> jt.atanh(x)
        jt.Var([ 0.634681         nan -0.6001876        nan  0.5667807], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲正切运算的结果

    注意事项：
        - 该函数的定义域是(-1, 1)，值域是实数集，如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.atanh()`` 进行调用

    '''
	...
def arctanh(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲正切运算。

    .. math::
        atanh(x) = \frac{1}{2} * log(\frac{1 + x}{1 - x})

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 0.561267    1.7155168  -0.53718305  1.6382825   0.5129911 ], dtype=float32)
        >>> jt.atanh(x)
        jt.Var([ 0.634681         nan -0.6001876        nan  0.5667807], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲正切运算的结果

    注意事项：
        - 该函数的定义域是(-1, 1)，值域是实数集，如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.atanh()`` 进行调用

    '''
	...
def cos(x: Var):
	''' 创建一个张量, 其将 ``x`` 中的每一个数值进行余弦运算。

    .. math::
        y_i = \cos(x_i) 

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-0.8017247   2.4553642  -0.57173574 -0.06912863  1.5478854 ], dtype=float32)
        >>> jt.cos(x)
        jt.Var([ 0.6954685  -0.7736413   0.84096307  0.9976116   0.0229089 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行余弦运算的结果

    注意事项：
        - 余弦函数的定义域是实数集，值域是 ``[-1, 1]`` 
        - 支持使用 ``x.cos()`` 进行调用

    
'''
	...
def acos(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反余弦运算。

    .. math::
        y_i = \cos^{-1}(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-0.8017247   2.4553642  -0.57173574 -0.06912863  1.5478854 ], dtype=float32)
        >>> jt.acos(x)
        jt.Var([2.5009716       nan 2.1794162 1.6399801       nan], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反余弦运算的结果

    注意事项：
        - 该函数的定义域是 ``[-1, 1]`` ，值域是[0,pi]， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.acos()`` 进行调用

    '''
	...
def arccos(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反余弦运算。

    .. math::
        y_i = \cos^{-1}(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-0.8017247   2.4553642  -0.57173574 -0.06912863  1.5478854 ], dtype=float32)
        >>> jt.acos(x)
        jt.Var([2.5009716       nan 2.1794162 1.6399801       nan], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反余弦运算的结果

    注意事项：
        - 该函数的定义域是 ``[-1, 1]`` ，值域是[0,pi]， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.acos()`` 进行调用

    '''
	...
def cosh(x: Var):
	'''  创建一个张量, 其将 ``x`` 中的每一个数值进行双曲余弦运算。

    .. math::
        y_i = \cosh(x_i) = \frac{e^x_i + e^{-x_i}}{2}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) + 1
        >>> x
        jt.Var([-0.46972263 -0.6531948   1.7161269   1.007987    1.3451817 ], dtype=float32)
        >>> jt.cosh(x)
        jt.Var([1.1123631 1.2210255 2.871351  1.5525162 2.0496883], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行双曲余弦运算的结果

    注意事项：
        - 双曲余弦运算的定义域和值域为 [1, ∞) 
        - 支持使用 ``x.cosh()`` 进行调用

    '''
	...
def acosh(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲余弦运算。

    .. math::
        y_i = \cosh^{-1}(x_i) 

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) + 1
        >>> x
        jt.Var([-0.46972263 -0.6531948   1.7161269   1.007987    1.3451817 ], dtype=float32)
        >>> jt.acosh(x)
        jt.Var([      nan       nan 1.1348774 0.1263045 0.8086661], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲余弦运算的结果

    注意事项：
        - 该函数的定义域是[1, ∞)，值域是实数集， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.acosh()`` 进行调用

    '''
	...
def arccosh(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲余弦运算。

    .. math::
        y_i = \cosh^{-1}(x_i) 

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) + 1
        >>> x
        jt.Var([-0.46972263 -0.6531948   1.7161269   1.007987    1.3451817 ], dtype=float32)
        >>> jt.acosh(x)
        jt.Var([      nan       nan 1.1348774 0.1263045 0.8086661], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲余弦运算的结果

    注意事项：
        - 该函数的定义域是[1, ∞)，值域是实数集， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.acosh()`` 进行调用

    '''
	...
def sigmoid(x: Var):
	'''    创建一个张量, 其将 ``x`` 中的每一个数值进行 ``sigmoid`` 变换。

    .. math::
         out_i =  \frac{1}{1 + e^{-x_i}}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 2.7442768  -0.6135011  -0.09157802 -2.3600576   0.98465353], dtype=float32)
        >>> jt.sigmoid(x)
        jt.Var([0.93958926 0.35126096 0.47712144 0.08626965 0.72803056], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行 ``sigmoid`` 变换后的值

    注意事项：
        -  ``sigmoid`` 变换的定义域是实数集，值域是(0, 1)
        - 支持使用 ``x.sigmoid()`` 进行调用

    '''
	...
def erf(x: Var):
	'''    创建一个张量,  ``x`` 中的每一个数值进行误差变换， 该函数为误差函数。

    .. math::
        \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} \, dt

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 2.7442768  -0.6135011  -0.09157802 -2.3600576   0.98465353], dtype=float32)
        >>> jt.erf(x)
        jt.Var([-0.91196376  0.19675992  0.9940873  -0.99938697  0.10797469], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行误差函数运算

    注意事项：
        - 该函数定义域是实数集，值域是 [-1,1]
        - 支持使用 ``x.erf()`` 进行调用

    '''
	...
def erfinv(x: Var):
	'''    创建一个张量,  ``x`` 中的每一个数值进行反误差变换， 该函数为反误差函数。

    .. math::
        \text{erfinv}(x) = \text{erf}^{-1}(x)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 0.73859763  1.0348558   2.4863744  -0.36372563 -0.6050172 ], dtype=float32)
        >>> jt.erfinv(x)
        jt.Var([ 0.79413944         nan         nan -0.33440086 -0.6014762 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反误差函数运算

    注意事项：
        - 该函数定义域是[-1,1]，值域是实数集
        - 支持使用 ``x.erfinv()`` 进行调用

    '''
	...
def transpose(x: Var, axes: Tuple[int]=()):
 ...
def fuse_transpose(x: Var, axes: Tuple[int]=()):
 ...
def safe_clip(x: Var, left: float, right: float):
	'''def safe_clip(x: Var, left: float, right: float) ->Var:
    对一个值在一个区间内进行安全剪裁,并保持梯度通过。当输入值在剪裁区间之外时,其梯度保持不变。

    公式如下：
        如果 :math:`left`<  ``x``  < :math:`right` ,那么返回值 :math:`y`  = ``x`` 。否则，如果 :math:`x`  <= :math:`left` ,那么 :math:`y`  =  :math:`left` ；如果 ``x``  >=  :math:`right` ,那么 :math:`y` =  :math:`right` 。

    参数：
        - x(Var):输入值。
        - left(float):剪裁的最小值。默认值：float64
        - right(float):剪裁的最大值。默认值：float64

    代码示例：
        >>> import jittor as jt
        >>> x = jt.Var([-1,0.75,2])
        >>> jt.safe_clip(x, 0.5, 1)
        [0.5,0.75,1.]
    返回值：
        剪裁后的值(Var)。




'''
	...
def array_(args: numpy.ndarray):
 ...
def array(obj: float | int | numpy.ndarray | Var):
 ...
@overload
def getitem(x: Var, slices: slice):
 ...
@overload
def getitem(x: Var, slices: slice, _: int):
 ...
def candidate(x: Var, fail_cond: str, dtype: str="int32"):
	'''Document:
	*
	    Candidate Operator Perform an indirect candidate filter by given a fail condition.
	    
	    x is input, y is output index, satisfy::
	
	        not fail_cond(y[0], y[1]) and
	        not fail_cond(y[0], y[2]) and not fail_cond(y[1], y[2]) and
	        ...
	        ... and not fail_cond(y[m-2], y[m-1])
	
	    Where m is number of selected candidates.
	
	    Pseudo code::
	    
	        y = []
	        for i in range(n):
	            pass = True
	            for j in y:
	                if (@fail_cond):
	                    pass = false
	                    break
	            if (pass):
	                y.append(i)
	        return y
	
	    * [in] x:   input var for filter
	
	    * [in] fail_cond:   code for fail condition
	
	    * [in] dtype:   type of return indexes
	
	    * [out] index: .
	
	    Example::
	
	        jt.candidate(jt.random(100,2), '(@x(j,0)>@x(i,0))or(@x(j,1)>@x(i,1))')
	        # return y satisfy:
	        #    x[y[0], 0] <= x[y[1], 0] and x[y[1], 0] <= x[y[2], 0] and ... and x[y[m-2], 0] <= x[y[m-1], 0] and
	        #    x[y[0], 1] <= x[y[1], 1] and x[y[1], 1] <= x[y[2], 1] and ... and x[y[m-2], 1] <= x[y[m-1], 1]'''
	...
@overload
def numpy_code(shape: Tuple[int], dtype: str, inputs: List[Var], forward: Callable, backward: List[Callable]):
	'''
    这是一个用于自定义操作的 :math:`Numpy Code` 操作符。
    
    参数:
        - shape (Tuple[int])：输出的形状，一个整数数组。
        - dtype (str)：输出的数据类型。
        - inputs (List[Var])：输入的 :math:`jittor Var` 的列表。
        - forward (Callable)：前向 :math:`python` 函数。
        - backward (List[Callable])：每个输入的梯度。
	    
    代码示例：
        >>> def forward_code(np, data):
        >>>     a = data["inputs"][0]
        >>>     b = data["outputs"][0]
        >>>     np.add(a,a,out=b)
        >>> 
        >>> def backward_code(np, data):
        >>>     dout = data["dout"]
        >>>     out = data["outputs"][0]
        >>>     np.copyto(out, dout*2.0)
        >>> 
        >>> a = jt.random((5,1))
        >>> b = jt.numpy_code(
        >>>     a.shape,
        >>>     a.dtype,
        >>>     [a],
        >>>     forward_code,
        >>>     [backward_code],
        >>> )

    返回值:
        执行自定义操作后的 :math:`Jittor variable` 。
        
    
'''
	...
@overload
def numpy_code(shapes: List[Tuple[int]], dtypes: List[str], inputs: List[Var], forward: Callable, backward: List[Callable]):
	'''
    这是一个用于自定义操作的 :math:`Numpy Code` 操作符。
    
    参数:
        - shape (Tuple[int])：输出的形状，一个整数数组。
        - dtype (str)：输出的数据类型。
        - inputs (List[Var])：输入的 :math:`jittor Var` 的列表。
        - forward (Callable)：前向 :math:`python` 函数。
        - backward (List[Callable])：每个输入的梯度。
	    
    代码示例：
        >>> def forward_code(np, data):
        >>>     a = data["inputs"][0]
        >>>     b = data["outputs"][0]
        >>>     np.add(a,a,out=b)
        >>> 
        >>> def backward_code(np, data):
        >>>     dout = data["dout"]
        >>>     out = data["outputs"][0]
        >>>     np.copyto(out, dout*2.0)
        >>> 
        >>> a = jt.random((5,1))
        >>> b = jt.numpy_code(
        >>>     a.shape,
        >>>     a.dtype,
        >>>     [a],
        >>>     forward_code,
        >>>     [backward_code],
        >>> )

    返回值:
        执行自定义操作后的 :math:`Jittor variable` 。
        
    
'''
	...
@overload
def numpy_code(shape: Tuple[int], dtype: str, inputs: List[Var], forward: Callable):
	'''
    这是一个用于自定义操作的 :math:`Numpy Code` 操作符。
    
    参数:
        - shape (Tuple[int])：输出的形状，一个整数数组。
        - dtype (str)：输出的数据类型。
        - inputs (List[Var])：输入的 :math:`jittor Var` 的列表。
        - forward (Callable)：前向 :math:`python` 函数。
        - backward (List[Callable])：每个输入的梯度。
	    
    代码示例：
        >>> def forward_code(np, data):
        >>>     a = data["inputs"][0]
        >>>     b = data["outputs"][0]
        >>>     np.add(a,a,out=b)
        >>> 
        >>> def backward_code(np, data):
        >>>     dout = data["dout"]
        >>>     out = data["outputs"][0]
        >>>     np.copyto(out, dout*2.0)
        >>> 
        >>> a = jt.random((5,1))
        >>> b = jt.numpy_code(
        >>>     a.shape,
        >>>     a.dtype,
        >>>     [a],
        >>>     forward_code,
        >>>     [backward_code],
        >>> )

    返回值:
        执行自定义操作后的 :math:`Jittor variable` 。
        
    
'''
	...
@overload
def numpy_code(shapes: List[Tuple[int]], dtypes: List[str], inputs: List[Var], forward: Callable):
	'''
    这是一个用于自定义操作的 :math:`Numpy Code` 操作符。
    
    参数:
        - shape (Tuple[int])：输出的形状，一个整数数组。
        - dtype (str)：输出的数据类型。
        - inputs (List[Var])：输入的 :math:`jittor Var` 的列表。
        - forward (Callable)：前向 :math:`python` 函数。
        - backward (List[Callable])：每个输入的梯度。
	    
    代码示例：
        >>> def forward_code(np, data):
        >>>     a = data["inputs"][0]
        >>>     b = data["outputs"][0]
        >>>     np.add(a,a,out=b)
        >>> 
        >>> def backward_code(np, data):
        >>>     dout = data["dout"]
        >>>     out = data["outputs"][0]
        >>>     np.copyto(out, dout*2.0)
        >>> 
        >>> a = jt.random((5,1))
        >>> b = jt.numpy_code(
        >>>     a.shape,
        >>>     a.dtype,
        >>>     [a],
        >>>     forward_code,
        >>>     [backward_code],
        >>> )

    返回值:
        执行自定义操作后的 :math:`Jittor variable` 。
        
    
'''
	...
@overload
def code(shape: Tuple[int], dtype: str, inputs: List[Var]={}, cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str=""):
	'''
    用于方便自定义运算操作的代码 :math:`Operator` 。

    参数:
        - inputs: 一个输入 :math:`jittor Var` 的列表
        - outputs: 一个输出 :math:`jittor Var` 的列表
        - cpu_src: 字符串类型，:math:`cpu` 源代码, 预设数值:
           - :math:`in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)`
           - :math:`out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)`
           - :math:`out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)`
        - cpu_grad_src: 字符串类型列表，:math:`cpu` 梯度源代码, 默认值：空字典
        - cpu_header: 字符串类型，:math:`cpu`  :math:`header` 源代码字符串, 默认值：空字符串
        - cuda_src: 字符串类型，:math:`cuda` 源代码, 默认值：空字符串.
        - cuda_grad_src: 字符串类型列表，:math:`cuda` 梯度源代码, 默认值：空字典
        - cuda_header: 字符串类型，:math:`cuda` _  :math:`header` 源代码字符串, 默认值：空字符串

    代码示例：
        >>> from jittor import Function
        >>> import jittor as jt
        >>> class Func(Function):
        >>> def execute(self, x):
            >>> self.save_vars = x
	    >>> return jt.code(x.shape, x.dtype, [x],cpu_src=""\" for (int i=0; i @alias(a, in0) @alias(b, out) ""\", cpu_src=""\" for (int i=0; i using namespace std; ""\", cpu_src=""\" @alias(a, in0) @alias(b, out0) @alias(c, out1) @b(0) = @c(0) = @a(0); for (int i=0; i0) @b(num_b++) = @a(i); else @c(num_c++) = @a(i); } b->set_shape({num_b}); c->set_shape({num_c}); ""\" ) 
        >>> assert (b.data == [5,3,1]).all()
        >>> assert (c.data == [-4,-2]).all()

    返回值:
        自定义的代码Operator(Tuple[Var])

    
'''
	...
@overload
def code(shapes: List[Tuple[int]], dtypes: List[str], inputs: List[Var]={}, cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str=""):
	'''
    用于方便自定义运算操作的代码 :math:`Operator` 。

    参数:
        - inputs: 一个输入 :math:`jittor Var` 的列表
        - outputs: 一个输出 :math:`jittor Var` 的列表
        - cpu_src: 字符串类型，:math:`cpu` 源代码, 预设数值:
           - :math:`in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)`
           - :math:`out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)`
           - :math:`out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)`
        - cpu_grad_src: 字符串类型列表，:math:`cpu` 梯度源代码, 默认值：空字典
        - cpu_header: 字符串类型，:math:`cpu`  :math:`header` 源代码字符串, 默认值：空字符串
        - cuda_src: 字符串类型，:math:`cuda` 源代码, 默认值：空字符串.
        - cuda_grad_src: 字符串类型列表，:math:`cuda` 梯度源代码, 默认值：空字典
        - cuda_header: 字符串类型，:math:`cuda` _  :math:`header` 源代码字符串, 默认值：空字符串

    代码示例：
        >>> from jittor import Function
        >>> import jittor as jt
        >>> class Func(Function):
        >>> def execute(self, x):
            >>> self.save_vars = x
	    >>> return jt.code(x.shape, x.dtype, [x],cpu_src=""\" for (int i=0; i @alias(a, in0) @alias(b, out) ""\", cpu_src=""\" for (int i=0; i using namespace std; ""\", cpu_src=""\" @alias(a, in0) @alias(b, out0) @alias(c, out1) @b(0) = @c(0) = @a(0); for (int i=0; i0) @b(num_b++) = @a(i); else @c(num_c++) = @a(i); } b->set_shape({num_b}); c->set_shape({num_c}); ""\" ) 
        >>> assert (b.data == [5,3,1]).all()
        >>> assert (c.data == [-4,-2]).all()

    返回值:
        自定义的代码Operator(Tuple[Var])

    
'''
	...
@overload
def code(inputs: List[Var], outputs: List[Var], cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str=""):
	'''
    用于方便自定义运算操作的代码 :math:`Operator` 。

    参数:
        - inputs: 一个输入 :math:`jittor Var` 的列表
        - outputs: 一个输出 :math:`jittor Var` 的列表
        - cpu_src: 字符串类型，:math:`cpu` 源代码, 预设数值:
           - :math:`in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)`
           - :math:`out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)`
           - :math:`out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)`
        - cpu_grad_src: 字符串类型列表，:math:`cpu` 梯度源代码, 默认值：空字典
        - cpu_header: 字符串类型，:math:`cpu`  :math:`header` 源代码字符串, 默认值：空字符串
        - cuda_src: 字符串类型，:math:`cuda` 源代码, 默认值：空字符串.
        - cuda_grad_src: 字符串类型列表，:math:`cuda` 梯度源代码, 默认值：空字典
        - cuda_header: 字符串类型，:math:`cuda` _  :math:`header` 源代码字符串, 默认值：空字符串

    代码示例：
        >>> from jittor import Function
        >>> import jittor as jt
        >>> class Func(Function):
        >>> def execute(self, x):
            >>> self.save_vars = x
	    >>> return jt.code(x.shape, x.dtype, [x],cpu_src=""\" for (int i=0; i @alias(a, in0) @alias(b, out) ""\", cpu_src=""\" for (int i=0; i using namespace std; ""\", cpu_src=""\" @alias(a, in0) @alias(b, out0) @alias(c, out1) @b(0) = @c(0) = @a(0); for (int i=0; i0) @b(num_b++) = @a(i); else @c(num_c++) = @a(i); } b->set_shape({num_b}); c->set_shape({num_c}); ""\" ) 
        >>> assert (b.data == [5,3,1]).all()
        >>> assert (c.data == [-4,-2]).all()

    返回值:
        自定义的代码Operator(Tuple[Var])

    
'''
	...
def copy(x: Var):
 ...
def setitem(x: Var, slices: slice, y: Var, op: str="void"):
 ...
@overload
def broadcast(x: Var, shape: Tuple[int], dims: Tuple[int]=()):
	'''Document:
	*
	    Broadcast ``x`` to a given shape.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] shape:   the output shape.
	
	    * [in] dims:    specifies the new dimension in the output shape, an integer array.
	
	    ----------------
	
	    Example-1::
	        >>> x = jt.randint(0, 10, shape=(2, 2))
	        >>> x
	        jt.Var([[8 1]
	         [7 6]], dtype=int32)
	        >>> jt.broadcast(x, shape=(2, 3, 2), dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)'''
	...
@overload
def broadcast(x: Var, y: Var, dims: Tuple[int]=()):
	'''Document:
	*
	    Broadcast ``x`` to the same shape as ``y``.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] y:       the reference jt.Var.
	
	    * [in] dims:    specifies the new dimension in the output shape, an integer array.
	
	    ----------------
	
	    .. note::
	      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
	
	    Example-1::
	        >>> x = jt.randint(0, 10, shape=(2, 2))
	        >>> x
	        jt.Var([[8 1]
	         [7 6]], dtype=int32)
	        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
	        >>> jt.broadcast(x, y, dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)
	        >>> jt.broadcast_var(x, y, dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)'''
	...
def broadcast_var(x: Var, y: Var, dims: Tuple[int]=()):
	'''Document:
	*
	    Broadcast ``x`` to the same shape as ``y``.
	
	    ----------------
	
	    * [in] x:       the input jt.Var.
	
	    * [in] y:       the reference jt.Var.
	
	    * [in] dims:    specifies the new dimension in the output shape, an integer array.
	
	    ----------------
	
	    .. note::
	      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
	
	    Example-1::
	        >>> x = jt.randint(0, 10, shape=(2, 2))
	        >>> x
	        jt.Var([[8 1]
	         [7 6]], dtype=int32)
	        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
	        >>> jt.broadcast(x, y, dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)
	        >>> jt.broadcast_var(x, y, dims=[1])
	        jt.Var([[[8 1]
	          [8 1]
	          [8 1]],
	         [[7 6]
	          [7 6]
	          [7 6]]], dtype=int32)'''
	...
def reshape(x: Var, shape: Tuple[int]):
	"\n    将输入的 :math:`tensor` 根据指定的形状进行重塑，并返回这个新的 :math:`tensor` 。保持元素的数据和数量不变。其中一维可以是-1，在这种情况下，它是根据其余的维度和输入的元素数量推断出来的。\n\n    参数：\n    - x(jt.Var): 输入的张量。\n    - shape(Tuple[int]): 输出的形状,一个整数列表。\n\n    返回值：\n        重塑过后的 :math:`tensor` （jt.Var）\n \n    代码示例：   \n        >>> jt.randint(0, 10, shape=(12,))\n         jt.Var([4 0 8 4 6 3 1 8 1 1 2 2], dtype=int32)\n        >>> jt.reshape(a, (3, 4))\n         jt.Var([[4 0 8 4]\n               [6 3 1 8]\n               [1 1 2 2]], dtype=int32)\n        >>> jt.reshape(a, (-1, 6))\n        jt.Var([[4 0 8 4 6 3]\n               [1 8 1 1 2 2]], dtype=int32)\n      \n    \n  \n"
	...
def empty(shape: Tuple[int], dtype: str="float32"):
 "\n    通过指定的形状和数据类型创建一个空的tensor。\n\n    参数：\n        - shape(Tuple[int]):创建的tensor的形状。\n        - dtype(str,可选):创建的tensor的数据类型，默认值：'float32'。\n    \n    返回值：\n        创建的空的tensor(Var)\n\n    代码示例：\n        >>> import jittor as jt\n        >>> jt.empty((2, 3))\n\n    \n"
 ...
def reindex_reduce(y: Var, op: str, shape: Tuple[int], indexes: List[str], overflow_conditions: List[str]={}, extras: List[Var]={}):
	'''Document:
	*
	    Reindex Reduce Operator is a many-to-one map operator.
	    It performs equivalent Python-pseudo implementation below::
	
	        # input is y, output is x
	        n = len(y.shape)-1
	        m = len(shape)-1
	        k = len(overflow_conditions)-1
	        x = np.zeros(shape, y.dtype)
	        x[:] = initial_value(op)
	        for i0 in range(y.shape[0]): # 1-st loop
	            for i1 in range(y.shape[1]): # 2-nd loop
	                ...... # many loops
	                for in in range(y.shape[n]) # n+1 -th loop
	                    # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
	                    xi0,xi1,...,xim = indexes[0],indexes[1],...,indexes[m]
	                    if not is_overflow(xi0,xi1,...,xim):
	                        x[xi0,xi1,...,xim] = op(x[xi0,xi1,...,xim], y[i0,i1,...,in])
	
	        # is_overflow is defined as following
	        def is_overflow(xi0,xi1,...,xim):
	            return (
	                xi0 < 0 || xi0 >= shape[0] ||
	                xi1 < 0 || xi1 >= shape[1] ||
	                ......
	                xim < 0 || xim >= shape[m] ||
	
	                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
	                overflow_conditions[0] ||
	                overflow_conditions[1] ||
	                ......
	                overflow_conditions[k]
	            )
	
	    * [in] y:   A input jittor Var
	    
	    * [in] op:  a string represent the reduce operation type
	    
	    * [in] shape:   the output shape, a integer array
	    
	    * [in] indexes: array of c++ style integer expression, its length should be the same with length of output shape, some buildin variables it can use are::
	    
	             XDIM, xshape0, ..., xshapem, xstride0, ..., xstridem
	             YDIM, yshape0, ..., yshapen, ystride0, ..., ystriden
	             i0, i1, ..., in
	             @e0(...), @e1(...) for extras input index
	             e0p, e1p , ... for extras input pointer
	    
	    * [in] overflow_conditions: array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes.
	    
	    * [in] extras:  extra var used for index
	    
	    Example 
	
	    Pooling implemented by reindex operation::
	
	        def pool(x, size, op):
	            N,H,W,C = x.shape
	            h = (H+size-1)//size
	            w = (W+size-1)//size
	            return x.reindex_reduce(op, [N,h,w,C], [
	                "i0", # Nid
	                f"i1/{size}", # Hid
	                f"i2/{size}", # Wid
	                "i3", # Cid
	            ])'''
	...
class Var:
	'''Variable that stores multi-dimensional data.'''
	def ternary(self, x: Var, y: Var): ...
	@overload
	def reindex(self, shape: Tuple[int], indexes: List[str], overflow_value: float=0, overflow_conditions: List[str]={}, extras: List[Var]={}):		
		'''Document:
		* 
		    Reindex Operator is a one-to-many map operator.
		    It performs equivalent Python-pseudo implementation below::
		
		        # input is x, output is y
		        n = len(shape)-1
		        m = len(x.shape)-1
		        k = len(overflow_conditions)-1
		        y = np.zeros(shape, x.dtype)
		        for i0 in range(shape[0]): # 1-st loop
		            for i1 in range(shape[1]): # 2-nd loop
		                ...... # many loops
		                for in in range(shape[n]) # n+1 -th loop
		                    if is_overflow(i0,i1,...,in):
		                        y[i0,i1,...,in] = overflow_value
		                    else:
		                        # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
		                        y[i0,i1,...,in] = x[indexes[0],indexes[1],...,indexes[m]]
		
		        # is_overflow is defined as following
		        def is_overflow(i0,i1,...,in):
		            return (
		                indexes[0] < 0 || indexes[0] >= x.shape[0] ||
		                indexes[1] < 0 || indexes[1] >= x.shape[1] ||
		                ......
		                indexes[m] < 0 || indexes[m] >= x.shape[m] ||
		
		                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
		                overflow_conditions[0] ||
		                overflow_conditions[1] ||
		                ......
		                overflow_conditions[k]
		            )
		    ----------------
		    * [in] x:	A input jittor Var
			
		    * [in] shape:	the output shape, a integer array
			
		    * [in] indexes:	array of c++ style integer expression, its length should be the same with the number of dimension of x, some buildin variables it can use are::
		        
		             XDIM, xshape0, ..., xshapen, xstride0, ..., xstriden
		             YDIM, yshape0, ..., yshapem, ystride0, ..., ystridem
		             i0, i1, ..., in
		             @e0(...), @e1(...) for extras input index
		             e0p, e1p , ... for extras input pointer
					 
		    * [in] overflow_value:	overflow value
			
		    * [in] overflow_conditions:	array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes
				
		    * [in] extras: extra var used for index
			
		    ----------------
		    Example
		    Convolution implemented by reindex operation::
		
		        def conv(x, w):
		            N,H,W,C = x.shape
		            Kh, Kw, _C, Kc = w.shape
		            assert C==_C
		            xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
		                'i0', # Nid
		                'i1+i3', # Hid+Khid
		                'i2+i4', # Wid+KWid
		                'i5', # Cid
		            ])
		            ww = w.broadcast_var(xx)
		            yy = xx*ww
		            y = yy.sum([3,4,5]) # Kh, Kw, C
		            return y, yy'''
		...
	@overload
	def reindex(self, indexes: List[Var], overflow_value: float=0, overflow_conditions: List[str]={}):		
		'''Document:
		* Alias x.reindex([i,j,k]) -> 
		        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])'''
		...
	def reindex_var(self, indexes: List[Var], overflow_value: float=0, overflow_conditions: List[str]={}):		
		'''Document:
		* Alias x.reindex([i,j,k]) -> 
		        x.reindex(i.shape, ['@e0(...)','@e1(...)','@e2(...)',], extras=[i,j,k])'''
		...
	@overload
	def index(self, dim: int, dtype: str="int32"):		
		''' 
``a`` 中每个元素在每一维的下标

    参数：
        - a (Var): 一个张量
        - dtype (str，可选): 返回的数据类型。默认值： ``'int32'``

    返回值：
        ``tuple[Var]`` 包含 ``a.ndims`` 个张量，每个张量形状和 ``a`` 一样，第 :math:`k` 个张量下标 :math:`(i_0, ..., i_k, ...)` 处元素值是 :math:`i_k` ，即第 :math:`k` 维的下标

    代码示例：
        >>> x = jt.zeros((2, 3, 4))
        >>> jt.index_var(x)
        (jt.Var([[[0 0 0 0]
                  [0 0 0 0]
                  [0 0 0 0]]

                 [[1 1 1 1]
                  [1 1 1 1]
                  [1 1 1 1]]], dtype=int32),
         jt.Var([[[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]

                 [[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]], dtype=int32),
         jt.Var([[[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]

                 [[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]], dtype=int32))

'''
		...
	@overload
	def index(self, dtype: str="int32"):		
		''' 
``a`` 中每个元素在每一维的下标

    参数：
        - a (Var): 一个张量
        - dtype (str，可选): 返回的数据类型。默认值： ``'int32'``

    返回值：
        ``tuple[Var]`` 包含 ``a.ndims`` 个张量，每个张量形状和 ``a`` 一样，第 :math:`k` 个张量下标 :math:`(i_0, ..., i_k, ...)` 处元素值是 :math:`i_k` ，即第 :math:`k` 维的下标

    代码示例：
        >>> x = jt.zeros((2, 3, 4))
        >>> jt.index_var(x)
        (jt.Var([[[0 0 0 0]
                  [0 0 0 0]
                  [0 0 0 0]]

                 [[1 1 1 1]
                  [1 1 1 1]
                  [1 1 1 1]]], dtype=int32),
         jt.Var([[[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]

                 [[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]], dtype=int32),
         jt.Var([[[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]

                 [[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]], dtype=int32))

'''
		...
	@overload
	def index_var(self, dim: int, dtype: str="int32"):		
		''' 
``a`` 中每个元素在每一维的下标

    参数：
        - a (Var): 一个张量
        - dtype (str，可选): 返回的数据类型。默认值： ``'int32'``

    返回值：
        ``tuple[Var]`` 包含 ``a.ndims`` 个张量，每个张量形状和 ``a`` 一样，第 :math:`k` 个张量下标 :math:`(i_0, ..., i_k, ...)` 处元素值是 :math:`i_k` ，即第 :math:`k` 维的下标

    代码示例：
        >>> x = jt.zeros((2, 3, 4))
        >>> jt.index_var(x)
        (jt.Var([[[0 0 0 0]
                  [0 0 0 0]
                  [0 0 0 0]]

                 [[1 1 1 1]
                  [1 1 1 1]
                  [1 1 1 1]]], dtype=int32),
         jt.Var([[[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]

                 [[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]], dtype=int32),
         jt.Var([[[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]

                 [[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]], dtype=int32))

'''
		...
	@overload
	def index_var(self, dtype: str="int32"):		
		''' 
``a`` 中每个元素在每一维的下标

    参数：
        - a (Var): 一个张量
        - dtype (str，可选): 返回的数据类型。默认值： ``'int32'``

    返回值：
        ``tuple[Var]`` 包含 ``a.ndims`` 个张量，每个张量形状和 ``a`` 一样，第 :math:`k` 个张量下标 :math:`(i_0, ..., i_k, ...)` 处元素值是 :math:`i_k` ，即第 :math:`k` 维的下标

    代码示例：
        >>> x = jt.zeros((2, 3, 4))
        >>> jt.index_var(x)
        (jt.Var([[[0 0 0 0]
                  [0 0 0 0]
                  [0 0 0 0]]

                 [[1 1 1 1]
                  [1 1 1 1]
                  [1 1 1 1]]], dtype=int32),
         jt.Var([[[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]

                 [[0 0 0 0]
                  [1 1 1 1]
                  [2 2 2 2]]], dtype=int32),
         jt.Var([[[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]

                 [[0 1 2 3]
                  [0 1 2 3]
                  [0 1 2 3]]], dtype=int32))

'''
		...
	def binary(self, y: Var, p: str): ...
	def pow(self, y: Var):		
		''' 
对两个张量逐个元素计算乘方，也可以使用 ``**`` 运算符调用

    参数：
        - x (Var): 乘方运算的底数
        - y (Var): 乘方运算的幂次，形状与 ``x`` 相同

    返回值：
        乘方结果，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 3])
        >>> y = jt.array([2, 3, 4])
        >>> jt.pow(x, y)
        jt.Var([ 1  8 81], dtype=int32)
        >>> x ** y
        jt.Var([ 1  8 81], dtype=int32)

'''
		...
	def maximum(self, y: Var):		
		''' 
对两个张量中对应元素计算最大值

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素中的较大者，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 2, 0])
        >>> jt.maximum(x, y)
        jt.Var([3 2 4], dtype=int32)


'''
		...
	def minimum(self, y: Var):		
		''' 
对两个张量中对应元素计算最小值

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素中的较小者，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 2, 0])
        >>> jt.minimum(x, y)
        jt.Var([1 2 0], dtype=int32)

'''
		...
	def add(self, y: Var):		
		''' 
对两个张量中对应元素计算求和，也可以使用 ``+`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相加结果，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.add(x, y)
        jt.Var([4 6 9], dtype=int32)
        >>> x + y
        jt.Var([4 6 9], dtype=int32)


'''
		...
	def subtract(self, y: Var):		
		''' 
对两个张量中对应元素计算求差，也可以使用 ``-`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相减结果，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.subtract(x, y)
        jt.Var([-2 -2 -1], dtype=int32)
        >>> x - y
        jt.Var([-2 -2 -1], dtype=int32)


'''
		...
	def multiply(self, y: Var):		
		''' 
对两个张量中对应元素计算求积，也可以使用 ``*`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相乘结果，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.multiply(x, y)
        jt.Var([ 3  8 20], dtype=int32)
        >>> x * y
        jt.Var([ 3  8 20], dtype=int32)


'''
		...
	def divide(self, y: Var):		
		''' 
对两个张量中对应元素计算求商，也可以使用 ``/`` 运算符调用。

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相除结果，形状与 ``x`` 和 ``y`` 相同

    注意：
        输入数据类型是整数的情况下，会返回浮点数结果

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.divide(x, y)
        jt.Var([0.3333333  0.49999997 0.8       ], dtype=float32)
        >>> x / y
        jt.Var([0.3333333  0.49999997 0.8       ], dtype=float32)

'''
		...
	def floor_divide(self, y: Var):		
		''' 
对两个张量中对应元素计算求商并向下取整，也可以使用 ``//`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相除向下取整结果，形状与 ``x`` 和 ``y`` 相同，输入数据类型是整数的情况下，会返回整数

    代码示例：
        >>> x = jt.array([3, 4, 5])
        >>> y = jt.array([1, 2, 4])
        >>> jt.floor_divide(x, y)
        jt.Var([3 2 1], dtype=int32)
        >>> x // y
        jt.Var([3 2 1], dtype=int32)

'''
		...
	def mod(self, y: Var):		
		''' 
对两个张量中对应元素计算求余数，也可以使用 ``%`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        ``x`` 和 ``y`` 每对对应元素相除的余数，形状与 ``x`` 和 ``y`` 相同。非零余数的符号和 ``y`` 中元素的符号相同，该行为与 Python 的 ``x % y`` 运算一致

    代码示例：
        >>> x = jt.array([3, 4, 5, -5, 5])
        >>> y = jt.array([1, 2, 3, 3, -3])
        >>> jt.mod(x, y)
        jt.Var([ 0  0  2 -2  2], dtype=int32)
        >>> x % y
        jt.Var([ 0  0  2 -2  2], dtype=int32)

'''
		...
	def less(self, y: Var):		
		''' 
两个张量中对应元素比较是否为小于，把结果布尔值放入输出张量的对应位置，也可以使用 ``<`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        输出张量，形状与 ``x`` 和 ``y`` 相同，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素小于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.less(x, y)
        jt.Var([False False  True], dtype=bool)
        >>> x < y
        jt.Var([False False  True], dtype=bool)


'''
		...
	def less_equal(self, y: Var):		
		''' 
两个张量中对应元素比较是否为小于或等于，把结果布尔值放入输出张量的对应位置，也可以使用 ``<=`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素小于或等于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.less_equal(x, y)
        jt.Var([False  True  True], dtype=bool)
        >>> x <= y
        jt.Var([False  True  True], dtype=bool)

'''
		...
	def greater(self, y: Var):		
		''' 两个张量中对应元素比较是否为大于，把结果布尔值放入输出张量的对应位置，也可以使用 ``>`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素大于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.greater(x, y)
        jt.Var([ True False False], dtype=bool)
        >>> x > y
        jt.Var([ True False False], dtype=bool)

'''
		...
	def greater_equal(self, y: Var):		
		''' 
两个张量中对应元素比较是否为大于或等于，把结果布尔值放入输出张量的对应位置，也可以使用 ``>=`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素大于或等于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.greater_equal(x, y)
        jt.Var([ True  True False], dtype=bool)
        >>> x >= y
        jt.Var([ True  True False], dtype=bool)

'''
		...
	def equal(self, y: Var):		
		''' 
两个张量中对应元素比较是否相等，把结果布尔值放入输出张量的对应位置，也可以使用 ``=`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素等于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.equal(x, y)
        jt.Var([False  True False], dtype=bool)
        >>> x == y
        jt.Var([False  True False], dtype=bool)

'''
		...
	def not_equal(self, y: Var):		
		''' 
两个张量中对应元素比较是否不相等，把结果布尔值放入输出张量的对应位置，也可以使用 ``!=`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，数据类型是 ``bool``。其中每个索引位置的布尔值表示该索引对应的 ``x`` 和 ``y`` 对应元素里是否 ``x`` 中的元素不等于 ``y`` 中的元素

    代码示例：
        >>> x = jt.array([3, 2, 1])
        >>> y = jt.array([1, 2, 3])
        >>> jt.not_equal(x, y)
        jt.Var([ True False  True], dtype=bool)
        >>> x != y
        jt.Var([ True False  True], dtype=bool)


'''
		...
	def left_shift(self, y: Var):		
		''' 
对两个张量中对应元素计算左移移位运算，也可以使用 ``<<`` 运算符调用

    参数：
        - x (Var): 被左移的张量，元素的数据类型是 ``int32`` 或 ``int64``
        - y (Var): 左移的位数张量，元素数据类型是 ``int32`` 或 ``int64`` ，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，其元素为 ``x`` 和 ``y`` 对应索引位置的元素计算左移操作的结果数

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.left_shift(x, y)
        jt.Var([  8  32 128], dtype=int32)
        >>> x << y
        jt.Var([  8  32 128], dtype=int32)

'''
		...
	def right_shift(self, y: Var):		
		''' 
对两个张量中对应元素计算右移移位运算，也可以使用 ``>>`` 运算符调用

    参数：
        - x (Var): 被右移的张量，元素的数据类型是 ``int32`` 或 ``int64``
        - y (Var): 右移的位数张量，元素的数据类型是 ``int32`` 或 ``int64`` ，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，其元素为 ``x`` 和 ``y`` 对应索引位置的元素计算右移操作的结果数

    代码示例：
        >>> x = jt.array([123, 345, -456])
        >>> y = jt.array([3, 4, 5])
        >>> jt.right_shift(x, y)
        jt.Var([ 15  21 -15], dtype=int32)
        >>> x >> y
        jt.Var([ 15  21 -15], dtype=int32)

'''
		...
	def logical_and(self, y: Var):		
		''' 
对两个张量中对应元素计算逻辑与

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
       形状与 ``x`` 和 ``y`` 相同的张量，元素数据类型为 ``bool`` 。其元素为 ``x`` 和 ``y`` 对应索引位置上的元素进行逻辑与操作的结果，即：如果都为非零值或 ``True`` 则输出为 ``True`` ，否则为 ``False`` 。

    代码示例：
        >>> x = jt.array([0, 0, 1, 1])
        >>> y = jt.array([0, 1, 0, 1])
        >>> jt.logical_and(x, y)
        jt.Var([False False False  True], dtype=bool)

'''
		...
	def logical_or(self, y: Var):		
		''' 
对两个张量中对应元素计算逻辑或

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量，元素数据类型为 ``bool`` 。其元素为 ``x`` 和 ``y`` 对应索引位置上的元素进行逻辑或操作的结果，即：如果两者有非零值或 ``True`` 则输出为 ``True`` ，否则为 ``False`` 

    代码示例：
        >>> x = jt.array([0, 0, 1, 1])
        >>> y = jt.array([0, 1, 0, 1])
        >>> jt.logical_or(x, y)
        jt.Var([False  True  True  True], dtype=bool)


'''
		...
	def logical_xor(self, y: Var):		
		''' 
两个张量的元素级逻辑异或

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
       形状与 ``x`` 和 ``y`` 相同的张量，元素数据类型为 ``bool`` 。其元素为 ``x`` 和 ``y`` 对应索引位置上的元素进行逻辑异或操作的结果，即：如果两者有且仅有一个为非零值或 ``True`` 则输出为 ``True`` ，否则为 ``False`` ，数据类型为 ``bool`` ，形状与 ``x`` 和 ``y`` 相同

    代码示例：
        >>> x = jt.array([0, 0, 1, 1])
        >>> y = jt.array([0, 1, 0, 1])
        >>> jt.logical_xor(x, y)
        jt.Var([False  True  True False], dtype=bool)


'''
		...
	def bitwise_and(self, y: Var):		
		''' 
对两个张量中对应元素计算按位与，也可以使用 ``&`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
       形状与 ``x`` 和 ``y`` 相同的张量。 其元素为 ``x`` 和 ``y`` 对应索引位置上元素的按位与结果。

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.bitwise_and(x, y)
        jt.Var([1 0 4], dtype=int32)
        >>> x & y
        jt.Var([1 0 4], dtype=int32)

'''
		...
	def bitwise_or(self, y: Var):		
		''' 
两个张量的元素级按位或，也可以使用 ``|`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量。 ``x`` 和 ``y`` 对应索引位置上元素的按位或结果。

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.bitwise_or(x, y)
        jt.Var([3 6 5], dtype=int32)
        >>> x | y
        jt.Var([3 6 5], dtype=int32)


'''
		...
	def bitwise_xor(self, y: Var):		
		''' 
两个张量的元素级按位异或，也可以使用 ``^`` 运算符调用

    参数：
        - x (Var): 输入数据
        - y (Var): 输入数据，形状与 ``x`` 相同

    返回值：
        形状与 ``x`` 和 ``y`` 相同的张量。 ``x`` 和 ``y`` 对应索引位置上元素的按位异或结果。

    代码示例：
        >>> x = jt.array([1, 2, 4])
        >>> y = jt.array([3, 4, 5])
        >>> jt.bitwise_xor(x, y)
        jt.Var([2 6 1], dtype=int32)
        >>> x ^ y
        jt.Var([2 6 1], dtype=int32)


'''
		...
	def tape(self): ...
	@overload
	def where(self, dtype: str="int32"):		
		''' 
对张量中元素计算条件选择

    参数：
        - cond (Var): 选择条件
        - x (Var): 条件我非零值或 ``True`` 时选择的数据，形状与 ``cond`` 一样
        - y (Var): 条件为零或 ``False`` 时选择的数据，形状与 ``cond`` 一样

    返回值：
        形状与输入都一样，每个值如果对应 ``cond`` 中的值为非零或 ``True`` 则返回 ``x`` 中对应的元素，否则返回 ``y`` 中对应的元素

    代码示例：
        >>> cond = jt.array([1, 0, 1])
        >>> x = jt.array([3, 4, 5])
        >>> y = jt.array([9, 8, 7])
        >>> jt.where(cond, x, y)
        jt.Var([3 8 5], dtype=int32)


'''
		...
	@overload
	def where(self, x: Var, y: Var):		
		''' 
对张量中元素计算条件选择

    参数：
        - cond (Var): 选择条件
        - x (Var): 条件我非零值或 ``True`` 时选择的数据，形状与 ``cond`` 一样
        - y (Var): 条件为零或 ``False`` 时选择的数据，形状与 ``cond`` 一样

    返回值：
        形状与输入都一样，每个值如果对应 ``cond`` 中的值为非零或 ``True`` 则返回 ``x`` 中对应的元素，否则返回 ``y`` 中对应的元素

    代码示例：
        >>> cond = jt.array([1, 0, 1])
        >>> x = jt.array([3, 4, 5])
        >>> y = jt.array([9, 8, 7])
        >>> jt.where(cond, x, y)
        jt.Var([3 8 5], dtype=int32)


'''
		...
	def argsort(self, dim: int=-1, descending: bool=False, dtype: str="int32"):		
		''' 
返回张量在 ``dim`` 维排序的所用的下标排列和排序后的值

    参数：
        - x (Var): 被排序的数据
        - dim (int): 在哪个维度排序，负数表示从后往前数。默认值：-1
        - descending (bool): 是否降序排列，False表示升序排列。默认值：False
        - dtype (str): 返回的下标的数据类型

    返回值：
        ``tuple[Var, Var]`` ，其中第一个返回张量中每个第 ``dim`` 维的切片是一个 ``[0, x.shape[dim])`` 的排列，表示将 ``x`` 中对应的切片中元素排序的下标顺序，第二个返回张量是排序结果

    代码示例：
        >>> x = jt.array([3, 2, 10, 8])
        >>> jt.argsort(x)
        (jt.Var([1 0 3 2], dtype=int32), jt.Var([ 2  3  8 10], dtype=int32))


'''
		...
	def fetch(self, func: Callable): ...
	def arg_reduce(self, op: str, dim: int, keepdims: bool):		
		'''Document:
		*
		    Returns the indices of the maximum / minimum of the input across a dimension.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] op:      "max" or "min". 
		
		    * [in] dim:     int. Specifies which dimension to be reduced.
		
		    * [in] keepdims: bool. Whether the output has ``dim`` retained or not.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 3))
		        >>> x
		        jt.Var([[4 2 5]
		         [6 7 1]], dtype=int32)
		        >>> jt.arg_reduce(x, 'max', dim=1, keepdims=False)
		        [jt.Var([2 1], dtype=int32), jt.Var([5 7], dtype=int32)]
		        >>> jt.arg_reduce(x, 'min', dim=1, keepdims=False)
		        [jt.Var([1 2], dtype=int32), jt.Var([2 1], dtype=int32)]'''
		...
	@overload
	def reduce(self, op: str, dim: int, keepdims: bool=False): ...
	@overload
	def reduce(self, op: str, dims: Tuple[int]=(), keepdims: bool=False): ...
	@overload
	def max(self, dim: int, keepdims: bool=False):		
		'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
		...
	@overload
	def max(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
		...
	@overload
	def max(self, dims_mask: int, keepdims_mask: int):		
		'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
		...
	@overload
	def reduce_maximum(self, dim: int, keepdims: bool=False):		
		'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
		...
	@overload
	def reduce_maximum(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
		...
	@overload
	def reduce_maximum(self, dims_mask: int, keepdims_mask: int):		
		'''
返回输入张量中的最大元素。

    参数:
        - x (Var): 输入的jt.Var.
        - dims_mask (int，可选): 指定的维度，如果指定，则沿给定的维度进行缩减。默认值： ``None`` 
        - keepdims_mask (int，可选): 输出是否保留 ``dim`` 。默认值： ``None`` 
    
    返回值:
        对应维度的最大元素。

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)

'''
		...
	@overload
	def min(self, dim: int, keepdims: bool=False):		
		'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
		...
	@overload
	def min(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
		...
	@overload
	def min(self, dims_mask: int, keepdims_mask: int):		
		'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
		...
	@overload
	def reduce_minimum(self, dim: int, keepdims: bool=False):		
		'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
		...
	@overload
	def reduce_minimum(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
		...
	@overload
	def reduce_minimum(self, dims_mask: int, keepdims_mask: int):		
		'''
    对输入变量进行最小元素操作。

    参数:
        - x (Var): 输入jt.Var。
        - dims_mask (int，可选): 如果指定，沿给定的维度进行缩减。默认为None。
        - keepdims_mask (int, 可选): 输出变量是否保持``dim``。默认为None。

    返回值:
        对应维度最小元素。

    代码示例:

        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
        
    .. note:: 

        ``dim``和``keepdims``是有掩码的int，只有当它们的掩码为真时，才会被视为jt.min的参数。
'''
		...
	@overload
	def sum(self, dim: int, keepdims: bool=False):		
		''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
		...
	@overload
	def sum(self, dims: Tuple[int]=(), keepdims: bool=False):		
		''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
		...
	@overload
	def sum(self, dims_mask: int, keepdims_mask: int):		
		''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
		...
	@overload
	def reduce_add(self, dim: int, keepdims: bool=False):		
		''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
		...
	@overload
	def reduce_add(self, dims: Tuple[int]=(), keepdims: bool=False):		
		''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
		...
	@overload
	def reduce_add(self, dims_mask: int, keepdims_mask: int):		
		''' 
对张量的 ``dims`` 这些维度求和

    参数：
        - x (Var): 输入数据
        - dim (tuple[int]): 在哪些维度求和。默认是全部元素的最大值。默认值： ``()``
        - keepdims (bool): 是否在结果中保留 ``dim`` 维度。默认值：False

    返回值：
        求和的结果。如果 ``keepdims`` 为 ``True`` 则形状为 ``x`` 将 ``dims`` 里给出的维度的长度变成 1，否则是 ``x`` 的形状去掉 ``dims`` 给出的维度

    代码示例：
        >>> x = jt.randint(0, 10, shape=(2, 3))
        >>> x
        jt.Var([[2 7 9]
                [2 6 3]], dtype=int32)
        >>> x.sum()
        jt.Var([29], dtype=int32)
        >>> x.sum((0, 1))
        jt.Var([29], dtype=int32)


'''
		...
	@overload
	def prod(self, dim: int, keepdims: bool=False):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def prod(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def prod(self, dims_mask: int, keepdims_mask: int):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def product(self, dim: int, keepdims: bool=False):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def product(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def product(self, dims_mask: int, keepdims_mask: int):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def reduce_multiply(self, dim: int, keepdims: bool=False):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def reduce_multiply(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def reduce_multiply(self, dims_mask: int, keepdims_mask: int):		
		'''
对输入的数组进行降维求积操作，计算指定维度上元素的乘积。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算乘法的轴，如果是 Tuple 则在这些维度上依次进行降维求积计算，如果是 int 则在这一维度上进行降维求积计算。如果为空，则对输入张量整体计算求积
        - keepdims(bool or int): 如果为 True，则进行求积的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False。
    
    代码示例:
        >>> a = jt.arange(1, 7).reshape(2, 3)
        jt.Var([[1 2 3]
                [4 5 6]], dtype=int32)
        >>> jt.prod(a, 1)
        jt.Var([ 6 120], dtype=int32)
        >>> jt.prod(a, 1, True)
        jt.Var([[ 6]
                [120]], dtype=int32)
        >>> jt.prod(a, (1, 0))
        jt.Var([ 720], dtype=int32)
        >>> jt.prod(a, 0)
        jt.Var([ 4 10 18], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求积后的结果。

'''
		...
	@overload
	def reduce_logical_and(self, dim: int, keepdims: bool=False):		
		'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
		...
	@overload
	def reduce_logical_and(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
		...
	@overload
	def reduce_logical_and(self, dims_mask: int, keepdims_mask: int):		
		'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
		...
	@overload
	def all_(self, dim: int, keepdims: bool=False):		
		'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
		...
	@overload
	def all_(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
		...
	@overload
	def all_(self, dims_mask: int, keepdims_mask: int):		
		'''
对输入的数组计算指定维度上所有元素的逻辑与。

    如果输入 Var 中沿此维度的元素中全为 True，结果对应位置才是 True，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int): 要计算逻辑与的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑与计算，如果是 int 则在这一维度上进行降维求逻辑与计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int): 如果为 True，则进行求逻辑与的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[1,1,1],[1,0,1],[0,1,1]])
        >>> jt.reduce_logical_and(a, 1)
        jt.Var([1 0 0], dtype=int32)
        >>> jt.reduce_logical_and(a, 1, True)
        jt.Var([[1]
                [0]
                [0]], dtype=int32)
        >>> jt.reduce_logical_and(a, (1, 0))
        jt.Var([0], dtype=int32)
        >>> jt.reduce_logical_and(a, 0)
        jt.Var([0 0 1], dtype=int32)

    返回值:
        Var: 返回一个新的数组，进行过一次或多次降维求逻辑与后的结果。

'''
		...
	@overload
	def reduce_logical_or(self, dim: int, keepdims: bool=False):		
		'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
		...
	@overload
	def reduce_logical_or(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
		...
	@overload
	def reduce_logical_or(self, dims_mask: int, keepdims_mask: int):		
		'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
		...
	@overload
	def any_(self, dim: int, keepdims: bool=False):		
		'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
		...
	@overload
	def any_(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
		...
	@overload
	def any_(self, dims_mask: int, keepdims_mask: int):		
		'''
对输入的数组计算指定维度上所有元素的逻辑或。

    如果输入 Var 中沿此维度的元素中全为 False，结果对应位置才是 False，否则为 True。 ``x`` 无论是什么数据类型，都将被转换为 bool 类型来计算。

    参数:
        - x(Var): 输入的张量
        - dims(int, or Tuple of int，可选): 要计算逻辑或的轴，如果是 Tuple 则在这些维度上依次进行降维求逻辑或计算，如果是 int 则在这一个维度上进行降维求逻辑或计算。如果为空则对张量整体求逻辑与运算
        - keepdims(bool or int，可选): 如果为 True，则进行求逻辑或的维度将被保留，否则消除这一维度。如果输入的是 int, 0 被认为是 False，非 0 被认为是 True。默认值：False
    
    代码示例:
        >>> a = jt.array([[0,0,0],[0,1,0],[1,0,0]])
        >>> jt.reduce_logical_or(a, 1)
        jt.Var([0 1 1], dtype=int32)
        >>> jt.reduce_logical_or(a, 1, True)
        jt.Var([[0]
                [1]
                [1]], dtype=int32)
        >>> jt.reduce_logical_or(a, (1, 0))
        jt.Var([1], dtype=int32)
        >>> jt.reduce_logical_or(a, 0)
        jt.Var([1 1 0], dtype=int32)

    返回值:
        Var: 返回一个新的张量，进行过一次或多次降维求逻辑或后的结果。
'''
		...
	@overload
	def reduce_logical_xor(self, dim: int, keepdims: bool=False): ...
	@overload
	def reduce_logical_xor(self, dims: Tuple[int]=(), keepdims: bool=False): ...
	@overload
	def reduce_logical_xor(self, dims_mask: int, keepdims_mask: int): ...
	@overload
	def reduce_bitwise_and(self, dim: int, keepdims: bool=False): ...
	@overload
	def reduce_bitwise_and(self, dims: Tuple[int]=(), keepdims: bool=False): ...
	@overload
	def reduce_bitwise_and(self, dims_mask: int, keepdims_mask: int): ...
	@overload
	def reduce_bitwise_or(self, dim: int, keepdims: bool=False): ...
	@overload
	def reduce_bitwise_or(self, dims: Tuple[int]=(), keepdims: bool=False): ...
	@overload
	def reduce_bitwise_or(self, dims_mask: int, keepdims_mask: int): ...
	@overload
	def reduce_bitwise_xor(self, dim: int, keepdims: bool=False): ...
	@overload
	def reduce_bitwise_xor(self, dims: Tuple[int]=(), keepdims: bool=False): ...
	@overload
	def reduce_bitwise_xor(self, dims_mask: int, keepdims_mask: int): ...
	@overload
	def mean(self, dim: int, keepdims: bool=False):		
		'''
    计算输入张量的均值。

    参数:
        - x (Var): 输入的 :math:`Jittor` 的变量。
        - dims_mask (int): 如果给定，沿着指定的维度缩减。
        - keepdims_mask (bool): 如果为True，则输出结果会保留维度。默认值：False

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[9 4 4]
         [1 9 6]], dtype=int32)
        >>> jt.mean(x)
        jt.Var([5.5000005], dtype=float32)
        >>> x.mean(dim=1)
        jt.Var([5.666667  5.3333335], dtype=float32)
        >>> x.mean(dim=1, keepdims=True)
        jt.Var([[5.666667 ]
         [5.3333335]], dtype=float32)
 
    返回值:
        计算得到的均值(Var)

    

  '''
		...
	@overload
	def mean(self, dims: Tuple[int]=(), keepdims: bool=False):		
		'''
    计算输入张量的均值。

    参数:
        - x (Var): 输入的 :math:`Jittor` 的变量。
        - dims_mask (int): 如果给定，沿着指定的维度缩减。
        - keepdims_mask (bool): 如果为True，则输出结果会保留维度。默认值：False

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[9 4 4]
         [1 9 6]], dtype=int32)
        >>> jt.mean(x)
        jt.Var([5.5000005], dtype=float32)
        >>> x.mean(dim=1)
        jt.Var([5.666667  5.3333335], dtype=float32)
        >>> x.mean(dim=1, keepdims=True)
        jt.Var([[5.666667 ]
         [5.3333335]], dtype=float32)
 
    返回值:
        计算得到的均值(Var)

    

  '''
		...
	@overload
	def mean(self, dims_mask: int, keepdims_mask: int):		
		'''
    计算输入张量的均值。

    参数:
        - x (Var): 输入的 :math:`Jittor` 的变量。
        - dims_mask (int): 如果给定，沿着指定的维度缩减。
        - keepdims_mask (bool): 如果为True，则输出结果会保留维度。默认值：False

    代码示例:
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[9 4 4]
         [1 9 6]], dtype=int32)
        >>> jt.mean(x)
        jt.Var([5.5000005], dtype=float32)
        >>> x.mean(dim=1)
        jt.Var([5.666667  5.3333335], dtype=float32)
        >>> x.mean(dim=1, keepdims=True)
        jt.Var([[5.666667 ]
         [5.3333335]], dtype=float32)
 
    返回值:
        计算得到的均值(Var)

    

  '''
		...
	def clone(self): ...
	def unary(self, op: str): ...
	def cast(self, op: str): ...
	def int8(self):		
		'''  创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``int8`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,3))
        >>> x
        jt.Var([[-0.3373416   0.28905255  1.496737  ]
                [-1.0225579  -1.0431367   0.3423414 ]], dtype=float32)
        >>> x.int8()
        jt.Var([[ 0  0  1]
                [-1 -1  0]], dtype=int8)
        
    返回值：
        返回一个新的张量, 数据类型为 ``int8`` 

    '''
		...
	def int16(self):		
		'''  创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``int16`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,3))
        >>> x
        jt.Var([[ 1.8623732  -1.060122    0.82836497]
                [-0.17111613  1.0357065  -0.4598468 ]], dtype=float32)
        >>> x.int16()
        jt.Var([[ 1 -1  0]
                [ 0  1  0]], dtype=int16)
        
    返回值：
        返回一个新的张量, 数据类型为 ``int16`` 

    '''
		...
	def int32(self):		
		'''     创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``int32`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,3))
        >>> x
        jt.Var([[-1.8707825  -0.5307898  -2.2583888 ]
                [-0.1800911   0.10568491  0.0891161 ]], dtype=float32)
        >>> x.int32()
        jt.Var([[-1  0 -2]
                [ 0  0  0]], dtype=int32)

    返回值：
        返回一个新的张量, 数据类型为 ``int32`` 

    '''
		...
	def int64(self):		
		'''  创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``int64`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,3))
        >>> x
        jt.Var([[-1.4805598  -0.34178254 -0.3623456 ]
                [-0.31025547  0.3924162  -1.5397007 ]], dtype=float32)
        >>> x.int64()
        jt.Var([[-1  0  0]
                [ 0  0 -1]], dtype=int64)

    返回值：
        返回一个新的张量, 数据类型为 ``int64`` 

    '''
		...
	def uint8(self):		
		''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``uint8`` ，数据范围为 ``[0, 255]``

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[-0.7970826   0.99872327]
                [-0.543484   -2.133479  ]], dtype=float32)
        >>> x.uint8()
        jt.Var([[  0   0]
                [  0 254]], dtype=uint8)

    返回值：
        返回一个新的张量, 数据类型为 ``uint8`` 

    '''
		...
	def uint16(self):		
		''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``uint16`` ，数据范围为 ``[0, 65535]``

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[-0.7970826   0.99872327]
                [-0.543484   -2.133479  ]], dtype=float32)
        >>> x.uint16()
        jt.Var([[    0     0]
                [    0 65534]], dtype=uint16)

    返回值：
        返回一个新的张量, 数据类型为 ``uint16`` 

    '''
		...
	def uint32(self):		
		''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``uint32`` ，数据范围为 ``[0, 4294967295]``

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[-0.7970826   0.99872327]
                [-0.543484   -2.133479  ]], dtype=float32)
        >>> x.uint32()
        jt.Var([[         0          0]
                [         0 4294967294]], dtype=uint32)

    返回值：
        返回一个新的张量, 数据类型为 ``uint32`` 

    '''
		...
	def uint64(self):		
		'''     创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``uint64`` ，数据范围为 ``[0, 18446744073709551615]``

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[-0.7970826   0.99872327]
                [-0.543484   -2.133479  ]], dtype=float32)
        >>> x.uint64()
        jt.Var([[                   0                    0]
                [                   0 18446744073709551614]], dtype=uint64)

    返回值：
        返回一个新的张量, 数据类型为 ``uint64`` 

    '''
		...
	def float16(self):		
		''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``float16`` ，返回半精度的浮点数，即用16个比特(2个字节)表示一个数。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[ 1.4581492   0.2615918 ]
                [ 0.26567948 -0.34585235]], dtype=float32)
        >>> x.float16()
        jt.Var([[ 1.458   0.2615]
                [ 0.2656 -0.346 ]], dtype=float16)
    返回值：
        返回一个新的张量, 数据类型为 ``float16`` 

    '''
		...
	def float32(self):		
		'''     创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``float32`` ，返回单精度的浮点数，即用32个比特(4个字节)表示一个数。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.ones((2,2), dtype='int8')
        >>> x
        jt.Var([[1 1]
                [1 1]], dtype=int8)
        >>> x.float32()
        jt.Var([[1. 1.]
                1. 1.]], dtype=float32)
    返回值：
        返回一个新的张量, 数据类型为 ``float32`` 

    '''
		...
	def float64(self):		
		'''     创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``float64`` ，返回双精度的浮点数，即用64个比特(8个字节)表示一个数。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[ 1.4581492   0.2615918 ]
                [ 0.26567948 -0.34585235]], dtype=float32)
        >>> x.float64()
        jt.Var([[ 1.45814919  0.26159179]
                [ 0.26567948 -0.34585235]], dtype=float64)
    返回值：
        返回一个新的张量, 数据类型为 ``float64`` 

    '''
		...
	def abs(self):		
		''' 创建一个张量，其将 ``x`` 中的每一个数值转换为其绝对值。

    .. math::
        out_i = |input_i|


    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> jt.abs(jt.float32([-1, 0, 1]))
        jt.Var([1. 0. 1.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置的绝对值
        
 
'''
		...
	def negative(self):		
		'''     创建一个张量，其将 ``x`` 中的每一个数值转换为其相反数。

    .. math::
        out_i = - input_i


    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> jt. logical_not(jt.float32([-1, 0, 1]))
        jt.Var([1. 0. -1.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置的相反数

    '''
		...
	def logical_not(self):		
		''' 创建一个张量，其将 ``x`` 中的每一个数值转换为其逻辑非, 数据类型为 ``bool``。 如果输入张量 ``x`` 不是数据类型为 ``bool`` 的张量，则零被视为 ``False`` ，非零被视为 ``True``。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> jt.logical_not(jt.float32([-1, 0, 1]))
        jt.Var([False  True False], dtype=bool)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置的逻辑非、

    '''
		...
	def bitwise_not(self):		
		''' 创建一个张量，其将 ``x`` 中的每一个数值进行位非运算。 

    参数：
        - x (Var): Var类型的张量，数据类型应为布尔型或整型
        
    代码示例：
        >>> jt.bitwise_not(jt.int32([-2, 3, 1]))
        jt.Var([ 1 -4 -2], dtype=int32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行位非运算后的值

    '''
		...
	def log(self):		
		'''
    创建一个张量，其将 ``x`` 中的每一个数值的进行自然对数运算。 


    .. math::
        y_i = \ln(x_i)


    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.rand(5) * 5
        >>> x
        jt.Var([2.8649285  0.854602   3.2959728  0.41697088 3.0296502 ], dtype=float32)
        >>> jt.log(x)
        jt.Var([ 1.0525434  -0.15711944  1.1927013  -0.8747389   1.1084472 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行自然对数运算后的值

    注意事项：
        - 自然对数是以常数e（数学常量，大约等于2.71828）为底的对数

    '''
		...
	def exp(self):		
		'''    创建一个张量，其将 ``x`` 中的每一个数值的进行指数运算。 

    .. math::
        y_i = e^{x_i}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.rand(5) * 5
        >>> x
        jt.Var([2.8649285  0.854602   3.2959728  0.41697088 3.0296502 ], dtype=float32)
        >>> jt.exp(x)
        jt.Var([17.547798   2.3504386 27.003672   1.5173583 20.689995 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行指数运算后的值

    '''
		...
	def sqrt(self):		
		'''  创建一个张量，其将 ``x`` 中的每一个数值进行平方根运算。 

    .. math::
        out_i = \sqrt{input_i}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 1.1675875  -0.36471805  2.0246403   1.1496693  -0.09650089], dtype=float32)
        >>> jt.sqrt(x)
        jt.Var([1.0805496       nan 1.4228985 1.0722263       nan], dtype=float32)


    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行平方根运算后的值

    注意事项：
        如果输入元素是负数，返回 ``nan`` 


    
'''
		...
	def round(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为离其最近的整数（四舍五入）。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 1.1675875  -0.36471805  2.0246403   1.1496693  -0.09650089], dtype=float32)
        >>> jt.round(x)
        jt.Var([ 1. -0.  2.  1. -0.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置取最近整数的值

    '''
		...
	def floor(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为向下取整离其最近的整数。
   如：``floor(3.7)`` 的结果是3, ``floor(-3.7)`` 的结果是-4。



    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 1.1675875  -0.36471805  2.0246403   1.1496693  -0.09650089], dtype=float32)
        >>> jt.round(x)
        jt.Var([ 1. -1.  2.  1. -1.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置向下取整离其最近的整数

    '''
		...
	def ceil(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为向上取整离其最近的整数。
   如：``ceil(3.7)`` 的结果是4, ``ceil(-3.7)`` 的结果是-3。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.rand(5) * 5
        >>> x
        jt.Var([ 1.1675875  -0.36471805  2.0246403   1.1496693  -0.09650089], dtype=float32)
        >>> jt.ceil(x)
        jt.Var([ 2. -0.  3.  2. -0.], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置向上取整离其最近的整数

    '''
		...
	def round_int(self):		
		''' 创建一个张量, 其将 ``x`` 中的每一个数值转换为离其最近的整数（四舍五入）, 数据类型为 ``int32`` 。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 2.00422    -2.081075   -0.7260027   0.46558696 -0.1570169 ], dtype=float32)
        >>> jt.round_int(x)
        jt.Var([ 2 -2 -1  0  0], dtype=int32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置取最近整数的值, 数据类型为 ``int32`` 

    '''
		...
	def floor_int(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为向下取整离其最近的整数。
   如：``floor_int(3.7)`` 的结果是3, ``floor_int(-3.7)`` 的结果是-4。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 2.00422    -2.081075   -0.7260027   0.46558696 -0.1570169 ], dtype=float32)
        >>> jt.floor_int(x)
        jt.Var([ 2 -3 -1  0 -1], dtype=int32)

    返回值：
        返回一个新的张量，其数值是 ``x`` 中对应位置向下取整离其最近的整数，数据类型为 ``int32`` 

    '''
		...
	def ceil_int(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值转换为向上取整离其最近的整数。
   如：``ceil_int(3.7)`` 的结果是4, ``ceil_int(-3.7)`` 的结果是-3。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 2.00422    -2.081075   -0.7260027   0.46558696 -0.1570169 ], dtype=float32)
        >>> jt.round_int(x)
        jt.Var([ 3 -2  0  1  0], dtype=int32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应向上取整离其最近的整数, 数据类型为 ``int32`` 

    
'''
		...
	def sin(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行正弦运算。

    .. math::
        y_i = \sin(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-1.6485653  -0.10077649 -0.7546536  -0.02543143  1.0830703 ], dtype=float32)
        >>> jt.sin(x)
        jt.Var([-0.9969775  -0.10060599 -0.68503636 -0.02542868  0.88340074], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行正弦运算的结果

    注意事项：
        - 正弦函数的定义域是实数集，值域是 ``[-1, 1]`` 
        - 支持使用 ``x.sin()`` 进行调用

    
'''
		...
	def asin(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反正弦运算。

    .. math::
        y_i = \sin^{-1}(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-1.6485653  -0.10077649 -0.7546536  -0.02543143  1.0830703 ], dtype=float32)
        >>> jt.asin(x)
        jt.Var([        nan -0.10094785 -0.85512596 -0.02543417         nan], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反正弦运算的结果

    注意事项：
        - 反正弦函数的定义域是 ``[-1, 1]`` ，值域是实数集， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.asin()`` 进行调用

    '''
		...
	def arcsin(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反正弦运算。

    .. math::
        y_i = \sin^{-1}(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-1.6485653  -0.10077649 -0.7546536  -0.02543143  1.0830703 ], dtype=float32)
        >>> jt.asin(x)
        jt.Var([        nan -0.10094785 -0.85512596 -0.02543417         nan], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反正弦运算的结果

    注意事项：
        - 反正弦函数的定义域是 ``[-1, 1]`` ，值域是实数集， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.asin()`` 进行调用

    '''
		...
	def sinh(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行双曲正弦运算。

    .. math::
        y_i = \sinh(x_i) = \frac{e^x - e^{-x}}{2}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 0.50530916 -1.0814334   0.0039318   1.1737247  -1.4199417 ], dtype=float32)
        >>> jt.sin(x)
        jt.Var([ 0.5270894  -1.3048973   0.00393181  1.4624014  -1.9475756 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行双曲正弦运算的结果

    注意事项：
        - 正弦函数的定义域和值域是实数集
        - 支持使用 ``x.sinh()`` 进行调用

    
'''
		...
	def asinh(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲正弦运算。

    .. math::
        y_i = \sinh^{-1}(x_i) = \ln\left(x_i + \sqrt{x_i^2 + 1}\right)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 0.50530916 -1.0814334   0.0039318   1.1737247  -1.4199417 ], dtype=float32)
        >>> jt.asin(x)
        jt.Var([ 0.48595542 -0.9377998   0.00393179  0.99904275 -1.1495185 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲正弦运算的结果

    注意事项：
        - 反正弦函数的定义域和值域是实数集
        - 支持使用 ``x.asinh()`` 进行调用

    '''
		...
	def arcsinh(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲正弦运算。

    .. math::
        y_i = \sinh^{-1}(x_i) = \ln\left(x_i + \sqrt{x_i^2 + 1}\right)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([ 0.50530916 -1.0814334   0.0039318   1.1737247  -1.4199417 ], dtype=float32)
        >>> jt.asin(x)
        jt.Var([ 0.48595542 -0.9377998   0.00393179  0.99904275 -1.1495185 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲正弦运算的结果

    注意事项：
        - 反正弦函数的定义域和值域是实数集
        - 支持使用 ``x.asinh()`` 进行调用

    '''
		...
	def tan(self):		
		''' 创建一个张量, 其将 ``x`` 中的每一个数值进行正切运算，即正弦值除以余弦值。

    .. math::
        y_i = \tan(x_i) = \frac{\sin(x)}{\cos(x)}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-0.1146331  -0.22897322 -1.1813502   0.7146521  -0.02658333], dtype=float32)
        >>> jt.tan(x)
        jt.Var([-0.11513788 -0.23306055 -2.4366024   0.8676503  -0.02658959], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行正切运算的结果

    注意事项：
        - 该函数的值域为实数集，定义域为

        .. math:: 
            x \in \mathbb{R} \setminus \left\{ \left(2k+1\right)\frac{\pi}{2} \mid k \in \mathbb{Z} \right\}


        - 支持使用 ``x.tan()`` 进行调用

    '''
		...
	def atan(self):		
		''' 创建一个张量, 其将 ``x`` 中的每一个数值进行反正切运算。

.. math::
    y_i = \tan^{-1}(x_i) 

参数：
    - x (Var): Var类型的张量
    
代码示例：
    >>> x = jt.randn(5)
    >>> x
    jt.Var([-0.1146331  -0.22897322 -1.1813502   0.7146521  -0.02658333], dtype=float32)
    >>> jt.atan(x)
    jt.Var([-0.1141349  -0.22509298 -0.86834407  0.6204921  -0.02657707], dtype=float32)

返回值：
    返回一个新的张量, 其数值是 ``x`` 中对应位置进行反正切运算的结果

注意事项：
    - 反正弦函数的定义域为实数集，值域为

    .. math:: 
        \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)

    - 支持使用 ``x.tan()`` 进行调用


'''
		...
	def arctan(self):		
		''' 创建一个张量, 其将 ``x`` 中的每一个数值进行反正切运算。

.. math::
    y_i = \tan^{-1}(x_i) 

参数：
    - x (Var): Var类型的张量
    
代码示例：
    >>> x = jt.randn(5)
    >>> x
    jt.Var([-0.1146331  -0.22897322 -1.1813502   0.7146521  -0.02658333], dtype=float32)
    >>> jt.atan(x)
    jt.Var([-0.1141349  -0.22509298 -0.86834407  0.6204921  -0.02657707], dtype=float32)

返回值：
    返回一个新的张量, 其数值是 ``x`` 中对应位置进行反正切运算的结果

注意事项：
    - 反正弦函数的定义域为实数集，值域为

    .. math:: 
        \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)

    - 支持使用 ``x.tan()`` 进行调用


'''
		...
	def tanh(self):		
		'''  创建一个张量, 其将 ``x`` 中的每一个数值进行双曲正切运算。

    .. math::
	    \tanh(x) = \frac{{e^{x} - e^{-x}}}{{e^{x} + e^{-x}}}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 0.561267    1.7155168  -0.53718305  1.6382825   0.5129911 ], dtype=float32)
        >>> jt.tanh(x)
        jt.Var([ 0.508917    0.93732095 -0.4908527   0.9272321   0.47227246], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行双曲正切运算的结果

    注意事项：
        - 双曲正切运算的定义域为所有实数，值域为 (-1, 1)
        - 支持使用 ``x.tanh()`` 进行调用

    '''
		...
	def atanh(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲正切运算。

    .. math::
        atanh(x) = \frac{1}{2} * log(\frac{1 + x}{1 - x})

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 0.561267    1.7155168  -0.53718305  1.6382825   0.5129911 ], dtype=float32)
        >>> jt.atanh(x)
        jt.Var([ 0.634681         nan -0.6001876        nan  0.5667807], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲正切运算的结果

    注意事项：
        - 该函数的定义域是(-1, 1)，值域是实数集，如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.atanh()`` 进行调用

    '''
		...
	def arctanh(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲正切运算。

    .. math::
        atanh(x) = \frac{1}{2} * log(\frac{1 + x}{1 - x})

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 0.561267    1.7155168  -0.53718305  1.6382825   0.5129911 ], dtype=float32)
        >>> jt.atanh(x)
        jt.Var([ 0.634681         nan -0.6001876        nan  0.5667807], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲正切运算的结果

    注意事项：
        - 该函数的定义域是(-1, 1)，值域是实数集，如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.atanh()`` 进行调用

    '''
		...
	def cos(self):		
		''' 创建一个张量, 其将 ``x`` 中的每一个数值进行余弦运算。

    .. math::
        y_i = \cos(x_i) 

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-0.8017247   2.4553642  -0.57173574 -0.06912863  1.5478854 ], dtype=float32)
        >>> jt.cos(x)
        jt.Var([ 0.6954685  -0.7736413   0.84096307  0.9976116   0.0229089 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行余弦运算的结果

    注意事项：
        - 余弦函数的定义域是实数集，值域是 ``[-1, 1]`` 
        - 支持使用 ``x.cos()`` 进行调用

    
'''
		...
	def acos(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反余弦运算。

    .. math::
        y_i = \cos^{-1}(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-0.8017247   2.4553642  -0.57173574 -0.06912863  1.5478854 ], dtype=float32)
        >>> jt.acos(x)
        jt.Var([2.5009716       nan 2.1794162 1.6399801       nan], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反余弦运算的结果

    注意事项：
        - 该函数的定义域是 ``[-1, 1]`` ，值域是[0,pi]， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.acos()`` 进行调用

    '''
		...
	def arccos(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反余弦运算。

    .. math::
        y_i = \cos^{-1}(x_i)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5)
        >>> x
        jt.Var([-0.8017247   2.4553642  -0.57173574 -0.06912863  1.5478854 ], dtype=float32)
        >>> jt.acos(x)
        jt.Var([2.5009716       nan 2.1794162 1.6399801       nan], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反余弦运算的结果

    注意事项：
        - 该函数的定义域是 ``[-1, 1]`` ，值域是[0,pi]， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.acos()`` 进行调用

    '''
		...
	def cosh(self):		
		'''  创建一个张量, 其将 ``x`` 中的每一个数值进行双曲余弦运算。

    .. math::
        y_i = \cosh(x_i) = \frac{e^x_i + e^{-x_i}}{2}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) + 1
        >>> x
        jt.Var([-0.46972263 -0.6531948   1.7161269   1.007987    1.3451817 ], dtype=float32)
        >>> jt.cosh(x)
        jt.Var([1.1123631 1.2210255 2.871351  1.5525162 2.0496883], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行双曲余弦运算的结果

    注意事项：
        - 双曲余弦运算的定义域和值域为 [1, ∞) 
        - 支持使用 ``x.cosh()`` 进行调用

    '''
		...
	def acosh(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲余弦运算。

    .. math::
        y_i = \cosh^{-1}(x_i) 

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) + 1
        >>> x
        jt.Var([-0.46972263 -0.6531948   1.7161269   1.007987    1.3451817 ], dtype=float32)
        >>> jt.acosh(x)
        jt.Var([      nan       nan 1.1348774 0.1263045 0.8086661], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲余弦运算的结果

    注意事项：
        - 该函数的定义域是[1, ∞)，值域是实数集， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.acosh()`` 进行调用

    '''
		...
	def arccosh(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行反双曲余弦运算。

    .. math::
        y_i = \cosh^{-1}(x_i) 

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) + 1
        >>> x
        jt.Var([-0.46972263 -0.6531948   1.7161269   1.007987    1.3451817 ], dtype=float32)
        >>> jt.acosh(x)
        jt.Var([      nan       nan 1.1348774 0.1263045 0.8086661], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反双曲余弦运算的结果

    注意事项：
        - 该函数的定义域是[1, ∞)，值域是实数集， 如果定义域不在规定范围内， 返回 ``nan``
        - 支持使用 ``x.acosh()`` 进行调用

    '''
		...
	def sigmoid(self):		
		'''    创建一个张量, 其将 ``x`` 中的每一个数值进行 ``sigmoid`` 变换。

    .. math::
         out_i =  \frac{1}{1 + e^{-x_i}}

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 2.7442768  -0.6135011  -0.09157802 -2.3600576   0.98465353], dtype=float32)
        >>> jt.sigmoid(x)
        jt.Var([0.93958926 0.35126096 0.47712144 0.08626965 0.72803056], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行 ``sigmoid`` 变换后的值

    注意事项：
        -  ``sigmoid`` 变换的定义域是实数集，值域是(0, 1)
        - 支持使用 ``x.sigmoid()`` 进行调用

    '''
		...
	def erf(self):		
		'''    创建一个张量,  ``x`` 中的每一个数值进行误差变换， 该函数为误差函数。

    .. math::
        \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} \, dt

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 2.7442768  -0.6135011  -0.09157802 -2.3600576   0.98465353], dtype=float32)
        >>> jt.erf(x)
        jt.Var([-0.91196376  0.19675992  0.9940873  -0.99938697  0.10797469], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行误差函数运算

    注意事项：
        - 该函数定义域是实数集，值域是 [-1,1]
        - 支持使用 ``x.erf()`` 进行调用

    '''
		...
	def erfinv(self):		
		'''    创建一个张量,  ``x`` 中的每一个数值进行反误差变换， 该函数为反误差函数。

    .. math::
        \text{erfinv}(x) = \text{erf}^{-1}(x)

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn(5) 
        >>> x
        jt.Var([ 0.73859763  1.0348558   2.4863744  -0.36372563 -0.6050172 ], dtype=float32)
        >>> jt.erfinv(x)
        jt.Var([ 0.79413944         nan         nan -0.33440086 -0.6014762 ], dtype=float32)

    返回值：
        返回一个新的张量, 其数值是 ``x`` 中对应位置进行反误差函数运算

    注意事项：
        - 该函数定义域是[-1,1]，值域是实数集
        - 支持使用 ``x.erfinv()`` 进行调用

    '''
		...
	def transpose(self, axes: Tuple[int]=()): ...
	def fuse_transpose(self, axes: Tuple[int]=()): ...
	def safe_clip(self, left: float, right: float):		
		'''def safe_clip(x: Var, left: float, right: float) ->Var:
    对一个值在一个区间内进行安全剪裁,并保持梯度通过。当输入值在剪裁区间之外时,其梯度保持不变。

    公式如下：
        如果 :math:`left`<  ``x``  < :math:`right` ,那么返回值 :math:`y`  = ``x`` 。否则，如果 :math:`x`  <= :math:`left` ,那么 :math:`y`  =  :math:`left` ；如果 ``x``  >=  :math:`right` ,那么 :math:`y` =  :math:`right` 。

    参数：
        - x(Var):输入值。
        - left(float):剪裁的最小值。默认值：float64
        - right(float):剪裁的最大值。默认值：float64

    代码示例：
        >>> import jittor as jt
        >>> x = jt.Var([-1,0.75,2])
        >>> jt.safe_clip(x, 0.5, 1)
        [0.5,0.75,1.]
    返回值：
        剪裁后的值(Var)。




'''
		...
	def array(self): ...
	@overload
	def getitem(self, slices: slice): ...
	@overload
	def getitem(self, slices: slice, _: int): ...
	def candidate(self, fail_cond: str, dtype: str="int32"):		
		'''Document:
		*
		    Candidate Operator Perform an indirect candidate filter by given a fail condition.
		    
		    x is input, y is output index, satisfy::
		
		        not fail_cond(y[0], y[1]) and
		        not fail_cond(y[0], y[2]) and not fail_cond(y[1], y[2]) and
		        ...
		        ... and not fail_cond(y[m-2], y[m-1])
		
		    Where m is number of selected candidates.
		
		    Pseudo code::
		    
		        y = []
		        for i in range(n):
		            pass = True
		            for j in y:
		                if (@fail_cond):
		                    pass = false
		                    break
		            if (pass):
		                y.append(i)
		        return y
		
		    * [in] x:   input var for filter
		
		    * [in] fail_cond:   code for fail condition
		
		    * [in] dtype:   type of return indexes
		
		    * [out] index: .
		
		    Example::
		
		        jt.candidate(jt.random(100,2), '(@x(j,0)>@x(i,0))or(@x(j,1)>@x(i,1))')
		        # return y satisfy:
		        #    x[y[0], 0] <= x[y[1], 0] and x[y[1], 0] <= x[y[2], 0] and ... and x[y[m-2], 0] <= x[y[m-1], 0] and
		        #    x[y[0], 1] <= x[y[1], 1] and x[y[1], 1] <= x[y[2], 1] and ... and x[y[m-2], 1] <= x[y[m-1], 1]'''
		...
	@overload
	def code(self, outputs: List[Var], cpu_src: str="", cpu_grad_src: List[str]={}, cpu_header: str="", cuda_src: str="", cuda_grad_src: List[str]={}, cuda_header: str=""):		
		'''
    用于方便自定义运算操作的代码 :math:`Operator` 。

    参数:
        - inputs: 一个输入 :math:`jittor Var` 的列表
        - outputs: 一个输出 :math:`jittor Var` 的列表
        - cpu_src: 字符串类型，:math:`cpu` 源代码, 预设数值:
           - :math:`in{x}, in{x}_shape{y}, in{x}_stride{y}, in{x}_type, in{x}_p, @in0(...)`
           - :math:`out{x}, out{x}_shape{y}, out{x}_stride{y}, out{x}_type, out{x}_p, @out0(...)`
           - :math:`out, out_shape{y}, out_stride{y}, out_type, out_p, @out(...)`
        - cpu_grad_src: 字符串类型列表，:math:`cpu` 梯度源代码, 默认值：空字典
        - cpu_header: 字符串类型，:math:`cpu`  :math:`header` 源代码字符串, 默认值：空字符串
        - cuda_src: 字符串类型，:math:`cuda` 源代码, 默认值：空字符串.
        - cuda_grad_src: 字符串类型列表，:math:`cuda` 梯度源代码, 默认值：空字典
        - cuda_header: 字符串类型，:math:`cuda` _  :math:`header` 源代码字符串, 默认值：空字符串

    代码示例：
        >>> from jittor import Function
        >>> import jittor as jt
        >>> class Func(Function):
        >>> def execute(self, x):
            >>> self.save_vars = x
	    >>> return jt.code(x.shape, x.dtype, [x],cpu_src=""\" for (int i=0; i @alias(a, in0) @alias(b, out) ""\", cpu_src=""\" for (int i=0; i using namespace std; ""\", cpu_src=""\" @alias(a, in0) @alias(b, out0) @alias(c, out1) @b(0) = @c(0) = @a(0); for (int i=0; i0) @b(num_b++) = @a(i); else @c(num_c++) = @a(i); } b->set_shape({num_b}); c->set_shape({num_c}); ""\" ) 
        >>> assert (b.data == [5,3,1]).all()
        >>> assert (c.data == [-4,-2]).all()

    返回值:
        自定义的代码Operator(Tuple[Var])

    
'''
		...
	def copy(self): ...
	def setitem(self, slices: slice, y: Var, op: str="void"): ...
	@overload
	def broadcast(self, shape: Tuple[int], dims: Tuple[int]=()):		
		'''Document:
		*
		    Broadcast ``x`` to a given shape.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] shape:   the output shape.
		
		    * [in] dims:    specifies the new dimension in the output shape, an integer array.
		
		    ----------------
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 2))
		        >>> x
		        jt.Var([[8 1]
		         [7 6]], dtype=int32)
		        >>> jt.broadcast(x, shape=(2, 3, 2), dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)'''
		...
	@overload
	def broadcast(self, y: Var, dims: Tuple[int]=()):		
		'''Document:
		*
		    Broadcast ``x`` to the same shape as ``y``.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] y:       the reference jt.Var.
		
		    * [in] dims:    specifies the new dimension in the output shape, an integer array.
		
		    ----------------
		
		    .. note::
		      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 2))
		        >>> x
		        jt.Var([[8 1]
		         [7 6]], dtype=int32)
		        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
		        >>> jt.broadcast(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)
		        >>> jt.broadcast_var(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)'''
		...
	def broadcast_var(self, y: Var, dims: Tuple[int]=()):		
		'''Document:
		*
		    Broadcast ``x`` to the same shape as ``y``.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] y:       the reference jt.Var.
		
		    * [in] dims:    specifies the new dimension in the output shape, an integer array.
		
		    ----------------
		
		    .. note::
		      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 2))
		        >>> x
		        jt.Var([[8 1]
		         [7 6]], dtype=int32)
		        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
		        >>> jt.broadcast(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)
		        >>> jt.broadcast_var(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)'''
		...
	def reshape(self, shape: Tuple[int]):		
		'''Document:
		*
		    Returns a tensor with the same data and number of elements as input, but with the specified shape. 
		
		    A single dimension may be -1, in which case it's inferred from the remaining dimensions and the number of elements in input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var
		
		    * [in] shape:   the output shape, an integer array
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.randint(0, 10, shape=(12,))
		        >>> a
		        jt.Var([4 0 8 4 6 3 1 8 1 1 2 2], dtype=int32)
		        >>> jt.reshape(a, (3, 4))
		        jt.Var([[4 0 8 4]
		         [6 3 1 8]
		         [1 1 2 2]], dtype=int32)
		        >>> jt.reshape(a, (-1, 6))
		        jt.Var([[4 0 8 4 6 3]
		         [1 8 1 1 2 2]], dtype=int32)'''
		...
	def reindex_reduce(self, op: str, shape: Tuple[int], indexes: List[str], overflow_conditions: List[str]={}, extras: List[Var]={}):		
		'''Document:
		*
		    Reindex Reduce Operator is a many-to-one map operator.
		    It performs equivalent Python-pseudo implementation below::
		
		        # input is y, output is x
		        n = len(y.shape)-1
		        m = len(shape)-1
		        k = len(overflow_conditions)-1
		        x = np.zeros(shape, y.dtype)
		        x[:] = initial_value(op)
		        for i0 in range(y.shape[0]): # 1-st loop
		            for i1 in range(y.shape[1]): # 2-nd loop
		                ...... # many loops
		                for in in range(y.shape[n]) # n+1 -th loop
		                    # indexes[i] is a c++ style integer expression consisting of i0,i1,...,in
		                    xi0,xi1,...,xim = indexes[0],indexes[1],...,indexes[m]
		                    if not is_overflow(xi0,xi1,...,xim):
		                        x[xi0,xi1,...,xim] = op(x[xi0,xi1,...,xim], y[i0,i1,...,in])
		
		        # is_overflow is defined as following
		        def is_overflow(xi0,xi1,...,xim):
		            return (
		                xi0 < 0 || xi0 >= shape[0] ||
		                xi1 < 0 || xi1 >= shape[1] ||
		                ......
		                xim < 0 || xim >= shape[m] ||
		
		                # overflow_conditions[i] is a c++ style boolean expression consisting of i0,i1,...,in
		                overflow_conditions[0] ||
		                overflow_conditions[1] ||
		                ......
		                overflow_conditions[k]
		            )
		
		    * [in] y:   A input jittor Var
		    
		    * [in] op:  a string represent the reduce operation type
		    
		    * [in] shape:   the output shape, a integer array
		    
		    * [in] indexes: array of c++ style integer expression, its length should be the same with length of output shape, some buildin variables it can use are::
		    
		             XDIM, xshape0, ..., xshapem, xstride0, ..., xstridem
		             YDIM, yshape0, ..., yshapen, ystride0, ..., ystriden
		             i0, i1, ..., in
		             @e0(...), @e1(...) for extras input index
		             e0p, e1p , ... for extras input pointer
		    
		    * [in] overflow_conditions: array of c++ style boolean expression, it length can be vary. the buildin variables it can use are the same with indexes.
		    
		    * [in] extras:  extra var used for index
		    
		    Example 
		
		    Pooling implemented by reindex operation::
		
		        def pool(x, size, op):
		            N,H,W,C = x.shape
		            h = (H+size-1)//size
		            w = (W+size-1)//size
		            return x.reindex_reduce(op, [N,h,w,C], [
		                "i0", # Nid
		                f"i1/{size}", # Hid
		                f"i2/{size}", # Wid
		                "i3", # Cid
		            ])'''
		...
	def sync(self, device_sync: bool=False, weak_sync: bool=True): ...
	def fetch_sync(self):		
		'''Document:
		*
		     * Returns a numpy array copy of the Var.'''
		...
	def numpy(self):		
		'''Document:
		*
		     * Returns a numpy array copy of the Var.'''
		...
	def assign(self, v: Var):		
		'''Document:
		*
		     * assign the data from another Var.'''
		...
	def update(self, v: Var):		
		'''Document:
		*
		     * update parameter and global variable,
		     * different from assign, it will
		     * stop grad between origin var and assigned var, and
		     * will update in the background'''
		...
	def _update(self, v: Var):		
		'''Document:
		*
		     * update parameter without set attribute.'''
		...
	def swap(self, v: Var):		
		'''Document:
		*
		     * swap the data with another Var.'''
		...
	@overload
	def name(self, s: str):		
		'''Document:
		* 
		     * set the name of the Var.'''
		...
	@overload
	def name(self):		
		'''Document:
		* 
		     * set the name of the Var.'''
		...
	def numel(self):		
		'''Document:
		* 
		     * return the number of elements in the Var.'''
		...
	def stop_grad(self):		
		'''Document:
		* 
		     * disable the gradient calculation for the Var.'''
		...
	def is_stop_grad(self):		
		'''Document:
		*
		     * return True if the gradient is stopped.'''
		...
	def detach(self):		
		'''
创建一个已经存在的张量 ``x`` 的副本，这个副本与原始张量共享数据，但不参与梯度计算。这在神经网络的训练过程中非常重要，尤其是在需要操作数据但又不想影响梯度计算时。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((3,2))
        jt.Var([[-1.7332076 -3.9374337]
                [ 2.2316337  1.2128713]
                [-2.067835   0.8186041]], dtype=float32)
        >>> x.detach()
        jt.Var([[-1.7332076 -3.9374337]
                [ 2.2316337  1.2128713]
                [-2.067835   0.8186041]], dtype=float32)
        
    返回值：
        返回副本张量（Var）

'''
		...
	def stop_fuse(self):		
		'''Document:
		*
		     * stop operator fusion.'''
		...
	def is_stop_fuse(self):		
		'''Document:
		*
		     * return True if operator fusion is stopped.'''
		...
	def out_hint(self):		
		'''Document:
		*
		     * output hint for training optimization'''
		...
	def start_grad(self):		
		'''Document:
		* 
		     * enable the gradient calculation for the Var.'''
		...
	def item(self):		
		'''Document:
		*
		     * returns the Python number if the Var contains only one element.
		     * For other cases, see data().'''
		...
	def share_with(self, other: Var): ...
	def debug_msg(self):		
		'''Document:
		*
		     * print the information of the Var to debug.'''
		...
	def _input(self, i: int): ...
	def _add_dependency(self, vars: List[Var]):		
		'''Document:
		 Add dependency, make var computed after vars'''
		...
	def compile_options(self): ...
	def data(self):		
		'''Document:
		*
		     * get a numpy array which shares the data with the Var.'''
		...
	def dtype(self):		
		'''Document:
		*
		     * return the data type of the Var.'''
		...
	def grad(self):		
		'''Document:
		 Jittor Var doesn't have this interface, please change your code as below::
		
		    model = Model()
		    optimizer = SGD(model.parameters())
		    ...
		    optimizer.backward(loss)
		    
		    for p in model.parameters():
		        # prev code:
		        # grad = p.grad
		
		        # change to:
		        grad = p.opt_grad(optimizer)'''
		...
	def ndim(self):		
		'''Document:
		*
		     * return the number of dimensions.'''
		...
	def requires_grad(self):		
		'''Document:
		* 
		     * return True if the Var requires gradient calculation.
		     * @see is_stop_grad'''
		...
	def shape(self):		
		'''Document:
		* 
		     * return the shape of the Var.'''
		...
	def uncertain_shape(self): ...
	def view(self, x: Var, shape: Tuple[int]):		
		'''Document:
		*
		    Returns a tensor with the same data and number of elements as input, but with the specified shape. 
		
		    A single dimension may be -1, in which case it's inferred from the remaining dimensions and the number of elements in input.
		
		    ----------------
		
		    * [in] x:       the input jt.Var
		
		    * [in] shape:   the output shape, an integer array
		
		    ----------------
		
		    Example-1::
		        >>> a = jt.randint(0, 10, shape=(12,))
		        >>> a
		        jt.Var([4 0 8 4 6 3 1 8 1 1 2 2], dtype=int32)
		        >>> jt.reshape(a, (3, 4))
		        jt.Var([[4 0 8 4]
		         [6 3 1 8]
		         [1 1 2 2]], dtype=int32)
		        >>> jt.reshape(a, (-1, 6))
		        jt.Var([[4 0 8 4 6 3]
		         [1 8 1 1 2 2]], dtype=int32)'''
		...
	def permute(self, x: Var, axes: Tuple[int]=()): ...
	def detach_inplace(self):		
		'''Document:
		* 
		     * enable the gradient calculation for the Var.'''
		...
	def astype(self, x: Var, op: str): ...
	def half(self, x: Var):		
		''' 创建并返回一个 ``x`` 的张量副本，并将其类型转换为 ``float16`` ，返回半精度的浮点数，即用16个比特(2个字节)表示一个数。

    参数：
        - x (Var): Var类型的张量
        
    代码示例：
        >>> x = jt.randn((2,2))
        >>> x
        jt.Var([[ 1.4581492   0.2615918 ]
                [ 0.26567948 -0.34585235]], dtype=float32)
        >>> x.float16()
        jt.Var([[ 1.458   0.2615]
                [ 0.2656 -0.346 ]], dtype=float16)
    返回值：
        返回一个新的张量, 数据类型为 ``float16`` 

    '''
		...
	def expand_as(self, x: Var, y: Var, dims: Tuple[int]=()):		
		'''Document:
		*
		    Broadcast ``x`` to the same shape as ``y``.
		
		    ----------------
		
		    * [in] x:       the input jt.Var.
		
		    * [in] y:       the reference jt.Var.
		
		    * [in] dims:    specifies the new dimension in the output shape, an integer array.
		
		    ----------------
		
		    .. note::
		      jt.broadcast_var(x, y, dims) is an alias of jt.broadcast(x, y, dims)
		
		    Example-1::
		        >>> x = jt.randint(0, 10, shape=(2, 2))
		        >>> x
		        jt.Var([[8 1]
		         [7 6]], dtype=int32)
		        >>> y = jt.randint(0, 10, shape=(2, 3, 2))
		        >>> jt.broadcast(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)
		        >>> jt.broadcast_var(x, y, dims=[1])
		        jt.Var([[[8 1]
		          [8 1]
		          [8 1]],
		         [[7 6]
		          [7 6]
		          [7 6]]], dtype=int32)'''
		...
class Flags:
	'''A set of flags to configure jittor running behaviors'''
	addr2line_path: str
	'''Path of addr2line. Default: ""'''
	amp_level: int
	'''Auto mixed-precision optimization level, 0: not use fp16, 1-3: preserve level, not use fp16 for now; 4: perfer fp16, but some ops use fp32 e.g. sum,exp; 5: simular with 4, and array op will automatically convert to fp16; 6: all ops prefer fp16. Default: 0'''
	amp_reg: int
	'''Auto mixed-precision control registers, bit 0: prefer 32; bit 1: prefer 16; bit 2: keep reduce type; bit 3 keep white list type; bit 4: array like op prefer too. Default: 0'''
	auto_convert_64_to_32: int
	'''auto convert 64bit numpy array into 32bit jittor array. Default: 1'''
	auto_mixed_precision_level: int
	'''Auto mixed-precision optimization level, 0: not use fp16, 1-3: preserve level, not use fp16 for now; 4: perfer fp16, but some ops use fp32 e.g. sum,exp; 5: simular with 4, and array op will automatically convert to fp16; 6: all ops prefer fp16. Default: 0'''
	cache_path: str
	'''Cache path of jittor. Default: ""'''
	cc_flags: str
	'''Flags of C++ compiler. Default: ""'''
	cc_path: str
	'''Path of C++ compiler. Default: ""'''
	cc_type: str
	'''Type of C++ compiler(clang, icc, g++). Default: ""): Type of C++ compiler(clang, icc, g++'''
	check_graph: int
	'''Unify graph sanity check. Default: 0'''
	compile_options: Any
	'''Override the default loop transfrom options. Default: {}'''
	disable_lock: bool
	'''Disable file lock. Default: 0'''
	enable_tuner: int
	'''Enable tuner. Default: 1'''
	exclude_pass: str
	'''Don't run certain pass. Default: ""'''
	extra_gdb_cmd: str
	'''Extra command pass to GDB, seperate by(;) . Default: ""): Extra command pass to GDB, seperate by(;'''
	gdb_attach: int
	'''gdb attach self process. Default: 0'''
	gdb_path: str
	'''Path of GDB. Default: ""'''
	gopt_disable: int
	'''Disable graph optimizer. Default: 0'''
	has_pybt: int
	'''GDB has pybt or not. Default: 0'''
	jit_search_kernel: int
	'''Jit search for the fastest kernel. Default: 0'''
	jit_search_rerun: int
	'''. Default: 10'''
	jit_search_warmup: int
	'''. Default: 2'''
	jittor_path: str
	'''Source path of jittor. Default: ""'''
	l1_cache_size: int
	'''size of level 1 cache (byte). Default: 32768): size of level 1 cache (byte'''
	lazy_execution: int
	'''Default enabled, if disable, use immediately eager execution rather than lazy execution, This flag makes error message and traceback infomation better. But this flag will raise memory consumption and lower the performance. Default: 1'''
	log_file: str
	'''log to file, mpi env will add $OMPI_COMM_WORLD_RANK suffix. Default: ""'''
	log_op_hash: str
	'''Output compiler pass result of certain hash of op. Default: ""'''
	log_silent: int
	'''The log will be completely silent. Default: 0'''
	log_sync: int
	'''Set log printed synchronously. Default: 1'''
	log_v: int
	'''Verbose level of logging. Default: 0'''
	log_vprefix: str
	'''Verbose level of logging prefix. Default: ""'''
	no_fuse: bool
	'''No fusion optimization for all jittor Var creation. Default: 0'''
	no_grad: bool
	'''No grad for all jittor Var creation. Default: 0'''
	node_order: int
	'''id prior. Default: 0'''
	nvcc_flags: str
	'''Flags of CUDA C++ compiler. Default: ""'''
	nvcc_path: str
	'''Path of CUDA C++ compiler. Default: ""'''
	para_opt_level: int
	'''para_opt_level. Default: 3'''
	profile_memory_enable: int
	'''Enable memory profiler. Default: 0'''
	profiler_enable: int
	'''Enable profiler. Default: 0'''
	profiler_hide_relay: int
	'''Profiler hide relayed op. Default: 0'''
	profiler_record_peek: int
	'''Profiler record peek mem bandwidth. Default: 0'''
	profiler_record_shape: int
	'''Profiler record shape for op. Default: 0'''
	profiler_rerun: int
	'''Profiler rerun. Default: 0'''
	profiler_warmup: int
	'''Profiler warmup. Default: 0'''
	python_path: str
	'''Path of python interpreter. Default: ""'''
	reuse_array: int
	'''try reuse np.array memory into jt.array. Default: 0'''
	rewrite_op: int
	'''Rewrite source file of jit operator or not. Default: 1'''
	stat_allocator_total_alloc_byte: int
	'''Total alloc byte. Default: 0'''
	stat_allocator_total_alloc_call: int
	'''Number of alloc function call. Default: 0'''
	stat_allocator_total_free_byte: int
	'''Total alloc byte. Default: 0'''
	stat_allocator_total_free_call: int
	'''Number of alloc function call. Default: 0'''
	th_mode: int
	'''th mode. Default: 0'''
	trace_depth: int
	'''trace depth for GDB. Default: 10'''
	trace_py_var: int
	'''Trace py stack max depth for debug. Default: 0'''
	trace_var_data: int
	'''Trace py stack max depth for debug. Default: 0'''
	try_use_32bit_index: int
	'''If not overflow, try to use 32 bit type as index type. Default: 0'''
	use_acl: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_cuda: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_device: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_nfef_allocator: int
	'''Enable never free exact fit allocator. Default: 0'''
	use_parallel_op_compiler: int
	'''Number of threads that parallel op comiler used, default 16, set this value to 0 will disable parallel op compiler. Default: 16'''
	use_rocm: int
	'''Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda. Default: 0'''
	use_sfrl_allocator: int
	'''Enable sfrl allocator. Default: 1'''
	use_stat_allocator: int
	'''Enable stat allocator. Default: 0'''
	use_temp_allocator: int
	'''Enable temp allocator. Default: 1'''
	use_tensorcore: int
	'''use tensor core. Default: 0'''
flags: Flags
'''Jittor running time flags instance'''
