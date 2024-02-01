# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#   Wenyang Zhou <576825820@qq.com>
#   Guoye Yang <498731903@qq.com>
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
import math
from collections.abc import Sequence,Iterable

def knn(unknown, known, k):
    '''
对未知向量寻找与已知向量里中的欧氏距离最近的 ``k`` 个向量

    参数：
        - unknown (Var): 形状 ``(b, n, c)`` ，其中 ``b`` 是批次数， ``n`` 是每批次未知向量个数， ``c`` 是每个向量维数
        - known (Var): 形状 ``(b, m, c)`` ，其中 ``b`` 和 ``c`` 上同， ``m`` 是每批次已知向量个数
        - k (int): 对于每个未知向量寻找多少个最近向量

    返回值：
        - ``tuple[Var, Var]`` ，其中第一个元素是形状 ``(b, n, k)`` 表示对每个未知向量，在当前批次中找到的最近 ``k`` 个已知向量的欧式距离的平方，第二个元素是形状 ``(b, n, k)`` 表示找到最近向量的下标

    代码示例：
        >>> unknown = jt.randn(1, 4, 3)
        >>> known = jt.randn(1, 10, 3)
        >>> jt.knn(unknown, known, 2)
        (jt.Var([[[0.04125667 1.0006377 ]
                  [1.2794693  2.05327   ]
                  [1.4255599  3.4547305 ]
                  [0.849751   1.5978208 ]]], dtype=float32),
         jt.Var([[[8 4]
                  [9 4]
                  [4 8]
                  [1 3]]], dtype=int32))

    这个函数只在 ``c`` 为 3 的时候正确工作。


'''
    b, n, c = unknown.shape
    _, m, _ = known.shape
    dists2 = jt.empty((b, n, k), dtype="float")
    idx = jt.empty((b, n, k), dtype="int")
    src = '''
__inline_static__
@python.jittor.auto_parallel(2, block_num=256)
void knn_kernel(int b, int batch_index, int n, int index, int m,
                        const float *__restrict__ unknown,
                        const float *__restrict__ known,
                        float *__restrict__ dist2,
                        int *__restrict__ idx) {

#define K %s
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * K;
    idx += batch_index * n * K;
    int j = index;
    {
        float ux = unknown[j * 3 + 0];
        float uy = unknown[j * 3 + 1];
        float uz = unknown[j * 3 + 2];

        float tmp_dist[K];
        int tmp_idx[K];
        #pragma unroll
        for (int i=0; i<K; i++) tmp_dist[i] = 1e30;
        for (int k = 0; k < m; ++k) {
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            
            int first = -1;
            #pragma unroll
            for (int i=0; i<K; i++)
                if (first == -1 && d<tmp_dist[i])
                    first = i;
            if (first == -1) continue;
            #pragma unroll
            for (int i=0; i<K; i++)
                if (K-1-i > first) {
                    tmp_dist[K-1-i] = tmp_dist[K-2-i];
                    tmp_idx[K-1-i] = tmp_idx[K-2-i];
                }
            tmp_dist[first] = d;
            tmp_idx[first] = k;
        }
        #pragma unroll
        for (int i=0; i<K; i++) {
            dist2[j * K + i] = tmp_dist[i];
            idx[j * K + i] = tmp_idx[i];
        }
    }
}
    knn_kernel(in0->shape[0], 0, in0->shape[1], 0, in1->shape[1], in0_p, in1_p, out0_p, out1_p);
    ''' % k
    return jt.code([unknown, known], [dists2, idx],
    cpu_src=src,
    cuda_src=src)

def index_add_(x, dim, index, tensor):
    '''
对于 ``index`` 中的每个下标和 ``tensor`` 里对应的每个张量，在 ``x`` 的 ``dim`` 维对应的下标处加上这个张量

    参数：
        - x (Var): 形状 ``(..., k, ...)``
        - dim (int): 上面 ``x`` 中 ``k`` 对应的是第几维
        - index (Var): 形状 ``(n)`` 选取要加上张量的下标
        - tensor (Var): 形状 ``(n, ..., ...)`` 每个下标处要加上的张量

    代码示例：
        >>> x = jt.ones(2, 3, 4)
        >>> jt.index_add_(x, 0, jt.array([0, 1]), jt.array([jt.ones(3, 4), jt.ones(3, 4) * 2]))
        >>> jt.index_add_(x, 1, jt.array([0, 1]), jt.array([jt.ones(2, 4), jt.ones(2, 4) * 2]))
        >>> x
        jt.Var([[[3. 3. 3. 3.]
                 [3. 3. 3. 3.]
                 [2. 2. 2. 2.]]
                [[5. 5. 5. 5.]
                 [5. 5. 5. 5.]
                 [3. 3. 3. 3.]]], dtype=float32)

'''
    assert len(index.shape) == 1
    assert tensor.shape[0] == index.shape[0]
    x[(slice(None,),)*dim+(index,)] += tensor
jt.Var.index_add_ = index_add_

def __copy__(x):
    return x.copy().detach()
jt.Var.__copy__ = __copy__

def __deepcopy__(x,memo):
    result = x.copy().detach()
    memo[id(x)]=result
    return result
jt.Var.__deepcopy__ = __deepcopy__

def __len__(x):
    return x.shape[0]
jt.Var.__len__ = __len__

def __iter__(x):
    result = []
    for i in range(x.shape[0]):
        result.append(x[i])
    return result.__iter__()
jt.Var.__iter__ = __iter__

def __contains__(x, key):
    return bool((x == key).any())
jt.Var.__contains__ = __contains__

def new(x, *args):
    "\n创建一个和 ``x`` 类型相同的张量\n\n    参数：\n        - x (Var): 一个张量\n        - a (Var): 待转换类型的张量\n       \n       或者\n        \n        - x (Var): 原张量\n        - n1, n2, ... (int): 新张量每维的长度\n\n    返回值：\n        如果给的参数是一个张量，则将其转换为和 ``x`` 类型相同的张量；否则新创建一个给定形状，类型与 ``x`` 相同的未初始化张量\n\n    代码示例：\n        >>> x = jt.array([1.0, 2.0, 3.0])\n        >>> x\n        jt.Var([1. 2. 3.], dtype=float32)\n        >>> x.new(3, 4)\n        jt.Var([[-0.62324923  0.81981313  0.49348482 -0.80826735]\n                [-2.185655   -0.12619412 -0.36160922 -0.28803992]\n                [-1.3812838  -0.3991161   0.9999802  -0.11363947]], dtype=float32)\n        >>> a = jt.array([4, 5])\n        >>> a\n        jt.Var([4 5], dtype=int32)\n        >>> x.new(a)\n        jt.Var([4. 5.], dtype=float32)\n\n"
    if len(args) != 1 or isinstance(args[0], int):
        return jt.empty(args, x.dtype)
    return jt.array(args[0]).cast(x.dtype)
jt.Var.new = new

def __index__(x):
    return int(x.item())
jt.Var.__index__ = __index__

def sort(input, dim=-1, descending=False, stable=False):
    "\n对输入张量 ``input`` 沿着指定维按升序非稳定排序。如果不给定 ``dim`` ，则默认为输入的最后一维。如果指定参数 ``descending`` 为 ``True`` ，则按降序排序。如果指定参数``stable`` 为 ``True`` ，则为稳定排序。\n\n    参数：\n        - input (Var): 要对比的张量\n        - dim (int, optional): 沿着此维排序，默认为最后一维\n        - descending (bool, optional): 布尔值，控制升降排序，默认为升序\n        - stable(bool, optional): 布尔值，控制稳定排序，默认非稳定排序\n\n    代码示例：\n\n    >>> x = jt.random((3, 4))\n    jt.Var([[ 0.25843996 0.6806731 0.48188502 -0.8385996 ]\n    [-0.97904104 -1.0399561 -0.97162217 3.834338 ]\n    [-0.6144879 -0.97539455 -0.81440794 -0.12566419]], dtype=float32)\n    >>> value, index = jt.sort(x)\n    >>> value\n    jt.Var([[-0.8385996 0.25843996 0.48188502 0.6806731 ]\n    [-1.0399561 -0.97904104 -0.97162217 3.834338 ]\n    [-0.97539455 -0.81440794 -0.6144879 -0.12566419]], dtype=float32)\n    >>> index\n    jt.Var([[3 0 2 1]\n    [1 0 2 3]\n    [1 2 0 3]], dtype=int32)\n\n    返回值：\n        返回元组 (value, index) ， ``index`` 为原始输入中的下标。\n   \n \n"
    index, value = jt.argsort(input, dim, descending)
    return value, index
jt.Var.sort = sort

def all(x, dim=()):
    "\n在指定维度上计算逻辑与\n\n    参数：\n        - x (Var): 被计算逻辑与的张量\n        - dim (int | tuple[int, ...]): 计算逻辑与的维度，默认为全部维度，即全部元素逻辑与得到一个元素\n\n    返回值：\n        计算逻辑与的结果，形状为 ``x`` 去掉 ``dim`` 所选维度，元素类型为 ``bool``\n\n    代码示例：\n        >>> x = jt.randn(2, 3) > 0\n        >>> x\n        jt.Var([[ True  True False]\n                [ True False  True]], dtype=bool)\n        >>> jt.all(x)\n        jt.Var([False], dtype=bool)\n        >>> jt.all(x, 0)\n        jt.Var([ True False False], dtype=bool)\n        >>> jt.all(x, 1)\n        jt.Var([False False], dtype=bool)\n\n"
    return x.all_(dim).bool()
jt.Var.all = all

def any(x,dim=()):
    "\n\n在指定维度上计算逻辑或\n\n    参数：\n        - x (Var): 被计算逻辑或的张量\n        - dim (int | tuple[int, ...]): 计算逻辑或的维度，默认为全部维度，即全部元素逻辑或得到一个元素\n\n    返回值：\n        计算逻辑或的结果，形状为 ``x`` 去掉 ``dim`` 所选维度，元素类型为 ``bool``\n\n    代码示例：\n        >>> x = jt.randn(2, 3) > 0\n        >>> x\n        jt.Var([[ True False  True]\n                [False False  True]], dtype=bool)\n        >>> jt.any(x)\n        jt.Var([ True], dtype=bool)\n        >>> jt.any(x, 0)\n        jt.Var([ True False  True], dtype=bool)\n        >>> jt.any(x, 1)\n        jt.Var([ True  True], dtype=bool)\n\n"
    return x.any_(dim).bool()
jt.Var.any = any
    
def bernoulli(input):
    "\n对于每个元素 ``p`` ，返回概率为 ``p`` 的伯努利分布的随机值：\n\n    .. math::\n        \\begin{cases}\n        1 & \\text{ with probability } p \\\\\n        0 & \\text{ with probability } (1 - p)\n        \\end{cases}\n\n    参数：\n        - input (Var): 每个采样所用的概率 ``p``\n\n    返回值：\n        形状与数据类型和 ``input`` 一样的张量，原张量中每个元素 ``p`` 对应返回值里的元素，以概率 ``p`` 为 1，概率 ``1 - p`` 为 0\n\n    代码示例：\n        >>> x = jt.array([0.0, 0.5, 1.0])\n        >>> jt.bernoulli(x)\n        jt.Var([0. 0. 1.], dtype=float32)\n        >>> jt.bernoulli(x)\n        jt.Var([0. 0. 1.], dtype=float32)\n        >>> jt.bernoulli(x)\n        jt.Var([0. 1. 1.], dtype=float32)\n\n"
    return (input>jt.rand_like(input)).cast(input.dtype)

def repeat(x, *shape):
    r'''
    Repeats this var along the specified dimensions.

    Args:

        x (var): jittor var.

        shape (tuple): int or tuple. The number of times to repeat this var along each dimension.
 
    Example:

        >>> x = jt.array([1, 2, 3])

        >>> x.repeat(4, 2)
        [[ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3]]

        >>> x.repeat(4, 2, 1).size()
        [4, 2, 3,]
    '''
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]
    len_x_shape = len(x.shape)
    len_shape = len(shape)
    x_shape = x.shape
    rep_shape = shape
    if len_x_shape < len_shape:
        x_shape = (len_shape - len_x_shape) * [1] + x.shape
        x = x.broadcast(x_shape)
    elif len_x_shape > len_shape:
        rep_shape = (len_x_shape - len_shape) * [1] + list(shape)

    reshape_shape = []
    broadcast_shape = []
    for x_s,r_s in zip(x_shape,rep_shape):
        if r_s != 1:
            reshape_shape.append(1)
            broadcast_shape.append(r_s)
        reshape_shape.append(x_s)
        broadcast_shape.append(1)

    x = x.reshape(reshape_shape)
    x = x.broadcast(broadcast_shape)

    tar_shape = (np.array(x_shape) * np.array(rep_shape)).tolist()

    x = x.reshape(tar_shape)
    return x

jt.Var.repeat = repeat
# tile = jt.Var.tile = repeat
ne = jt.Var.ne = jt.Var.not_equal

def repeat_interleave(x,repeats,dim=None):
    "\n将张量在某个维度上每个元素重复多次\n\n    参数：\n        - x (Var): 被重复的张量，形状 :math:`(n_0, \\ldots, n_{\\mathtt{dim}}, \\ldots, n_k)`\n        - repeats (int): 每个元素重复的次数\n        - dim (int): 在哪个维度上重复，负数表示从后往前数。默认为将向量扁平化后重复\n\n    返回值：\n        ``x`` 在 ``dim`` 维度每个元素重复 ``repeat`` 次得到的向量，形状 :math:`(n_0, \\ldots, n_{\\mathtt{dim}} * \\mathtt{repeats}, \\ldots, n_k)`\n\n    示例代码：\n        >>> x = jt.array([[1, 2], [3, 4]])\n        >>> x.repeat_interleave(2, 0)\n        jt.Var([[1 2]\n                [1 2]\n                [3 4]\n                [3 4]], dtype=int32)\n        >>> x.repeat_interleave(2, 1)\n        jt.Var([[1 1 2 2]\n                [3 3 4 4]], dtype=int32)\n        >>> x.repeat_interleave(2)\n        jt.Var([1 1 2 2 3 3 4 4], dtype=int32)\n\n"
    assert isinstance(repeats,int)
    if dim == None:
        x = x.reshape(-1)
        dim=0
    if dim<0: dim+=x.ndim
    
    tar_shape = list(x.shape)
    x_shape = list(x.shape)
    tar_shape[dim] = tar_shape[dim]*repeats 
    dims = []
    for i in range(len(tar_shape)):
        if dim==i:
            dims.append(f"i{i}/{repeats}")
        else:
            dims.append(f"i{i}")
    return x.reindex(tar_shape,dims)

jt.Var.repeat_interleave = repeat_interleave

def chunk(x, chunks, dim=0):
    "\n将张量在 ``dim`` 维度切分为 ``chunks`` 块\n\n    切块尽量均匀。如果 ``x`` 在 ``dim`` 维的长度不是 ``chunks`` 的整倍数，最后一个块的长度会小于之前块的长度。如果无法切分出 ``chunks`` 个非空的块，则会返回少于 ``chunks`` 个块。\n\n    参数：\n        - x (Var): 被切块的张量，形状 :math:`(n_0, \\ldots, n_{\\mathtt{dim}}, \\ldots, n_k)`\n        - chunks (int): 切成的块数\n        - dim (int): 在哪个维度切块，负数表示从后往前数。默认值：0\n\n    返回值：\n        ``list[Var]`` 为切块后按顺序得到的各个块组成的列表。\n\n    示例代码：\n        >>> x = jt.arange(10)\n        >>> x.chunk(2)\n        [jt.Var([0 1 2 3 4], dtype=int32), jt.Var([5 6 7 8 9], dtype=int32)]\n        >>> x.chunk(4)\n        [jt.Var([0 1 2], dtype=int32),\n         jt.Var([3 4 5], dtype=int32),\n         jt.Var([6 7 8], dtype=int32),\n         jt.Var([9], dtype=int32)]\n        >>> x.chunk(6)\n        [jt.Var([0 1], dtype=int32),\n         jt.Var([2 3], dtype=int32),\n         jt.Var([4 5], dtype=int32),\n         jt.Var([6 7], dtype=int32),\n         jt.Var([8 9], dtype=int32)]\n\n"
    if dim<0:
        dim += x.ndim
    l = x.shape[dim]
    res = []
    if l <= chunks:
        for i in range(l):
            res.append(x[(slice(None,),)*dim+([i,],)])
    else:
        nums = (l-1) // chunks + 1
        for i in range(chunks-1):
            res.append(x[(slice(None,),)*dim+(slice(i*nums,(i+1)*nums),)])
        if (i+1)*nums < l:
            res.append(x[(slice(None,),)*dim+(slice((i+1)*nums,None),)])
    return res
jt.Var.chunk = chunk


def expand(x, *shape):
    '''
扩展并广播一个张量到指定形状

    指定的形状每一维必须为下列值之一

    - ``-1`` 表示不改变这维度的长度
    - 和 ``x`` 对应维长度一样，表示不改变这个维度的长度
    - 新的长度，要求 ``x`` 对应维长度为 1，表示在这个维度上扩展

    参数：
        - x (Var): 被扩展的张量
        - shape (tuple[int, ...]): 扩展到的形状
        
       或者

        - x (Var): 被扩展的张量
        - n0, n1, ... (int): 扩展到的形状

    返回值：
        扩展到指定形状的张量

    示例代码：
        >>> a = jt.zeros((3,1))
        >>> a.expand(3, 4).shape
        [3,4,]
        >>> a.expand(-1, 4).shape
        [3,4,]

'''
    if len(shape) == 1 and isinstance(shape[0], (tuple,list,jt.NanoVector)):
        shape = shape[0]
    shape = list(shape)
    for i in range(len(shape)):
        if shape[i] == -1:
            shape[i] = x.shape[i]
    return x.broadcast(shape)
jt.Var.expand = expand


def t(x):
    "\n转置张量的最后两个维度\n\n    参数：\n        - x (Var): 被转置的张量，形状 :math:`(n_0, n_1, \\ldots, n_{k - 3}, n_{k - 2}, n_{k - 1})`\n\n    返回值：\n        转置后的张量，形状 :math:`(n_0, n_1, \\ldots, n_{k - 3}, n_{k - 1}, n_{k - 2})`\n\n    示例代码：\n        >>> x = jt.array([[1, 2], [3, 4]])\n        >>> x.t()\n        jt.Var([[1 3]\n                [2 4]], dtype=int32)\n\n"
    pose = [i for i in range(x.ndim)]
    pose[-1], pose[-2] = pose[-2], pose[-1]
    return x.transpose(*pose)
jt.Var.t = t 

def median(x,dim=None,keepdim=False, keepdims=False):
    keepdim = keepdim or keepdims
    if dim is None:
        x = x.reshape(-1)
        dim=0
    _,x = jt.argsort(x, dim)
    slices = [slice(None) for i in range(dim-1)]
    k = (x.shape[dim]-1)//2
    if keepdim:
        slices.append(slice(k,k+1))
    else:
        slices.append(k)
    return x[tuple(slices)]

jt.Var.median = median

def stack(x, dim=0):
    "\n将多个形状相同的张量组合成为一个更高维的张量\n\n    参数：\n        - x (list[Var]): 被组合的张量，每个形状相同，都是 :math:`(n_0, \\ldots, n_{\\mathtt{dim}}, \\ldots, n_k)`\n        - dim (int): 第几个维度前插入一个维度。默认值：0\n\n    返回值：\n        组合为的张量，形状 :math:`(n_0, \\ldots, n_{\\mathtt{dim} - 1}, \\mathtt{len(x)}, n_{\\mathtt{dim}}, \\ldots, n_k)`\n\n    示例代码：\n        >>> a = jt.array([1, 2, 3])\n        >>> b = jt.array([4, 5, 6])\n        >>> jt.stack([a, b])\n        jt.Var([[1 2 3]\n                [4 5 6]], dtype=int32)\n        >>> jt.stack([a, b], dim=1)\n        jt.Var([[1 4]\n                [2 5]\n                [3 6]], dtype=int32)\n\n"
    assert isinstance(x, Sequence)
    if len(x) < 2:
        return x[0].unsqueeze(dim)

    res = [x_.unsqueeze(dim) for x_ in x]
    return jt.concat(res, dim=dim)
jt.Var.stack = stack

def flip(x, dim=0):
    "将张量在指定维度镜像翻转\n\n    参数：\n        - x (Var): 需要翻转的张量\n        - dim (int | tuple[int, ...]): 需要翻转的维度，负数表示从后往前数。默认值：0\n\n    返回值：\n        x 在指定维度翻转后的结果\n\n    代码示例：\n        >>> x = jt.array([[1, 2], [3, 4]])\n        >>> x.flip()\n        jt.Var([[3 4]\n                [1 2]], dtype=int32)\n        >>> x.flip(1)\n        jt.Var([[2 1]\n                [4 3]], dtype=int32)\n        >>> x.flip((0, 1))\n        jt.Var([[4 3]\n                [2 1]], dtype=int32)\n\n"
    if isinstance(dim, int):
        dim = [dim]
    for i in range(len(dim)):
        if dim[i]<0:
            dim[i] += x.ndim
        assert dim[i]>=0 and dim[i]<x.ndim
    dim = set(dim)

    tar_dims = []
    for i in range(len(x.shape)):
        if i in dim:
            tar_dims.append(f"xshape{i}-1-i{i}")
        else:
            tar_dims.append(f"i{i}")
    return x.reindex(x.shape, tar_dims)
jt.Var.flip = flip

def cross(input, other, dim=-1):
    " 批量计算向量叉乘：\n\n    :math:`(a_1, a_2, a_3) \\times (b_1, b_2, b_3) = (a_2 b_3 - a_3 b_2 ,  a_3 b_1 - a_1 b_3 ,  a_1 b_2 - a_2 b_1)`\n\n    参数：\n        - input (Var): 叉乘的左参数\n        - other (Var): 叉乘的右参数，形状与 ``input`` 相同\n        - dim (int): 计算叉乘的向量所在的维度，要求 ``input`` 和 ``other`` 在 ``dim`` 维的长度都是 3。负数表示从后往前数。默认值：-1\n\n    返回值：\n        ``input`` 和 ``other`` 对应向量叉乘的结果，形状与 ``input`` 和 ``other`` 相同\n\n    代码示例：\n        >>> x = jt.randn(2, 3)\n        >>> y = jt.randn(2, 3)\n        >>> x, y, x.cross(y)\n        (jt.Var([[ 0.6452728  -1.7699949   0.7562031 ]\n                 [ 0.32050335  0.7093985  -0.44102713]], dtype=float32),\n         jt.Var([[ 0.42947495  0.6956272  -0.35488197]\n                 [ 0.44775873 -0.19148853 -0.52726024]], dtype=float32),\n         jt.Var([[ 0.10210377  0.553766    1.2090378 ]\n                 [-0.45848927 -0.02848507 -0.3790121 ]], dtype=float32))\n        >>> x.cross(x)\n        jt.Var([[0. 0. 0.]\n                [0. 0. 0.]], dtype=float32)\n\n"
    assert input.shape==other.shape, "input shape and other shape must be same"
    if dim < 0: dim += len(input.shape)
    assert input.shape[dim] == 3, "input dim shape must be 3"
    a1 = input[(slice(None,),)*dim+(1,)]*other[(slice(None,),)*dim+(2,)]-input[(slice(None,),)*dim+(2,)]*other[(slice(None,),)*dim+(1,)]
    a2 = input[(slice(None,),)*dim+(2,)]*other[(slice(None,),)*dim+(0,)]-input[(slice(None,),)*dim+(0,)]*other[(slice(None,),)*dim+(2,)]
    a3 = input[(slice(None,),)*dim+(0,)]*other[(slice(None,),)*dim+(1,)]-input[(slice(None,),)*dim+(1,)]*other[(slice(None,),)*dim+(0,)]
    return jt.concat([a1.unsqueeze(dim),a2.unsqueeze(dim),a3.unsqueeze(dim)], dim=dim)
jt.Var.cross = cross

def normalize(input, p=2, dim=1, eps=1e-30):
    " 在指定维度对张量 :math:`L_p` 归一化\n\n    :math:`v' = \\frac{v}{\\max(\\lVert v \\rVert_p, \\epsilon)}`\n\n    参数：\n        - input (Var): 被归一化的张量\n        - p (float): 计算向量范数所用的指数。默认值：2\n        - dim (int | tuple[int, ...]): 被归一化的维度。默认值：1\n        - eps (float): 避免除以零所用的小量 :math:`\\epsilon` 。默认值： ``1e-30``\n\n    返回值：\n        在 ``dim`` 指定维度归一化后的张量\n\n    代码示例：\n        >>> x = jt.randn(2, 3)\n        >>> x, x.normalize()\n        (jt.Var([[ 0.6452728  -1.7699949   0.7562031 ]\n                 [ 0.32050335  0.7093985  -0.44102713]], dtype=float32),\n         jt.Var([[ 0.31786057 -0.8718973   0.3725047 ]\n                 [ 0.35822764  0.792897   -0.49293745]], dtype=float32))\n        >>> x.normalize().norm()\n        jt.Var([1. 1.], dtype=float32)\n\n"
    return input / input.norm(p, dim, True, eps)
jt.Var.normalize = normalize

def unbind(x, dim=0):
    "\n解除一个var的维度。\n\n返回沿着给定维度的所有切片的元组，已经没有它。\n\n参数:\n\n    x (var) – 需要解除维度的var\n\n    dim (int，默认值为0) – 需要移除的维度\n\n例子:\n\n   >>> a = jt.random((2,3))\n   >>> b = jt.unbind(a, 0)\n   >>> c = jt.unbind(a, 1)\n   >>> a,b,c\n   jt.Var([[0.2874082  0.46987164 0.13281713]\n        [0.2576157  0.7470389  0.48285943]], dtype=float32)\n   [jt.Var([0.2874082  0.46987164 0.13281713], dtype=float32), jt.Var([0.2576157  0.7470389  0.48285943],  dtype=float32)]\n   [jt.Var([0.2874082 0.2576157], dtype=float32), jt.Var([0.46987164 0.7470389 ], dtype=float32), jt.Var([0.13281713 0.48285943], dtype=float32)]\n\n\n注意：\n\n    如果dim值给定为负数，那么需要首先对其进行处理，使其等于dim加上x的形状的长度之和。\n\n\n"
    if dim < 0: dim += len(x.shape)
    return [x[(slice(None),)*dim+(i,)] for i in range(x.shape[dim])]

jt.Var.unbind = unbind

def make_grid(x, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    " 将多个图片按网格状拼成一个图片\n\n    参数：\n        - x (Var): 图片数据，形状 :math:`(n, c, h, w)`\n        - nrow (int): 每行多少个图片，最终拼成的网格形状是 :math:`(n / \\mathtt{nrow}, nrow)` 。默认值：8\n        - padding (int): 图片间的间隔。默认值：2\n        - normalize (bool): 是否将图片值线性映射到 :math:`[0, 1]` 。默认值：False\n        - range (None | tuple[float, float]): 映射到 :math:`[0, 1]` 的范围，或者不指定则使用图片数据中的最小最大值。默认值：None\n        - scale_each (bool): 不支持，必须为 False。默认值：False\n        - pad_value (float): 图片间填充的值。默认值：0\n    示例：\n       >>> x = jt.randn(100, 3, 64, 64)\n       >>> grid = jt.make_grid(x)\n       >>> print(grid.shape)\n           [3,860,530,]\n    返回值：\n        形状 :math:`(c, h', w')` 的图片，由 ``x`` 中的图片组合成网格状而成\n\n\n"
    assert isinstance(range, tuple) or range is None
    assert scale_each == False
    if isinstance(x, list): x = jt.stack(x)
    assert isinstance(x, jt.Var)
    if normalize: 
        if range is None: x = (x - x.min()) / (x.max() - x.min())
        else: x = (x - range[0]) / (range[1] - range[0])
    if x.ndim < 4: return x
    if x.ndim == 4 and x.shape[0] <= 1: return x
    nrow = min(nrow, x.shape[0])
    b,c,h,w = x.shape
    ncol = math.ceil(b / nrow)
    return x.reindex([c, h*ncol+(ncol+1)*padding, w*nrow+(nrow+1)*padding], 
                     [f"i1/{padding+h}*{nrow}+i2/{padding+w}", "i0", 
                      f"i1-i1/{padding+h}*{padding+h}-{padding}", f"i2-i2/{padding+w}*{padding+w}-{padding}"], overflow_value=pad_value)

def save_image(
    x,
    filepath,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range = None,
    scale_each = False,
    pad_value = 0,
    format = None
):
    " 将多个图片按网格状拼成一个图片后保存到 ``filepath`` 指定的文件\n\n    参数：\n        - x (Var): 图片数据，形状 :math:`(n, c, h, w)`\n        - filepath (str): 保存目标，文件名或打开的文件对象\n        - nrow (int): 每行多少个图片，最终拼成的网格形状是 :math:`(n / \\mathtt{nrow}, nrow)` 。默认值：8\n        - padding (int): 图片间的间隔。默认值：2\n        - normalize (bool): 是否将图片值线性映射到 :math:`[0, 1]` 。默认值：False\n        - range (None | tuple[float, float]): 映射到 :math:`[0, 1]` 的范围，或者不指定则使用图片数据中的最小最大值。默认值：None\n        - scale_each (bool): 不支持，必须为 False。默认值：False\n        - pad_value (float): 图片间填充的值。默认值：0\n        - format：保存的文件格式，默认由文件名扩展名确定\n     \n    示例代码：\n        >>> x = jt.randn(100, 3, 64, 64)\n        >>> save_image(x, 'random.png')\n\n\n"
    from PIL import Image
    grid = make_grid(x, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ndarr = (grid*255+0.5).clamp(0, 255).permute(1, 2, 0).uint8().numpy()
    im = Image.fromarray(ndarr)
    im.save(filepath, format=format)


def _ntuple(n):
    " 构造一个将值转换为 ``n`` 元组的函数\n\n    参数：\n        - n (int): 要构造的是几元组\n\n    返回值：\n        一个函数，将输入的可迭代对象不变，不可迭代对象重复 ``n`` 次变成 ``n`` 元组\n\n    代码示例：\n        >>> f = jt.misc._ntuple(3)\n        >>> f(1)\n        (1, 1, 1)\n        >>> f((1, 2, 3))\n        (1, 2, 3)\n\n\n"
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple([x]*n)
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def unique(
    input: jt.Var, 
    return_inverse: bool=False, 
    return_counts: bool=False, 
    dim: int=None):

    " 返回张量中去重后的元素\n\n    参数：\n        - input (Var): 被去重的张量\n        - return_inverse (bool): 是否返回 ``input`` 每个元素对应去重结果中的不重复元素下标。默认值：False\n        - return_counts (bool): 是否返回去重结果中每个不重复元素在 ``input`` 里出现次数，必须同时指定 ``return_inverse=True`` 才有效。默认值：False\n        - dim (int | None): 去重的维度，如果为 ``None`` 则将输入扁平化后再去重。默认值：None\n\n    返回值：\n        - 如果 ``return_inverse`` 未指定，则返回 ``Var`` 为去重结果\n        - 否则，如果 ``return_inverse`` 和 ``return_counts`` 指定，则返回 ``tuple[Var, Var, Var]`` 为去重结果、下标、出现次数\n        - 否则，仅 ``return_inverse`` 指定，则返回 ``tuple[Var, Var]`` 为去重结果、下标\n\n    代码示例：\n        >>> jt.unique(jt.array([1, 3, 2, 3]))\n        jt.Var([1 2 3], dtype=int32)\n        >>> jt.unique(jt.array([1, 3, 2, 3, 2]), return_inverse=True, return_counts=True)\n        (jt.Var([1 2 3], dtype=int32), jt.Var([0 2 1 2 1], dtype=int32), jt.Var([1 2 2], dtype=int32))\n        >>> jt.unique(jt.array([[1, 3], [2, 3]]), return_inverse=True)\n        (jt.Var([1 2 3], dtype=int32), jt.Var([[0 2]\n                                               [1 2]], dtype=int32))\n        >>> jt.unique(jt.array([[1, 3], [1, 3]]), dim=0)\n        jt.Var([[1 3]], dtype=int32)\n\n\n"

    temp_shape = None
    if dim == None:
        temp_shape = list(input.shape)
        input_flatten = input.flatten()
        dim = 0
    else:
        input_flatten = input

    input_flatten = input_flatten.transpose(dim, 0)
    orig_shape = input_flatten.shape
    input_flatten = input_flatten.view(orig_shape[0], -1)
    
    with jt.flag_scope(compile_options = {"FLAGS:  --extended-lambda ": 1} if jt.flags.use_cuda else {}):
        indice = jt.code((input_flatten.shape[0], ), 'int32', [input_flatten],
            cpu_header='''
            #include <algorithm>
            ''',
            cpu_src='''
            @alias(input_flatten, in0)
            @alias(indice, out)

            int dimlen = input_flatten_shape0, dimsize = input_flatten_shape1;
            for(int i = 0; i < dimlen; ++i) @indice(i) = i;
            std::sort(&@indice(0), &@indice(dimlen), [&](int a, int b){
                for(int i = 0; i < dimsize; ++i) {
                    int lhs = @input_flatten(a, i), rhs = @input_flatten(b, i);
                    if (lhs != rhs) return lhs < rhs;
                }
                return false;
            });
            ''',
            cuda_header='''
            #undef out
            #include <thrust/extrema.h>
            #include <thrust/device_ptr.h>
            #include <thrust/execution_policy.h>
            #include <thrust/device_vector.h>
            #include <thrust/sequence.h>
    
            #include <cub/cub.cuh> 
            #include <executor.h>
            ''',
            cuda_src=
            '''
                @alias(input_flatten, in0)
                @alias(indice, out)
                int dimlen = indice_shape0, dimsize = input_flatten_shape1;

                if (dimsize == 1) {
                    size_t raw_allocation, d_allocation, temp_storage_bytes = 0;
                    void *d_temp_storage = NULL;
                    int32_t* raw_ptr = (int32_t*)exe.allocator->alloc(dimlen * (sizeof(int32_t) + sizeof(input_flatten_type)), raw_allocation);

                    thrust::device_ptr<int32_t> arange_ptr = thrust::device_pointer_cast(raw_ptr);
                    thrust::sequence(arange_ptr, arange_ptr + dimlen);

                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_flatten_p, 
                                                    (input_flatten_type*)(raw_ptr + dimlen), thrust::raw_pointer_cast(arange_ptr), indice_p, dimlen);
                    d_temp_storage = exe.allocator->alloc(temp_storage_bytes, d_allocation);
                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_flatten_p,
                                                    (input_flatten_type*)(raw_ptr + dimlen), thrust::raw_pointer_cast(arange_ptr), indice_p, dimlen);

                    exe.allocator->free(raw_ptr, dimlen * (sizeof(int) + sizeof(input_flatten_type)), raw_allocation);
                    exe.allocator->free(d_temp_storage, temp_storage_bytes, d_allocation);
                } else {
                    thrust::device_ptr<input_flatten_type> input_ptr = thrust::device_pointer_cast(input_flatten_p);
                    thrust::device_ptr<int32_t> indice_ptr = thrust::device_pointer_cast(indice_p);

                    thrust::sequence(indice_ptr, indice_ptr + dimlen);
                    thrust::sort(thrust::device, indice_ptr, indice_ptr + dimlen,
                        [=] __device__ (int32_t a, int32_t b)->bool {
                            for(int i = 0; i < dimsize; ++i) {
                                input_flatten_type lhs = input_ptr[i + a * dimsize],
                                                rhs = input_ptr[i + b * dimsize];
                                if (lhs != rhs) return lhs < rhs;
                            }
                            return false;
                        });
                }
            '''
        )
    input_sorted = input_flatten[indice][:]
    
    dimlen = indice.shape[0]

    diff = jt.logical_not(jt.all(input_sorted[1:] == input_sorted[: -1], 1))
    diff = jt.concat([jt.Var([False]), diff], 0)
    diff = jt.array(diff, dtype = jt.int32)
  
    with jt.flag_scope(compile_options = {"FLAGS:  --extended-lambda ": 1} if jt.flags.use_cuda else {}):
        output, inverse = jt.code(
            [(-input_sorted.shape[0], ), (indice.shape)],
            [input_sorted.dtype, indice.dtype],
            [input_sorted, diff, indice],
            cpu_header='''
                #include <algorithm>
                @alias(input_sorted, in0)
                @alias(diff, in1)
                @alias(indice, in2)
                @alias(output, out0)
                @alias(inverse, out1)
            ''',
            cpu_src=
            f"bool return_inverse = {int(return_inverse)};" +
            '''
                int tot = -1;
                for (int i = 0; i < input_sorted_shape0; ++i) {
                    if (i == 0 || @diff(i)) {
                        ++tot; @output(tot) = i;
                    }
                    if (return_inverse)
                        @inverse(@indice(i)) = tot;
                }
                output->set_shape({tot + 1});
            ''',
            cuda_header='''
                #undef out

                #include <thrust/extrema.h>
                #include <thrust/device_ptr.h>
                #include <thrust/execution_policy.h>
                #include <thrust/scan.h>
                #include <executor.h>

                @alias(input_sorted, in0)
                @alias(diff, in1)
                @alias(indice, in2)
                @alias(output, out0)
                @alias(inverse, out1)
            ''',
            cuda_src=
            f"bool return_inverse = {int(return_inverse)};" +
            '''
                int dimlen = input_sorted_shape0, dimsize = input_sorted_shape1;
                size_t raw_allocation;
                int32_t* raw_ptr = (int32_t*)exe.allocator->alloc(2 * dimlen * sizeof(int), raw_allocation);

                thrust::device_ptr<int32_t> diff_ptr = thrust::device_pointer_cast(diff_p),
                                            inverse_ptr = thrust::device_pointer_cast(inverse_p),
                                            array_ptr = thrust::device_pointer_cast(raw_ptr),
                                            sum_ptr = thrust::device_pointer_cast(raw_ptr + dimlen),
                                            indice_ptr = thrust::device_pointer_cast(indice_p);
                thrust::device_ptr<input_sorted_type> input_ptr = thrust::device_pointer_cast(input_sorted_p);

                if (return_inverse) {
                    thrust::inclusive_scan(diff_ptr, diff_ptr + dimlen, sum_ptr);
                    thrust::scatter(sum_ptr, sum_ptr + dimlen, indice_ptr, inverse_ptr);
                }

                thrust::sequence(array_ptr, array_ptr + dimlen);
                int32_t num = thrust::unique(array_ptr, array_ptr + dimlen,
                    [=] __device__ (int32_t a, int32_t b)->bool {
                        for(int i = 0; i < dimsize; ++i) {
                            input_sorted_type lhs = input_ptr[i + a * dimsize],
                                            rhs = input_ptr[i + b * dimsize];
                            if (lhs != rhs) return false;
                        }
                        return true;
                    }) - array_ptr;

                cudaMemcpy(output_p, raw_ptr, sizeof(int32_t) * num, cudaMemcpyDeviceToDevice);
                exe.allocator->free(raw_ptr, 2 * dimlen * sizeof(int32_t), raw_allocation);
                output->set_shape({ num });
            '''
        )
    indice_shape = (output.shape[0], )
    output = input_sorted[output][:]

    new_shape = list(orig_shape[1:])
    new_shape.insert(0, -1)
    output = output.view(new_shape).transpose(dim, 0)
    if temp_shape != None:
        inverse = inverse.view(temp_shape).transpose(dim, 0)

    if return_inverse:
        if return_counts:
            counts = jt.zeros(indice_shape, dtype=jt.int32)
            jt.scatter_(counts, 0, inverse.flatten(), jt.ones(dimlen), reduce='add')
            return output, inverse, counts
        else:
            return output, inverse
    else:
        return output

jt.Var.unique = unique


def hypot(a,b):
    " 计算输入平方之和的平方根\n\n    :math:`\\sqrt{a^2 + b^2}`\n\n    参数：\n        - a (Var): 输入参数\n        - b (Var): 输入参数，形状和 ``b`` 一样\n\n    返回值：\n        ``a`` 和 ``b`` 平方之和的平方根\n\n    代码示例：\n        >>> a = jt.array([3.0, 5.0])\n        >>> b = jt.array([4.0, 12.0])\n        >>> jt.hypot(a, b)\n        jt.Var([ 5. 13.], dtype=float32)\n\n"
    return jt.sqrt(a.sqr()+b.sqr())

def rad2deg(x):
    " 将弧度转换为角度\n\n    :math:`x' = \\frac{180 x}{\\pi}`\n\n    参数：\n        - x (Var): 弧度值组成的张量\n\n    返回值：\n        ``x`` 中每个元素换成角度\n\n    代码示例：\n        >>> x = jt.array([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]])\n        >>> x.rad2deg()\n        jt.Var([[ 180.02333 -180.02333]\n                [ 359.98935 -359.98935]\n                [  89.95437  -89.95437]], dtype=float32)\n\n"
    return 180 * x / np.pi

jt.Var.rad2deg = rad2deg

def deg2rad(x):
    "\n将输入中的每一个元素从角度值（以度为单位）转换为弧度值。转换的公式为:\n\n.. math::\n    rad = deg * \\frac{\\pi}{180}\n\n参数:\n    - x (Var) : 输入的角度值，可以是单一浮点数或任意形状的输入 Var\n\n返回:\n    float 或 Var: 转换后的弧度值，其形状与输入相同。\n\n代码示例:\n\n    >>> input = jt.float32([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])\n    >>> jt.deg2rad(a)\n    jt.Var([[ 3.141593  -3.141593 ]\n        [ 6.283186  -6.283186 ]\n        [ 1.5707965 -1.5707965]], dtype=float32)\n\n"
    return x * np.pi / 180.

jt.Var.deg2rad = deg2rad

def arctan2(y,x):
    "\n    按元素计算y/x的反正切值，以弧度计算。注意第一个参数 y 中的元素是 y 坐标，第二个参数 x 中的元素是 x 坐标。\n\n    .. math::\n    \n        \\text{arctan2}(y, x) =\n        \\begin{cases}\n        \\arctan(y / x) - \\pi, & \\text{ if } x \\lt 0, y \\lt 0 \\\\\n        \\arctan(y / x) + \\pi, & \\text{ if } x \\lt 0, y \\geq 0 \\\\\n        \\arctan(y / x), & \\text{ otherwise }\n        \\end{cases}\n    \n    参数：\n    \n        - y (Var): Jittor数组，任何shape和dtype。\n        - x (Var): Jittor数组，与y相同的shape和dtype。+\n    \n    返回：\n    \n        Jittor数组，具有和输入相同的shape和dtype。\n    \n    代码示例：\n\n        >>> x = jt.randn(3)\n        jt.Var([0.8616086  0.81152385 0.0437891], dtype=float32)\n        >>> y = jt.randn(3)\n        jt.Var([-0.0904924   1.4761028  -1.367789], dtype=float32)\n        >>> c = jt.arctan2(y, x)\n        jt.Var([-0.10464364  1.0681262  -1.5387927], dtype=float32)\n\n"
    angle = jt.zeros(x.shape,dtype=x.dtype)
    x = (x!=0.0).ternary(x, 1e-30)
    angle = (y/x).arctan()
    mask = (x<0)&(y<0)
    angle = angle - mask*np.pi
    mask = (x<0)&(y>=0)
    angle = angle + mask*np.pi
    return angle
atan2 = arctan2


def nonzero(x):
    "     返回输入tensor中每一个非零元素的索引。\n\n    参数:\n        x (Var): 任意形状的输入 Var\n\n    代码示例:\n\n    >>> x = jt.array([0, 1, 2, 0, 3, 0])\n    >>> y = jt.array([[0, 1, 2], [0, 3, 4], [0, 0, 1]])\n    >>> jt.nonzero(x)\n    jt.Var([[1]\n            [2]\n            [4]], dtype=int32)\n    >>> jt.nonzero(y)\n    jt.Var([[0 1]\n            [0 2]\n            [1 1]\n            [1 2]\n            [2 2]], dtype=int32)\n\n    返回值:\n        返回一个二维 Var, 每一行代表一个非零元素的索引。注意每一行的长度取决于输入 Var 的维度。\n    \n"
    x = jt.where(x)
    x = [xx.unsqueeze(1) for xx in x]
    if len(x)<2:
        return x[0]
    x = jt.concat(x,dim=1)
    return x

jt.Var.nonzero = nonzero


def arange(start=0, end=None, step=1,dtype=None):
    "    生成一个包含在区间[ ``start`` , ``end`` )内等间隔分布的数字的序列。\n\n    参数:\n        - start(float / Var): 默认值=0, 序列的开始值。也可以是一个仅包含一个元素的 Var\n\n        - end(float / Var): , 默认值=None, 序列的结束值，也可以是一个仅包含一个元素的 Var\n\n        - step(float / Var):, 默认值=1, 生成的数字序列之间的固定间隔，也可以是一个仅包含一个元素的 Var\n\n        - dtype(type): 可选, 把返回的 Var 转换为指定的数据类型。\n\n    代码示例:\n        >>> jt.arange(1, 3.5, 0.5, jt.float32)\n        jt.Var([1.  1.5 2.  2.5 3. ], dtype=float32)\n\n    返回值:\n        一个 1-D 的 Var 对象，包含从  ``start`` (包含)到 ``end`` (不包含)以 ``step`` 为步长的等差序列。\n    \n"
    if isinstance(start, jt.Var): start = start.item()
    if isinstance(end, jt.Var): end = end.item()
    if isinstance(step, jt.Var): step = step.item()
    if end is None:
        end,start = start,0
    l = round((end-start)//step)+1
    if (l-1)*step+start>=end:
        l-=1
    x = jt.index((l,),0)
    x = x*step+start
    if dtype is not None:
        x= x.cast(dtype)
    return x

def log2(x):
    "    计算 ``x`` 中每个元素的以 2 为底的对数值。\n\n    .. math::\n\n       y_i=\\log _2\\left(x_i\\right)\n\n    参数:\n        x(Var): 任意形状的输入 Var\n\n    代码示例:\n        >>> jt.array([1,2,4,8,10]).log2()\n        jt.Var([0. 1. 2.  3.  3.321928], dtype=float32)\n\n    返回值:\n        返回与输入形状相同的 Var, 其元素为输入元素的以2为底的对数。\n    \n"
    return jt.log(x)/math.log(2.0)

jt.Var.log2 = log2

def meshgrid(*tensors):
    "    创建由 ``*tensors`` 这个输入的 Var 列表指定的坐标网格列表。\n\n    给定 :math:`N` 个 1D 张量 :math:`T_0 \\ldots T_{N-1}` 作为输入，输入对应的尺寸分别为 :math:`S_0 \\ldots S_{N-1}`，输出 :math:`N` 个 :math:`N` 维张量 :math:`G_0 \\ldots G_{N-1}`，每个形状为 :math:`\\left(S_0, \\ldots, S_{N-1}\\right)`，其中输出 :math:`G_i` 是通过将 :math:`T_i` 扩展到结果形状而构建的。\n\n    参数:\n        ``*tensors``: (jt.Var, 不限个数)- 输入的张量, 每个张量需要是一维的jt.Var向量。\n    \n    代码示例:\n        >>> a = jt.array([1, 2, 3])\n        >>> b = jt.array([4, 5, 6, 7])\n        >>> jt.meshgrid(a, b)\n        [jt.Var([[1 1 1 1]\n        [2 2 2 2]\n        [3 3 3 3]], dtype=int32), \n         jt.Var([[4 5 6 7]\n        [4 5 6 7]\n        [4 5 6 7]], dtype=int32)]\n\n    返回值:\n        grids(Var): 包含了 N 个 Var 的列表, 由输入张量列表决定。\n    \n"
    if len(tensors)==1 and isinstance(tensors[0], list):
        tensors = tensors[0]
    size = len(tensors)
    shape = []
    for i in range(size):
        assert isinstance(tensors[i],jt.Var) and tensors[i].ndim==1
        shape.append(tensors[i].shape[0])
    grids = []
    view_shape = [1]*size
    for i in range(size):
        vs = view_shape[:]
        vs[i]=-1
        grids.append(tensors[i].reshape(vs).expand(shape))

    return grids


def split(d, split_size, dim=0):
    "    将张量分割成多个块。每个块都是原始张量的一个视图。\n\n    如果 ``split_size`` 是整型，那么张量将被平均分割成等大小的块（如果可能的话）。如果给定维度 ``dim`` 上的张量大小不能被 ``split_size`` 整除，那么最后一个块的大小将会小于其他块。\n\n    如果 ``split_size`` 是列表，那么张量将被分割成 ``len(split_size)`` 块，每个块在 ``dim`` 维度上的大小将按照 ``split_size`` 列表指定。\n\n    参数:\n        - d (Var): 需要被分割的张量 Var, 任意形状。\n\n        - split_size (int 或者 list(int)): 单个块的大小，或者每个块的大小的列表。注意如果这里传入列表的话，应当确保列表的元素之和等于输入张量在给定分割维度上的长度 ``d.shape[dim]``。\n\n        - dim (int): 需要分割的张量的维度。默认值为0。\n    \n    代码示例:\n        >>> a = jt.arange(10).reshape(5, 2))\n        >>> jt.split(a, 2)\n        (jt.Var([[0 1]\n                [2 3]], dtype=int32), \n        jt.Var([[4 5]\n                [6 7]], dtype=int32), \n        jt.Var([[8 9]], dtype=int32))\n        >>> jt.split(a, [3, 1, 1])\n        (jt.Var([[0 1]\n                [2 3]\n                [4 5]], dtype=int32), \n        jt.Var([[6 7]], dtype=int32), \n        jt.Var([[8 9]], dtype=int32))\n\n    返回值:\n        分割后的张量块，以元组的形式返回: Tuple[Var, …]\n    \n"
    if isinstance(split_size,int):
        shape = d.shape[dim]
        if shape % split_size == 0:
            split_size = [split_size]*(shape//split_size)
        else:
            split_size = [split_size]*(shape//split_size)+[shape%split_size]
    if isinstance(split_size, Iterable):
        assert sum(split_size)==d.shape[dim]

    if dim<0:
        dim+=d.ndim
        
    ans = []
    last = 0
    s_last = len(split_size)-1
    gopt_disable = jt.flags.gopt_disable
    for j, i in enumerate(split_size):
        if i==0:
            shape = list(d.shape)
            shape[dim]=0
            new_d = jt.zeros(tuple(shape),dtype=d.dtype)
            ans.append(new_d)
            continue

        ss = (slice(None),)*dim+(slice(last,last+i),)
        if gopt_disable:
            new_d = d.getitem(ss)
        else:
            new_d, d = d.getitem(ss, int(j==s_last))

        last +=i
        ans.append(new_d)
    return tuple(ans)

jt.Var.split = split

def tolist(x):
    "    将输入的 Jittor Var 类型变量转化为 Python 的 list 类型。\n\n    参数:\n        x (Var): 任意形状的输入张量 Var。\n    \n    代码示例:\n        >>> a = jt.array([1,2,3])\n        >>> jt.tolist(a)\n        [1, 2, 3]\n\n    返回值:\n        转化后的 Python 列表类型 (list)\n  \n  "
    return x.numpy().tolist()
jt.Var.tolist = tolist

def view_as(x,y):
    "    以 ``y`` 的形状对 ``x`` 进行 reshape。\n\n    也就是说，对 ``x`` 中的元素进行重排，使得 ``x`` 与 ``y`` 的形状相同。注意，``x`` 中的元素数量必须与 ``y`` 中的元素数量相同，否则报错。\n\n    参数:\n        - x (Var): 需要重塑的输入张量 Var。\n        - y (Var): 作为参照形状的的输入张量 Var。\n    \n    代码示例:\n        >>> x = jt.arange(4)\n        >>> y = jt.array([[1,2],[3,4]])\n        >>> jt.view_as(x, y)\n        jt.Var([[0 1]\n                [2 3]], dtype=int32)\n\n    返回值:\n        重塑后的 Var, 数据来自 ``x``, 形状和 ``y`` 相同。\n  \n  "
    return x.reshape(y.shape)
jt.Var.view_as = view_as

def diag(x,diagonal=0):
    "    从向量构建对角矩阵或从方阵中提取对角元素。\n\n    如果输入是一个一维向量，返回一个二维方阵，其对角线上的元素由输入向量中的元素构成。\n\n    如果输入是一个二维方阵，返回一个由它的对角元素构成的一维向量。\n    \n    对角线的偏移量可以用 ``diagonal`` 确定，``diagonal`` = 0 代表主对角线， ``diagonal`` > 0 为沿右上方移动， ``diagonal`` < 0 为沿左下方移动。\n\n    参数:\n        - x (Var): 输入张量 Var, 向量或矩阵。\n        - diagonal (int): 对角偏移量，默认为 0。\n    \n    代码示例:\n        >>> a = jt.arange(4)\n        >>> jt.diag(a)\n        jt.Var([[0 0 0 0]\n                [0 1 0 0]\n                [0 0 2 0]\n                [0 0 0 3]], dtype=int32)\n        >>> b = jt.randn(2,2)\n        jt.Var([[-0.36892104  1.5115978 ]\n                [ 0.8735136   0.81891364]], dtype=float32)\n        >>> jt.diag(b)\n        jt.Var([-0.36892104, 0.81891364], dtype=float32)\n\n    返回值:\n        向量或方阵 (Var)\n    \n"
    assert x.ndim==1 or (x.ndim==2 and x.shape[0]==x.shape[1])
    d = diagonal if diagonal>=0 else -diagonal
    d_str = f'+{diagonal}' if diagonal>=0 else f'{diagonal}'

    if x.ndim==1:
        output_shape = (x.shape[0]+d,)*2
        return x.reindex(output_shape,[f'i1-{d}' if diagonal>=0 else f'i0-{d}'],overflow_conditions=[f'i0{d_str}!=i1'])
    else:
        output_shape = (x.shape[0]-d,)
        return x.reindex(output_shape,[f'i0+{d}' if diagonal<=0 else 'i0',f'i0+{d}' if diagonal>=0 else 'i0'])

jt.Var.diag = diag
        

def topk(input, k, dim=None, largest=True, sorted=True):
    "    在指定维度上保留最大/最小的 ``k`` 个元素\n\n    该函数实现了在输入 Var ``input`` 的指定维度上找出最大（或最小）的 ``k`` 个元素，并返回这些元素的值以及其在原始 Var 中的索引。返回的 Var 的大小与 ``input`` 一致，除了在指定的维度上大小为 ``k`` \n\n    参数:\n        - input (Var): 输入张量 Var, 任形状\n        - k (int):  指定寻找最大/最小元素个数\n        - dim (int): 指定在哪个维度上寻找最大/最小元素，默认为最后一个维度\n        - largest (bool): 为 ``True`` 时，返回最大的 ``k`` 个元素；为 ``False`` 时，返回最小的 ``k`` 个元素。默认为 ``True``\n        - sorted (bool): 为 ``True`` 时，返回的 ``k`` 个元素按照从大到小的顺序排列；为 ``False`` 时，返回的 ``k`` 个元素顺序是未指定的。默认为 ``True``\n    \n    代码示例:\n        >>> var = jt.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])\n        >>> jt.topk(var, 2, dim=0)\n        (jt.Var([[ 9 10 11 12]\n                [ 5  6  7  8]], dtype=int32), \n        jt.Var([[2 2 2 2]\n                [1 1 1 1]], dtype=int32))\n        >>> jt.topk(var, 2, dim=1)\n        (jt.Var([[ 4  3]\n                [ 8  7]\n                [12 11]], dtype=int32), \n        jt.Var([[3 2]\n                [3 2]\n                [3 2]], dtype=int32))\n\n    返回值:\n        Tuple of Var, 其中包含 ``input`` 在指定维度上仅保留最大/最小 ``k`` 个元素后的张量 (Var)，以及这些元素在 ``input`` 中的索引 (Var)。\n  \n  "
    if input.numel()==0:
        return jt.array([],dtype=input.dtype),jt.array([],dtype='int32')
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim
    
    index,values = jt.argsort(input,dim=dim,descending=largest)
    dims = (slice(None),)*dim+(slice(0,k),)
    indices = index[dims]
    values = values[dims]
    return values,indices

jt.Var.topk = topk

def kthvalue(input, k, dim=None, keepdim=False, keepdims=False):
    "    在指定维度上保留最小的第 ``k`` 个元素\n\n    在输入 Var ``input`` 的指定维度上找出最小的第 ``k`` 个元素，并返回这些元素的值以及其在原始 Var 中的索引。返回的 Var 的大小在指定的维度上大小为 1，其他维度与 ``input`` 一致。\n\n    参数:\n        - input (Var): 输入张量 Var, 任意形状\n        - k (int):  指定寻找第 k 小的元素\n        - dim (int): 指定在哪个维度上寻找第 k 小的元素，默认为最后一个维度\n        - keepdim (bool): 为 ``True`` 时，则在输出中保持缩减的维度。默认为 ``False`` \n        - keepdims (bool): 和 keepdim 的作用相同。旨在与numpy对齐，其在numpy中采用的参数名称是keepdims。默认为 ``False``\n    \n    代码示例:\n        >>> var = jt.arange(1, 7).reshape(2, 3)\n        jt.Var([[1 2 3]\n                [4 5 6]], dtype=int32)\n        >>> jt.kthvalue(var, 2, dim=0, keepdim=True)\n        (jt.Var([[4 5 6]], dtype=int32), jt.Var([[1 1 1]], dtype=int32))\n        >>> jt.kthvalue(var, 2, dim=0, keepdim=False)\n        (jt.Var([4 5 6], dtype=int32), jt.Var([1 1 1], dtype=int32))\n\n    返回值:\n        Tuple of Var, 其中包含 ``input`` 在指定维度上仅保留第 ``k`` 小元素后的张量 (Var)，以及这些元素在 ``input`` 中的索引 (Var)。\n    \n"
    keepdim = keepdim or keepdims
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim
    index,values = jt.argsort(input,dim=dim)
    dims = (slice(None),)*dim+(slice(k-1,k),)
    indices = index[dims]
    values = values[dims]
    if not keepdim and indices.ndim>1:
        indices = indices.squeeze(dim)
        values = values.squeeze(dim)
    return values,indices

jt.Var.kthvalue = kthvalue

def _prod(x,dim=0):
    "    沿着指定维度计算输入张量在这个维度上所有元素的乘积。\n\n    参数:\n        - x (Var): 输入张量 Var, 任意形状\n        - dim (int): 指定要缩减的维度。默认为 0。但未填写dim时会默认计算所有元素乘积。\n    \n    代码示例:\n        >>> a = jt.randn(4, 2)\n        jt.Var([[-0.09882022  1.0750706 ]\n                [ 0.7829371   1.4028096 ]\n                [-0.62049896  1.2268904 ]\n                [ 0.3696484  -1.2867309 ]], dtype=float32)\n        >>> jt.prod(a, 1)\n        jt.Var([-0.10623872  1.0983117  -0.76128423 -0.475638  ], dtype=float32)\n\n    返回值:\n        Var，输出的张量\n    \n"
    x = jt.log(x)
    x = x.sum(dim=dim)
    return jt.exp(x)


def numpy_cumsum(x, dim=None):
    '''    沿着指定维度计算输入张量在这个维度上的累加求和。

    累加求和是指，在指定方向上：

    .. math::
        y_i=x_1+x_2+x_3+\cdots+x_i

    注意，该函数不应该直接调用。推荐使用jittor.misc.cumsum。

    参数:
        - x (Var): 输入张量 Var, 任意形状
        - dim (int): 指定进行累加求和的轴。默认为最后一维。
    
    代码示例:
        >>> a = jt.randn(4, 2)
        jt.Var([[ 0.7793252  -2.1805465 ]
                [-0.46900165  2.0724497 ]
                [-1.2677554   0.31553593]
                [ 0.14260457 -1.4511695 ]], dtype=float32)
        >>> jt.numpy_cumsum(a, 1)
        jt.Var([[ 0.7793252  -1.4012213 ]
                [-0.46900165  1.603448  ]
                [-1.2677554  -0.9522195 ]
                [ 0.14260457 -1.3085649 ]], dtype=float32)
        >>> jt.numpy_cumsum(a, 0)
        jt.Var([[ 0.7793252  -2.1805465 ]
                [ 0.31032354 -0.10809684]
                [-0.95743185  0.2074391 ]
                [-0.81482726 -1.2437304 ]], dtype=float32)

    返回值:
        Var，输出的张量，累加求和后的结果
    
'''
    def cumsum_forward(np, data):
        a = data['inputs'][0]
        b = data['outputs'][0]
        np.cumsum(a, axis=dim, out=b)

    def cumsum_backward(np, data):
        dout = data['dout']
        out = data['outputs'][0]
        np.cumsum(np.flip(dout, dim), axis=dim, out=out)
        np.copyto(out, np.flip(out, dim))
    if (dim == None):
        dim = -1
    assert(dim >= -1 and dim < len(x.shape))
    return jt.numpy_code(x.shape, x.dtype, [x], cumsum_forward, [cumsum_backward])

def cub_cumsum(x, dim=None):
    '''    沿着指定维度计算输入张量在这个维度上的累加求和，使用CUB实现。

    累加求和是指，在指定方向上：

    .. math::
        y_i=x_1+x_2+x_3+\cdots+x_i

    注意，该函数不应该直接调用。推荐使用jittor.misc.cumsum。该函数不支持 CPU 模式，需要先指定在 GPU 上计算。

    参数:
        - x (Var): 输入张量 Var, 任意形状
        - dim (int): 指定进行累加求和的轴。默认为最后一维。
    
    代码示例:
        >>> jt.flags.use_cuda = 1 # 使用 GPU 计算
        >>> a = jt.randn(4, 2)
        jt.Var([[ 0.7793252  -2.1805465 ]
                [-0.46900165  2.0724497 ]
                [-1.2677554   0.31553593]
                [ 0.14260457 -1.4511695 ]], dtype=float32)
        >>> jt.cub_cumsum(a, 1)
        jt.Var([[ 0.7793252  -1.4012213 ]
                [-0.46900165  1.603448  ]
                [-1.2677554  -0.9522195 ]
                [ 0.14260457 -1.3085649 ]], dtype=float32)
        >>> jt.cub_cumsum(a, 0)
        jt.Var([[ 0.7793252  -2.1805465 ]
                [ 0.31032354 -0.10809684]
                [-0.95743185  0.2074391 ]
                [-0.81482726 -1.2437304 ]], dtype=float32)

    返回值:
        Var，输出的张量，累加求和后的结果
    
'''
    if (dim == None):
        dim = -1
    assert(dim >= -1 and dim < len(x.shape))
    shape = list(x.shape)
    if (dim != -1 and dim != len(shape) - 1):
        order = list(range(len(shape)))
        order[dim], order[-1] = order[-1], order[dim]
        shape[dim], shape[-1] = shape[-1], shape[dim]
        x = x.permute(order)
    if (len(shape) > 2):
        x = x.reshape([-1, shape[-1]])
    x = jt.compile_extern.cub_ops.cub_cumsum(x)
    if (len(shape) > 2):
        x = x.reshape(shape)
    if (dim != -1 and dim != len(shape) - 1):
        x = x.permute(order)
    return x

def cumsum(x, dim=None):
    '''    沿着指定维度计算输入张量在这个维度上的累加求和。

    累加求和是指，在指定方向上：

    .. math::
        y_i=x_1+x_2+x_3+\cdots+x_i

    推荐直接调用该函数计算 cumsum。取决于是否指定在 GPU 上计算，该函数会调用不同的算子实现。

    参数:
        - x (Var): 输入张量 Var, 任意形状
        - dim (int): 指定进行累加求和的轴。默认为最后一维。
    
    代码示例:
        >>> a = jt.randn(4, 2)
        jt.Var([[ 0.7793252  -2.1805465 ]
                [-0.46900165  2.0724497 ]
                [-1.2677554   0.31553593]
                [ 0.14260457 -1.4511695 ]], dtype=float32)
        >>> jt.cumsum(a, 1)
        jt.Var([[ 0.7793252  -1.4012213 ]
                [-0.46900165  1.603448  ]
                [-1.2677554  -0.9522195 ]
                [ 0.14260457 -1.3085649 ]], dtype=float32)
        >>> jt.cumsum(a, 0)
        jt.Var([[ 0.7793252  -2.1805465 ]
                [ 0.31032354 -0.10809684]
                [-0.95743185  0.2074391 ]
                [-0.81482726 -1.2437304 ]], dtype=float32)

    返回值:
        Var，输出的张量，累加求和后的结果

   '''
    if (dim == None):
        dim = -1
    assert(dim >= -1 and dim < len(x.shape))
    if jt.flags.use_cuda:
        return cub_cumsum(x, dim)
    else:
        return numpy_cumsum(x, dim)

jt.Var.cumsum = cumsum

def cumprod(x,dim=None):
    "    沿着指定维度计算输入张量在这个维度上的累乘。\n\n    累乘是指，在指定方向上：\n\n    .. math::\n        y_i=x_1 \\times x_2 \\times x_3 \\times \\cdots \\times x_i\n\n    注意，由于此函数的实现是先取 :math:`\\log` 再做累加，因此务必保证输入为正数，否则计算结果会有 ``nan``。\n\n    参数:\n        - x (Var): 输入张量 Var, 任意形状\n        - dim (int): 指定进行乘法的轴。默认为最后一维。\n    \n    代码示例:\n        >>> a = jt.arange(1, 7).reshape(2, 3)\n        jt.Var([[1 2 3]\n                [4 5 6], dtype=int32)\n        >>> jt.cumprod(a, 1)\n        jt.Var([[  1.        2.        6.     ]\n                [  4.       20.      120.00001]], dtype=float32)\n        >>> jt.cumprod(a, 0)\n        jt.Var([[ 1.  2.  3.]\n                [ 4. 10. 18.]], dtype=float32)\n\n    返回值:\n        Var，输出的张量，累乘后的结果，形状和输入一致\n   \n "
    x = jt.log(x)
    x = cumsum(x,dim=dim)
    return jt.exp(x)

jt.Var.cumprod=cumprod

def nms(dets,thresh):
    '''    对给定的一个未排序的矩阵，执行非极大值抑制(non-maximum suppression, NMS)操作。

    输入应当是一个二维矩阵，矩阵的每一行代表一个边界框，格式为： ``[x1, y1, x2, y2, score]`` ，其中， ``[x1, y1]`` 为左上角的坐标， ``[x2, y2]`` 为右下角的坐标， ``score`` 是该边界框对应的置信度分数。

    函数使用的是贪婪策略，首先将所有边界框按照置信度得分降序排序，然后从中选取分数最高的边界框并移除所有与该边界框的 IoU 值大于阈值的边界框。然后对剩余的边界框重复上述过程直到没有边界框，最后返回被选出的边界框的索引。

    IoU 是衡量两个边界框重叠程度的指标，其定义为两个边界框的交集面积除以并集面积。

    参数:
        - dets (Var): 输入的 Var， shape 为 (N, 5), N代表边界框的个数，5列分别代表 ``[x1,y1,x2,y2,score]``
        - thresh (float): IoU 阈值，IoU 值大于此的边界框将被移除
    
    代码示例:
        >>> a = dets = jt.array([[20, 20, 40, 40, 0.9],[22, 22, 42, 42, 0.8],[200, 200, 240, 240, 0.7]], dtype="float32")
        >>> jt.nms(dets, 0.5)
        jt.Var([0 2], dtype=int32)

    返回值:
        被选出的边界框的索引，类型为 Var，在原 `dets` 中的行索引。
  
  
'''
    threshold = str(thresh)
    order = jt.argsort(dets[:,4],descending=True)[0]
    dets = dets[order]
    s_1 = '(@x(j,2)-@x(j,0)+1)*(@x(j,3)-@x(j,1)+1)'
    s_2 = '(@x(i,2)-@x(i,0)+1)*(@x(i,3)-@x(i,1)+1)'
    s_inter_w = 'max((Tx)0,min(@x(j,2),@x(i,2))-max(@x(j,0),@x(i,0))+1)'
    s_inter_h = 'max((Tx)0,min(@x(j,3),@x(i,3))-max(@x(j,1),@x(i,1))+1)'
    s_inter = s_inter_h+'*'+s_inter_w
    iou = s_inter + '/(' + s_1 +'+' + s_2 + '-' + s_inter + ')'
    fail_cond = iou+'>'+threshold
    selected = jt.candidate(dets, fail_cond)
    return order[selected]


jt.Var.expand_as = jt.Var.broadcast_var


def index_fill_(x,dim,indexs,val):
    "    根据给定的索引 ``indexs`` 在指定维度 ``dim`` 上替换为给定元素值 ``val`` \n\n    参数:\n        - x (Var): 输入的 Var， 任意形状\n        - dim (int): 进行索引替换的维度\n        - indexs (Var): 需要替换的索引，类型为 Var，shape 为 (N, )，N 代表需要替换的索引个数\n        - val (float): 替换的元素值\n    \n    代码示例:\n        >>> x = jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=\"float32\")\n        >>> jt.index_fill_(x, 0, jt.array([0, 2]), -1)\n        jt.Var([[-1. -1. -1.]\n                [ 4.  5.  6.]\n                [-1. -1. -1.]], dtype=float32)\n        >>> jt.index_fill_(x, 1, jt.array([0, 2]), -1)\n        jt.Var([[-1.  2. -1.]\n                [-1.  5. -1.]\n                [-1.  8. -1.]], dtype=float32)\n\n    返回值:\n        进行索引替换之后的 Var，shape 与输入的 Var 一致\n\n    "
    overflow_conditions = [f'i{dim}=={i}'for i in indexs]
    indexs = [f'i{i}' for i in range(len(x.shape))]
    return x.reindex(shape = x.shape,indexes = indexs,overflow_conditions=overflow_conditions,overflow_value=val)

def triu_(x,diagonal=0):
    "    返回输入二维矩阵或批量矩阵输入 ``x`` 的上三角部分，其余部分为 0。注意输入矩阵可以不是方阵。\n\n    对角线的偏移量可以用 ``diagonal`` 确定，``diagonal`` = 0 代表主对角线， ``diagonal`` > 0 为沿右上方移动， ``diagonal`` < 0 为沿左下方移动。 \n\n    参数:\n        - x (Var): 输入的 Var， 维度需要大于等于 2\n        - diagonal (int): 对角偏移量，默认为 0。\n    \n    代码示例:\n        >>> x = jt.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=\"float32\")\n        >>> jt.triu_(x, 0)\n        jt.Var([[1. 2. 3.]\n                [0. 5. 6.]\n                [0. 0. 9.]], dtype=float32)\n        >>> jt.triu_(x, 1)\n        jt.Var([[0. 2. 3.]\n                [0. 0. 6.]\n                [0. 0. 0.]], dtype=float32)\n        >>> jt.triu_(x, -1)\n        jt.Var([[1. 2. 3.]\n                [4. 5. 6.]\n                [0. 8. 9.]], dtype=float32)\n\n    返回值:\n        基于输入的上三角矩阵, Var\n    \n"
    l = len(x.shape)
    assert l>1
    overflow_conditions=[f'i{l-1}<i{l-2}+{diagonal}']
    indexs = [f'i{i}' for i in range(l)]
    return x.reindex(x.shape,indexs,overflow_conditions=overflow_conditions,overflow_value=0)

jt.Var.triu_ = triu_

def print_tree(now, max_memory_size, prefix1, prefix2, build_by):
    def format_size(s, end='B'):
        if (s < 1024):
            s = str(s)
            return s + ' '+end

        if (s < 1024*1024):
            s = format(s/1024, '.2f')
            return s + ' K'+end

        if (s < 1024*1024*1024):
            s = format(s/1024/1024, '.2f')
            return s + ' M'+end

        s = format(s/1024/1024/1024, '.2f')
        return s + ' G'+end

    out = ''
    tab = '   '
    out += prefix1+now['name']+'('+now['type']+')\n'
    out += prefix2+'['+format_size(now['size'])+'; '+format(now['size']/max_memory_size*100, '.2f')+'%; cnt:'+format_size(now['cnt'],'') + ']\n'
    if len(now['children']) == 0 and len(now['vinfo']):
        out += prefix2+now['vinfo'][0]
        if len(now['vinfo']) > 1: out += "..."
        out += '\n'
    if (build_by == 0):
        for p in now['path']:
            out += prefix2+p+'\n'
    else:
        out += prefix2+now['path'] + '\n'
    if (len(now['children']) > 0):
        out += prefix2 + tab + '| ' + '\n'
    else:
        out += prefix2 + '\n'
    for i in range(len(now['children'])):
        c = now['children'][i]
        if i < len(now['children']) - 1:
            prefix1_ = prefix2 + tab + '├─'
            prefix2_ = prefix2 + tab + '| '
        else:
            prefix1_ = prefix2 + tab + '└─'
            prefix2_ = prefix2 + tab + '  '
        out += print_tree(c, max_memory_size, prefix1_, prefix2_, build_by)
    return out

def get_max_memory_treemap(build_by=0, do_print=True):
    '''show treemap of max memory consumption

Example::

        net = jt.models.resnet18()
        with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
            imgs = jt.randn((1,3,224,224))
            net(imgs).sync()
            jt.get_max_memory_treemap()

Output::

    | 
    ├─./python/jittor/test/test_memory_profiler.py:100(test_sample)
    | [19.03 MB; 29.67%]
    | ./python/jittor/test/test_memory_profiler.py:100
    |    | 
    |    └─./python/jittor/__init__.py:730(__call__)
    |      [19.03 MB; 29.67%]
    |      ./python/jittor/__init__.py:730
    |         | 
    |         └─./python/jittor/models/resnet.py:152(execute)
    |           [19.03 MB; 29.67%]
    |           ./python/jittor/models/resnet.py:152
    |              | 
    |              ├─./python/jittor/models/resnet.py:142(_forward_impl)
    |              | [6.13 MB; 9.55%]
    |              | ./python/jittor/models/resnet.py:142
    |              |    | 



    '''
    div1 = "[!@#div1!@#]"
    div2 = "[!@#div2!@#]"
    div3 = "[!@#div3!@#]"
    info = jt.get_max_memory_info()

    vars = []
    vars_ = info.split(div1)
    max_memory_size = int(vars_[0])
    vars_ = vars_[1:]
    for v_ in vars_:
        v__ = v_.split(div2)
        vinfo = v__[0].split("{")[0]
        var = {'size':int(v__[1]), 'stack':[], 'cnt':1, "vinfo":vinfo}
        v__ = v__[2:-1]
        for s_ in v__:
            s__ = s_.split(div3)
            s = {'path':s__[0], 'name':s__[1], 'type':s__[2]}
            var['stack'].append(s)
        vars.append(var)
    if (build_by == 0): # build tree by name
        tree = {'name':'root', "children":[], 'size':0, 'cnt':1, 'path':[], 'type':'', 'vinfo':[]}

        def find_child(now, key):
            for c in now['children']:
                if (c['name'] == key):
                    return c
            return None
        for v in vars:
            now = tree
            now['size'] += v['size']
            now['cnt'] += v['cnt']
            now['vinfo'].append(v['vinfo'])
            for s in v['stack']:
                ch = find_child(now, s['name'])
                if (ch is not None):
                    if (not s['path'] in ch['path']):
                        ch['path'].append(s['path'])
                    assert(ch['type']==s['type'])
                    now = ch
                    now['size'] += v['size']
                    now['cnt'] += v['cnt']
                    now['vinfo'].append(v['vinfo'])
                else:
                    now_ = {'name':s['name'], "children":[], 'size':v['size'], 'cnt':v['cnt'], 'path':[s['path']], 'type':s['type'], 'vinfo':[v['vinfo']]}
                    now['children'].append(now_)
                    now = now_
    elif (build_by == 1): # build tree by path
        tree = {'name':'root', "children":[], 'size':0, 'cnt':0, 'path':'_root_', 'type':'', 'vinfo':[]}

        def find_child(now, key):
            for c in now['children']:
                if (c['path'] == key):
                    return c
            return None
        for v in vars:
            now = tree
            now['size'] += v['size']
            now['cnt'] += v['cnt']
            now['vinfo'].append(v['vinfo'])
            for s in v['stack']:
                ch = find_child(now, s['path'])
                if (ch is not None):
                    now = ch
                    now['size'] += v['size']
                    now['cnt'] += v['cnt']
                    now['vinfo'].append(v['vinfo'])
                else:
                    now_ = {'name':s['name'], "children":[], 'size':v['size'],  'cnt':v['cnt'], 'path':s['path'], 'type':s['type'], 'vinfo':[v['vinfo']]}
                    now['children'].append(now_)
                    now = now_
    else:
        assert(False)
        
    def sort_tree(now):
        def takeSize(elem):
            return elem['size']
        now['children'].sort(key=takeSize, reverse=True)
        for c in now['children']:
            sort_tree(c)
    sort_tree(tree)
    out = print_tree(tree, max_memory_size, '', '', build_by)
    if (do_print):
        print(out)
    return tree, out
    
def python_pass_wrapper(mod_func, args, kw):
    "    使用给定的模块函数 ``mod_func`` 、参数 ``args`` 和关键字参数 ``kw`` 通过 Python 的 \n    importlib 导入模块函数并执行它，最后返回执行结果。\n\n    参数:\n        - mod_func (str): 完全限定名称的模块函数，例如 'numpy.matmul' 或 'torch.nn.functional.relu'。\n\n        - args (tuple of str): 需要传递给模块函数的参数，这些参数应以元组的形式提供，元组中的每一项都要是字符串。 例如，如果函数需要两个参数 a 和 b，调用应该为python_pass_wrapper('func', ('a', 'b'), kw)\n\n        - kw (dict): 需要传递给模块函数的关键字参数，应以字典的形式提供。\n    \n    代码示例:\n        >>> jt.python_pass_wrapper('math.sqrt', ('25',), {})\n        5\n        >>> jt.python_pass_wrapper('builtins.len', ('[1, 2, 3]',), {})\n        3\n\n    返回值:\n        Any: 返回模块函数调用的结果。\n    \n"
    import importlib
    mod, func = mod_func.rsplit(".", 1)
    mod = importlib.import_module(mod)
    func = getattr(mod, func)
    args = args + ("**kw",)
    args = ",".join(args)
    return eval(f"func({args})")

def auto_parallel(n, src, block_num=1024, **kw):
    """
    auto parallel(CPU and GPU) n-d for loop function like below:

    Before:

    void inner_func(int n0, int i0, int n1, int i1) {
        ...
    }

    for (int i0=0; i0<n0; i0++)
        for (int i1=0; i1<n1; i1++)
            inner_func(n0, i0, n1, i1, ...);

    After:

    @python.jittor.auto_parallel(2)
    void inner_func(int n0, int i0, int n1, int i1) {
        ...
    }

    inner_func(n0, 0, n1, 0, ...);


    """
    # src = prev_func func_name(args)code
    a, b = src.split('(', 1)
    prev_func, func_name = a.rsplit(None, 1)
    args, code = b.split(')', 1)
    args = args.split(',')
    assert len(args) >= n*2, (args, n)
    oargs = args[n*2:]
    pargs = args[:n*2]
    piargs = pargs[1::2]
    pnargs = pargs[0::2]
    pnargs2 = [ a.split()[-1] for a in pnargs ]
    oargs2 = [ a.split()[-1] for a in oargs ]
    entry_func_args_def = ",".join(["int tn"+str(i) for i in range(n)]
        + pnargs + oargs)
    entry_func_args = ",".join(["tn"+str(i) for i in range(n)]
        + pnargs2 + oargs2)
    tid_def = ""
    tid_loop = ""
    call_args = []
    for i in reversed(range(n)):
        tid_def += f"\nauto tid{i} = tid & ((1<<tn{i})-1);"
        tid_def += f"\nauto tnum{i} = 1<<tn{i};"
        tid_def += f"\ntid = tid>>tn{i};"
    for i in range(n):
        tid_loop += f"\nfor (int i{i}=tid{i}; i{i}<{pnargs2[i]}; i{i}+=tnum{i})"
        call_args.append(pnargs2[i])
        call_args.append(f"i{i}")
    call_args += oargs2
    call_args = ",".join(call_args)
    xn = '\n'
    new_src = f"""
#ifdef JIT_cuda
__device__
#endif
{src.replace(func_name, func_name+"_inner", 1)}

#ifdef JIT_cuda
__global__ static void {func_name}_entry({entry_func_args_def}) {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    {tid_def}
    {tid_loop}
    {func_name}_inner({call_args});
}}
#endif

inline static void {func_name}({",".join(pargs+oargs)}) {{
#ifdef JIT_cuda
    int thread_num = 256*{block_num};
    {xn.join([f"int tn{i} = NanoVector::get_nbits(std::min(thread_num, {pnargs2[i]})) - 2;thread_num >>= tn{i};" for i in reversed(range(n))])}
    thread_num = 1<<({"+".join([f"tn{i}" for i in range(n)])});
    int p1 = std::max(thread_num/{block_num}, 1);
    int p2 = std::min(thread_num, {block_num});
    {func_name}_entry<<<p1,p2>>>({entry_func_args});
#else
    {xn.join([f"for (int i{i}=0; i{i}<{pnargs2[i]}; i{i}++)" for i in range(n)])}
    {func_name}_inner({call_args});
#endif
}}
"""
    return new_src


def numpy_cumprod(a, dim):
    "    沿着指定维度计算输入张量在这个维度上的累乘。\n\n    累乘是指，在指定方向上：\n\n    .. math::\n        y_i=x_1 \\times x_2 \\times x_3 \\times \\cdots \\times x_i\n\n    参数:\n        - a (Var): 输入张量 Var, 任意形状\n        - dim (int): 指定进行乘法的轴。默认为最后一维。\n    \n    代码示例:\n        >>> a = jt.arange(-3, 6).reshape(3, 3)\n        jt.Var([[-3 -2 -1]\n                [ 0  1  2]\n                [ 3  4  5]], dtype=int32)\n        >>> jt.numpy_cumprod(a, 1)\n        jt.Var([[-3  6 -6]\n                [ 0  0  0]\n                [ 3 12 60]], dtype=int32)\n        >>> jt.numpy_cumprod(a, 0)\n        jt.Var([[ -3  -2  -1]\n                [  0  -2  -2]\n                [  0  -8 -10]], dtype=int32)\n\n    返回值:\n        Var，输出的张量，累乘后的结果，形状和输入一致\n    \n"
    class CumprodFunc(jt.Function):
        def forward_code(self, np, data):
            a = data["inputs"][0]
            b = data["outputs"][0]
            out = np.cumprod(a, self.dim)
            np.copyto(b, out)

        def backward_code(self, np, data):
            a, b, dout = data["inputs"]
            out = data["outputs"][0]

            sdim = a.shape[self.dim]
            dim = (len(a.shape)+1)*[1]
            dim[self.dim+1] = sdim
            res = np.tile(np.expand_dims(b, self.dim+1), dim)
            dout = np.tile(np.expand_dims(dout, self.dim+1), dim)

            dim[self.dim]=sdim
            dim[self.dim+1]=1
            a = np.tile(np.expand_dims(a, self.dim), dim)
            res = res/a
            
            mask = np.tril(np.ones((sdim, sdim)))
            for i in range(self.dim):
                mask = np.expand_dims(mask, 0)
            for i in range(len(a.shape)-self.dim-2):
                mask = np.expand_dims(mask, -1)
            res = np.sum(mask*res*dout, self.dim)
            
            np.copyto(out, res)

        def execute(self, a, dim):
            self.save_vars = a
            self.dim = dim
            self.res = jt.numpy_code(
                a.shape,
                a.dtype,
                [a],
                self.forward_code,
            )
            return self.res

        def grad(self, grad_a):
            a = self.save_vars
            b = self.res
            return jt.numpy_code(
                a.shape,
                a.dtype,
                [a, b, grad_a],
                self.backward_code,
            )

    func = CumprodFunc()
    if dim<0:
        dim+=len(a.shape)
    return func(a, dim)

def linspace(start, end, steps):
    "    生成一个由 ``start`` 到 ``end`` 等距的一维张量，其中共有 ``steps`` 个元素，并且包含end。也就是说输出张量为：\n\n    .. math::\n        \\left(\\text { start, } \\text { start }+\\frac{\\text { end }- \\text { start }}{\\text { steps }-1}, \\ldots, \\text { start }+(\\text { steps }-2) * \\frac{\\text { end }- \\text { start }}{\\text { steps }-1}, \\text { end }\\right)\n\n    参数:\n        - start (int, float): 起始值\n        - end (int, float): 结束值\n        - steps (int): 构造的 Var 的长度\n    \n    代码示例:\n        >>> jt.linspace(3, 10, 5)\n        jt.Var([ 3.    4.75  6.5   8.25 10.  ], dtype=float32)\n\n    返回值:\n        Var，一个一维张量，包含在 ``start`` 和 ``end`` 之间的等间隔步数。\n    \n"
    if steps > 1:
        res = jt.index((steps,))[0]
        res = res*float((end-start)/(steps-1))+start
    else:
        res = jt.array([start])
    return res

def randperm(n, dtype="int32"):
    "    生成一个随机排列张量 Var，其中元素为从 0 到 ``n - 1`` 这 ``n`` 个整数。\n\n    参数:\n        - n (int): 要生成的随机排列数组的长度。\n        - dtype (str): 返回的张量中的数据类型，默认为 'int32'\n    \n    代码示例:\n        >>> print(jt.randperm(7))\n        jt.Var([5 3 6 0 4 2 1], dtype=int32)\n\n    返回值:\n        一个从0到n-1的随机排列的 Var 对象。\n   \n "
    key = jt.random((n,))
    index, _ = jt.argsort(key)
    return index.cast(dtype)

def set_global_seed(seed, different_seed_for_mpi=True):
    '''    设置Python，numpy和jittor的随机数生成器的种子。

    这个函数同步地设置了这三个随机数生成器的种子，将它们设置为同一个值，以保证在使用这些生成器时获得一致的随机数。

    Jittor还确保了jittor.dataset.Dataset的每一个工作进程都持有一个不同的种子，同时也保证了使用mpi的每一个进程都持有一个不同的种子。这是通过如下的公式实现的: (global_seed ^ (worker_id*1167)) ^ 1234 + jt.rank * 2591

    参数:
        - seed (int): 指定要设置的随机数种子的值，无默认值
        - different_seed_for_mpi (bool): 是否为每一个使用mpi的进程设置一个不同的种子，默认为True。
    
    返回值:
        无返回值
    
'''
    if (different_seed_for_mpi):
        seed = seed + jt.rank * 2591
    import random
    random.seed(seed)
    jt.set_seed(seed)
    np.random.seed(seed)
    try:
        import cupy
        cupy.random.seed(seed)
    except:
        pass

import time
set_global_seed(int(time.time() * 1000000) % 100000007)

def searchsorted(sorted, values, right=False):
    '''    对有序张量 ``sorted`` 的最后一个维度进行搜索以查找 ``values`` 中每个值应该插入到（插入后仍保持数组有序）的索引位置。

    参数:
        - sorted (Var): 输入的有顺序的数组
        - values (Var): 要进行插入的值
        - right (bool): 默认为 False。如果为 False，则返回的索引 ``i`` 满足 ``sorted[i-1] < value <= sorted[i]``；如果为 True，则返回的索引 ``i`` 满足 ``sorted[i-1] <= value < sorted[i]``。
    
    代码示例:
     >>> sorted = jt.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
     >>> values = jt.array([[3, 6, 9], [3, 6, 9]])
     >>> ret = jt.searchsorted(sorted, values)
     >>> assert (ret == [[1, 3, 4], [1, 2, 4]]).all(), ret
     >>> ret = jt.searchsorted(sorted, values, right=True)
     >>> assert (ret == [[2, 3, 5], [1, 3, 4]]).all(), ret
     >>> sorted_1d = jt.array([1, 3, 5, 7, 9])
     >>> ret = jt.searchsorted(sorted_1d, values)
     >>> assert (ret == [[1, 3, 4], [1, 3, 4]]).all(), ret

    返回值:
        返回一个形状与 ``values`` 相同的 Var ，数组中的每个值表示对应的 ``values`` 在 ``sorted`` 中的索引位置。
    
'''
    _searchsorted_header = f"""
namespace jittor {{

@python.jittor.auto_parallel(2)
inline static void searchsorted(
    int batch_num, int batch_id, int value_num, int value_id,
    int sorted_num, int batch_stride,
    {sorted.dtype}* __restrict__  sort_p, {values.dtype}* __restrict__  value_p, 
    int32* __restrict__ index_p) {{
    int32 l = batch_id * batch_stride;
    int32 r = l + sorted_num;
    auto v = value_p[batch_id * value_num + value_id];
    while (l<r) {{
        int32 m = (l+r)/2;
        if (sort_p[m] {"<=" if right else "<"} v)
            l = m+1;
        else
            r = m;
    }}
    index_p[batch_id * value_num + value_id] = l - batch_id * batch_stride;
}}

}}
"""
    _searchsorted_src = """
    int value_num = in1->shape[in1->shape.size()-1];
    int sorted_num = in0->shape[in0->shape.size()-1];
    int32 batch_num = in0->num / sorted_num;
    int32 batch_num2 = in1->num / value_num;
    int32 batch_stride = batch_num == 1 ? 0 : sorted_num;
    CHECK(batch_num == batch_num2 || batch_num == 1);

    searchsorted(batch_num2, 0, value_num, 0, sorted_num, batch_stride, in0_p, in1_p, out0_p);
"""
    return jt.code(values.shape, "int32", [sorted, values], 
        cpu_header=_searchsorted_header,
        cpu_src=_searchsorted_src,
        cuda_header=_searchsorted_header,
        cuda_src=_searchsorted_src)


def scatter(x:jt.Var, dim:int, index:jt.Var, src:jt.Var, reduce='void'):
    "    根据指定的索引 ``index`` 将 ``src`` 中的元素散布到输入数组 ``x`` 的 ``dim`` 维度，返回修改后的 ``x`` 数组。 \n\n    当 ``x`` 是3维数组时，该函数的效果可表述为:\n\n    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0\n    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1\n    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2\n\n    参数:\n        - x (Var): 输入数组\n        - dim (int): 指定的维度，用于指定沿哪个轴进行索引\n        - index (Var): 用于散布元素的索引，可以为空，或与 ``src`` 的维数相同。当索引为空时，该操作返回的是原始的 ``x`` 数组，即没有做任何修改。\n        - src (Var): 需要散布的源元素\n        - reduce (str): 可选，确定如何执行散布操作。此参数可以是 \"add\" 或 \"multiply\"。\n    \n    代码示例:\n        >>> src = jt.arange(1, 11).reshape((2, 5))\n        >>> index = jt.array([[0, 1, 2, 0]])\n        >>> x = jt.zeros((3, 5), dtype=src.dtype)\n        >>> jt.scatter(x, 0, index, src)\n        jt.Var([[1 0 0 4 0]\n                [0 2 0 0 0]\n                [0 0 3 0 0]], dtype=int32)\n\n    返回值:\n        Var，修改后的 ``x`` 数组，形状与其一致\n    \n"
    shape = index.shape
    if src.shape != shape and src.numel() != 1:
        src = src[tuple( slice(None,s) for s in shape )]
    indexes = [ f'i{i}' for i in range(len(shape)) ]
    indexes[dim] = index
    return x.setitem(tuple(indexes), src, reduce)

def scatter_(x, dim, index, src, reduce='void'):
    return x.assign(x.scatter(dim, index, src, reduce))

jt.Var.scatter = scatter
jt.Var.scatter_ = scatter_

def gather(x, dim, index):
    '''    该函数用给定的索引数组重新索引源数组（如果源数组是3-D数组）。
    
    对于索引数组中的每个元素，假设当前元素的坐标是(i,j,k)，则在结果数组中当前索引对应的元素取决于参数 ``dim`` 的值：
    
    - 如果dim == 0, out[i][j][k] = input[index[i][j][k]][j][k]
    - 如果dim == 1, out[i][j][k] = input[i][index[i][j][k]][k]
    - 如果dim == 2, out[i][j][k] = input[i][j][index[i][j][k]]

    参数:
        - x (Var): 输入数组
        - dim (int): 索引操作进行的轴
        - index (Var): 元素的索引数组
    
    代码示例:
        >>> t = jt.array([[1, 2], [3, 4]])
        >>> jt.gather(t, 1, jt.array([[0, 0], [1, 0]]))
        jt.Var([[1 1]
                [4 3]], dtype=int32)
        >>> jt.gather(t, 0, jt.array([[0, 0], [1, 0]]))
        jt.Var([[1 2]
                [3 2]], dtype=int32)

    返回值:
        Var，通过指定的索引方式重新索引源数组，汇总得到的结果。
    
'''
    shape = index.shape
    indexes = [ f'i{i}' for i in range(len(shape)) ]
    indexes[dim] = index
    return x.getitem(tuple(indexes))

jt.Var.gather = gather

def roll(x, shifts, dims=None):
    '''    将给定的张量在指定维度上滚动，或者说循环移动 ``shifts`` 位。

    参数:
        - x (Var): 源数组
        - shifts (int or tuple): 滚动的偏移量
        - dims (int or tuple, default=None): 滚动的维度，默认为最后一维
    
    代码示例:
        >>> x = jt.array([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        >>> jt.roll(x, 1, 0)
        jt.Var([[7 8]
                [1 2]
                [3 4]
                [5 6]], dtype=int32)
        >>> jt.roll(x, -2, 0)
        jt.Var([[5 6]
                [7 8]
                [1 2]
                [3 4]], dtype=int32)

    返回值:
        Var，循环移位后的张量
    
'''
    if isinstance(shifts, int):
        shifts = (shifts,)
    if dims is None:
        dims = tuple(range(len(shifts)))
    elif isinstance(dims, int):
        dims = (dims,)
    assert len(dims) == len(shifts)
    ids = [ f'i{i}' for i in range(x.ndim) ]
    for i in range(len(dims)):
        shift = shifts[i]
        d = dims[i]
        size = x.shape[d]
        shift = shift % size
        if shift<0: shift += size
        ids[d] = f'(i{d}<{shift}?i{d}+{size-shift}:(i{d}-{shift}))'
    return x.reindex(x.shape, ids)

jt.Var.roll = roll

def safe_log(x):
    "    计算输入矩阵 x 的对数值。\n\n    ..math::\n        y_i=\\log _e\\left(x_i\\right)\n\n    该函数在执行对数运算之前，使用 ``jt.safe_clip`` 将 x 的值限制在一个安全范围内 (1e-30, 1e+30)，以避免因取对数时出现零或极大值造成计算错误。\n\n    参数:\n        - x (Var): 源数组，任意形状\n    \n    代码示例:\n        >>> x = jt.array([0, 0.1, 1.0, 10.0, 100.0, -123])\n        >>> jt.safe_log(x)\n        jt.Var([-69.07755  -2.3025851   0.  2.3025851   4.6051702   -69.07755], dtype=float32)\n\n    返回值:\n        Var，``x`` 中元素的对数值，与 ``x`` 的形状相同。\n    \n"
    return jt.safe_clip(x, 1e-30, 1e30).log()
jt.Var.safe_log = safe_log

class _CTCLossFunction(jt.Function):
    def execute(self, log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False):
        self.blank = blank
        T, N, C = log_probs.shape
        _N, S = targets.shape
        assert _N == N
        log_alpha = jt.full([T,N,S*2+1], -1e30)
        result = jt.empty((N,))
        jt.code([log_probs, targets, input_lengths, target_lengths], [log_alpha, result], cpu_src=f"""
            constexpr int blank = {blank};
            for (int i=0; i<in0_shape1; i++) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                @out0(0,i,0) = @in0(0,i,blank);
                if (target_len)
                    @out0(0,i,1) = @in0(0,i,@in1(i,0));
                for (int j=1; j<input_len; j++)
                    for (int k=0; k<target_len*2+1; k++) {{
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @out0(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @out0(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @out0(j-1,i,k-2);
                        out_type m = std::max(l1, std::max(l2, l3));
                        @out0(j,i,k) = std::log(
                            std::exp(l1-m) +
                            std::exp(l2-m) +
                            std::exp(l3-m)
                        ) + m + @in0(j,i,target);
                    }}
                if (input_len==0)
                    @out1(i) = @out0(0,i,0);
                else {{
                    out_type l1 = @out0(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @out0(input_len-1, i, target_len*2-1);
                    out_type m = std::max(l1, l2);
                    out_type log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
                    @out1(i) = -log_likelihood;
                }}
            }}
        """, cuda_src=f"""
        __global__ void kernel(@ARGS_DEF) {{
            @PRECALC;
            constexpr int blank = {blank};
            for (int i=blockIdx.x; i<in0_shape1; i+=gridDim.x) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                @out0(0,i,0) = @in0(0,i,blank);
                if (target_len)
                    @out0(0,i,1) = @in0(0,i,@in1(i,0));
                for (int j=1; j<input_len; j++)
                    for (int k=threadIdx.x; k-threadIdx.x<target_len*2+1; k+=blockDim.x) {{
                        __syncthreads();
                        if (k>=target_len*2+1)
                            continue;
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @out0(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @out0(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @out0(j-1,i,k-2);
                        out_type m = ::max(l1, ::max(l2, l3));
                        @out0(j,i,k) = ::log(
                            ::exp(l1-m) +
                            ::exp(l2-m) +
                            ::exp(l3-m)
                        ) + m + @in0(j,i,target);
                    }}
                 __syncthreads();
                if (input_len==0)
                    @out1(i) = @out0(0,i,0);
                else {{
                    out_type l1 = @out0(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @out0(input_len-1, i, target_len*2-1);
                    out_type m = ::max(l1, l2);
                    out_type log_likelihood = ::log(::exp(l1-m)+::exp(l2-m))+m;
                    @out1(i) = -log_likelihood;
                }}
            }}
        }}
        kernel<<<std::min(in0_shape1, 1024), std::min(in1_shape1*2+1, 1024)>>>(@ARGS);
        """)
        self.saved_var = [log_probs, targets, input_lengths, target_lengths, log_alpha, result]
        return result

    def grad(self, dout):
        blank = self.blank
        inputs = self.saved_var + [dout]
        dlog_probs = jt.zeros_like(inputs[0])
        dlog_alpha = jt.zeros_like(inputs[4])
        jt.code(inputs, [dlog_probs, dlog_alpha], cpu_src=f"""
            constexpr int blank = {blank};
            for (int i=0; i<in0_shape1; i++) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                if (input_len==0)
                    // write out1 --> read in6
                    // out1(i) = out0(0,i,0);
                    @out1(0,i,0) = @in6(i);
                else {{
                    out_type l1 = @in4(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @in4(input_len-1, i, target_len*2-1);
                    out_type m = std::max(l1, l2);
                    // out_type log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
                    // out1(i) = -log_likelihood;
                    out_type l1_exp = std::exp(l1-m);
                    out_type l2_exp = std::exp(l2-m);
                    out_type sumexp = l1_exp + l2_exp;

                    out_type dlog_likelihood = -@in6(i);
                    out_type dl1 = dlog_likelihood * l1_exp / sumexp;
                    out_type dl2 = dlog_likelihood * l2_exp / sumexp;

                    @out1(input_len-1, i, target_len*2) = dl1;
                    if (target_len)
                        @out1(input_len-1, i, target_len*2-1) = dl2;
                }}
                for (int j=input_len-1; j>0; j--)
                    for (int k=0; k<target_len*2+1; k++) {{
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @in4(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @in4(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @in4(j-1,i,k-2);
                        out_type m = std::max(l1, std::max(l2, l3));
                        out_type l1_exp = std::exp(l1-m);
                        out_type l2_exp = std::exp(l2-m);
                        out_type l3_exp = std::exp(l3-m);
                        out_type sumexp = l1_exp + l2_exp + l3_exp;
                        out_type dalpha = @out1(j,i,k);

                        @out0(j,i,target) += dalpha;

                        @out1(j-1,i,k) += dalpha * l1_exp / sumexp;
                        if (k>0)
                            @out1(j-1,i,k-1) += dalpha * l2_exp / sumexp;
                        if (k>1 && target_2 != target)
                            @out1(j-1,i,k-2) += dalpha * l3_exp / sumexp;
                    }}
                // read in0 -> white out0
                // write out0 ->read out1
                // out0(0,i,0) = in0(0,i,blank);
                @out0(0,i,blank) += @out1(0,i,0);
                if (target_len)
                    @out0(0,i,@in1(i,0)) += @out1(0,i,1);
            }}
        """, cuda_src=f"""
        __global__ void kernel(@ARGS_DEF) {{
            @PRECALC;
            constexpr int blank = {blank};
            for (int i=blockIdx.x; i<in0_shape1; i+=gridDim.x) {{
                int input_len = @in2(i);
                int target_len = @in3(i);
                if (input_len==0)
                    // write out1 --> read in6
                    // out1(i) = out0(0,i,0);
                    @out1(0,i,0) = @in6(i);
                else {{
                    out_type l1 = @in4(input_len-1, i, target_len*2);
                    out_type l2 = -1e30;
                    if (target_len)
                        l2 = @in4(input_len-1, i, target_len*2-1);
                    out_type m = ::max(l1, l2);
                    // out_type log_likelihood = ::log(::exp(l1-m)+::exp(l2-m))+m;
                    // out1(i) = -log_likelihood;
                    out_type l1_exp = ::exp(l1-m);
                    out_type l2_exp = ::exp(l2-m);
                    out_type sumexp = l1_exp + l2_exp;

                    out_type dlog_likelihood = -@in6(i);
                    out_type dl1 = dlog_likelihood * l1_exp / sumexp;
                    out_type dl2 = dlog_likelihood * l2_exp / sumexp;

                    @out1(input_len-1, i, target_len*2) = dl1;
                    if (target_len)
                        @out1(input_len-1, i, target_len*2-1) = dl2;
                }}
                for (int j=input_len-1; j>0; j--)
                    for (int k=threadIdx.x; k-threadIdx.x<target_len*2+1; k+=blockDim.x) {{
                        __syncthreads();
                        if (k>=target_len*2+1)
                            continue;
                        int target = k%2 ? @in1(i,k/2) : blank;
                        int target_2 = target;
                        if (k>1 && k%2) target_2 = @in1(i,k/2-1);
                        out_type l1 = @in4(j-1,i,k);
                        out_type l2 = -1e30;
                        if (k>0) l2 = @in4(j-1,i,k-1);
                        out_type l3 = -1e30;
                        if (k>1 && target_2 != target)
                            l3 = @in4(j-1,i,k-2);
                        out_type m = ::max(l1, ::max(l2, l3));
                        out_type l1_exp = ::exp(l1-m);
                        out_type l2_exp = ::exp(l2-m);
                        out_type l3_exp = ::exp(l3-m);
                        out_type sumexp = l1_exp + l2_exp + l3_exp;
                        out_type dalpha = @out1(j,i,k);

                        atomicAdd(&@out0(j,i,target), dalpha);

                        atomicAdd(&@out1(j-1,i,k), dalpha * l1_exp / sumexp);
                        if (k>0)
                            atomicAdd(&@out1(j-1,i,k-1), dalpha * l2_exp / sumexp);
                        if (k>1 && target_2 != target)
                            atomicAdd(&@out1(j-1,i,k-2), dalpha * l3_exp / sumexp);
                    }}
                // read in0 -> white out0
                // write out0 ->read out1
                // out0(0,i,0) = in0(0,i,blank);
                __syncthreads();
                if (threadIdx.x==0) {{
                    @out0(0,i,blank) += @out1(0,i,0);
                    if (target_len)
                        @out0(0,i,@in1(i,0)) += @out1(0,i,1);
                }}
            }}
        }}
        kernel<<<std::min(in0_shape1, 1024), std::min(in1_shape1*2+1, 1024)>>>(@ARGS);
        """)
        return (dlog_probs,)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    '''    完成连通时间分类 (CTC, Connectionist Temporal Classification) 损失的计算。

    计算连续（未分段）时间序列与目标序列之间的损失。CTCLoss 对输入到目标的可能对齐的概率求和，生成一个损失值，该值对每个输入节点可微分。假定输入到目标的对齐是“多对一”的，这限制了目标序列的长度，使其必须小于等于输入长度。

    参数:
        - log_probs (Var): 形状为[T, N, C]的张量，T为序列长度，N为批次大小，C为类别数量。
        - targets (Var): 形状为[N, S]的张量，N为批次大小，S为目标序列长度，其中的元素应在[0,C)范围内。
        - input_lengths (Var): 形状为[N]的张量，每个元素代表输入序列的长度，元素应在[0,T]范围内。
        - target_lengths (Var): 形状为[N]的张量，每个元素代表目标序列的长度，元素应在[0,S]范围内。
        - blank (int)： 默认为 0，空白标签索引
        - reduction (str)：减少批量损失，如果reduction是none，它会返回一个(N,)的数组，如果reduction是mean或sum，它会返回一个标量。默认为 "mean"。
        - zero_infinity (bool):  默认为 False, 用于梯度计算
    
    代码示例:
        
    .. code-block:: python

        import jittor as jt
        T = 50      # Input sequence length
        C = 20      # Number of classes (including blank)
        N = 16      # Batch size
        S = 30      # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        input = jt.randn(T, N, C).log_softmax(2)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target = jt.randint(low=1, high=C, shape=(N, S), dtype=jt.int)

        input_lengths = jt.full((N,), T, dtype=jt.int)
        target_lengths = jt.randint(low=S_min, high=S+1, shape=(N,), dtype=jt.int)
        loss = jt.ctc_loss(input, target, input_lengths, target_lengths)
        # calculation result, loss is: jt.Var([115.42529], dtype=float32)
        dinput = jt.grad(loss, input)

    返回值:
        Var，包含一个数，是计算出的 CTC 值。
    
'''
    result = _CTCLossFunction.apply(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity)
    if reduction=="mean":
        return result.mean()
    elif reduction=="sum":
        return result.sum()
    assert reduction=="none"
    return result


class CTCLoss(jt.Module):
    '''The Connectionist Temporal Classification loss.


    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf


    Args:

        blank (int, default 0): blank label index
        reduction (string): reduce batch loss,
            if reduction is none, it will return (N,) array,
            if reduction is mean or sum, it will return one scalar
        zero_infinity (bool, default False):
            zero_infinity for grad

    Input:

        log_probs: shape is [T, N, C], T is the sequence length, N is the batch size, C is the class number.
        targets: shape is [N, S], N is the batch size, S is the target sequence length, element should between [0,C).
        input_lengths: shape is [N], which represents the length of input, element should between [0,T].
        target_lengths: shape is N, which represents the length of target, element should between [0,S].

    Example:

        import jittor as jt
        T = 50      # Input sequence length
        C = 20      # Number of classes (including blank)
        N = 16      # Batch size
        S = 30      # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        input = jt.randn(T, N, C).log_softmax(2)
        # Initialize random batch of targets (0 = blank, 1:C = classes)
        target = jt.randint(low=1, high=C, shape=(N, S), dtype=jt.int)

        input_lengths = jt.full((N,), T, dtype=jt.int)
        target_lengths = jt.randint(low=S_min, high=S+1, shape=(N,), dtype=jt.int)
        ctc_loss = jt.CTCLoss()
        loss = ctc_loss(input, target, input_lengths, target_lengths)

        dinput = jt.grad(loss, input)

    '''
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def execute(self, log_probs, targets, input_lengths, target_lengths):
        return ctc_loss(log_probs, targets, input_lengths, target_lengths, self.blank, self.reduction, self.zero_infinity)

def _simple_for(x, func):
    with jt.flag_scope(compile_options={"FLAGS: -O2 ":1}):
        src = f'''
        __inline_static__
        @python.jittor.auto_parallel(1)
        void kernel(int n0, int i0, in0_type* _x, out0_type* y) {{
            using namespace std;
            auto x = _x[i0];
            y[i0] = {func};
        }}
        kernel(in0->num, 0, in0_p, out0_p);
        '''
        return jt.code(x.shape, "bool", [x], cpu_src=src, cuda_src=src)

def isnan(x): 
    "     检查输入 Var 中的元素是否为 NaN。\n\n    参数:\n        x (Var): 输入的张量。\n    \n    代码示例:\n        >>> x = jt.array([1.0, float('nan'), 3])\n        >>> jt.isnan(x)\n        jt.Var([False  True False], dtype=bool)\n\n    返回值:\n        Var, 返回一个与输入张量x同形状的张量，每个元素是判断x对应元素是否为NaN的结果。\n    \n"
    return _simple_for(x, "isnan(float(x))")
jt.Var.isnan = isnan
def isfinite(x): 
    "     检查输入 Var 中的元素是否为有限数值。\n\n    如果 ``x`` 中元素既不是NaN（Not a Number）也不是无穷大或无穷小（Infinity 或 -Infinity）,则返回True，否则返回False。\n\n    参数:\n        x (Var): 输入的张量。\n    \n    代码示例:\n        >>> x = jt.array([1.0, float('Infinity'), float('nan'), 3])\n        >>> jt.isfinite(x)\n        jt.Var([ True False False  True], dtype=bool)\n\n    返回值:\n        Var, 返回一个与输入张量x同形状的张量，每个元素是判断x对应元素是否为有限数值的结果。\n    \n"
    return _simple_for(x, "!isnan(float(x)) && !isinf(float(x))")
jt.Var.isfinite = isfinite
def isinf(x): 
    "     检查输入 Var 中的元素是否为无穷大值。\n\n    如果 ``x`` 中元素是无穷大或无穷小（Infinity 或 -Infinity）,则返回True，否则返回False。\n\n    参数:\n        x (Var): 输入的张量。\n    \n    代码示例:\n        >>> x = jt.array([1.0, float('inf'), float('nan'), float('-inf'), 3])\n        >>> jt.isinf(x)\n        jt.Var([False  True False  True False], dtype=bool)\n\n    返回值:\n        Var, 返回一个与输入张量x同形状的张量，每个元素是判断x对应元素是否为无穷大值的结果。\n    \n"
    return _simple_for(x, "isinf(float(x))")
jt.Var.isinf = isinf
def isneginf(x): 
    "     检查输入 Var 中的元素是否为负无穷。\n\n    如果 ``x`` 中元素是无穷小（-Infinity）,则返回True，否则返回False。\n\n    参数:\n        x (Var): 输入的张量。\n    \n    代码示例:\n        >>> x = jt.array([1.0, float('inf'), float('nan'), float('-inf'), 3])\n        >>> jt.isneginf(x)\n        jt.Var([False False False True False], dtype=bool)\n\n    返回值:\n        Var, 返回一个与输入张量x同形状的张量，每个元素是判断x对应元素是否为负无穷的结果。\n    \n"
    return _simple_for(x, "x<0 && isinf(float(x))")
jt.Var.isneginf = isneginf
def isposinf(x): 
    "     检查输入 Var 中的元素是否为正无穷。\n\n    如果 ``x`` 中元素是无穷大（+Infinity）,则返回True，否则返回False。\n\n    参数:\n        x (Var): 输入的张量。\n    \n    代码示例:\n        >>> x = jt.array([1.0, float('inf'), float('nan'), float('-inf'), 3])\n        >>> jt.isposinf(x)\n        jt.Var([False True False False False], dtype=bool)\n\n    返回值:\n        Var, 返回一个与输入张量x同形状的张量，每个元素是判断x对应元素是否为正无穷的结果。\n    \n"
    return _simple_for(x, "x>0 && isinf(float(x))")
jt.Var.isposinf = isposinf

# fake torch interface
def contiguous(x): 
    "    在内存中创建给定 Var 的深拷贝。\n\n    该函数确保结果的内存是连续的，且与源张量分开。这在您需要更改张量，但不希望这些更改反映在原始张量中时非常有用。\n\n    参数:\n        x (Var): 输入的张量。\n    \n    代码示例:\n        >>> x = jt.array([1,2,3])\n        >>> y = contiguous(x)\n        >>> y[0] = 5\n        >>> print(x) \n        jt.Var([1 2 3], dtype=int32)\n        >>> print(y)\n        jt.Var([5 2 3], dtype=int32)\n\n    返回值:\n        Var, ``x`` 的深拷贝，占用连续的内存\n    \n"
    return x.clone()
jt.Var.contiguous = contiguous
def cpu(x): 
    "    将指定的 Var 复制到CPU。\n\n    参数:\n        x (Var): 输入的张量。\n    \n    代码示例:\n        >>> a = jt.array([1, 2, 3])\n        >>> b = jt.cpu(a)\n        >>> print(b)\n        jt.Var([1 2 3], dtype=int32)\n\n    返回值:\n        Var, 与输入数据有相同值的新数据变量，位于 CPU 上。\n    \n"
    return x.clone()
jt.Var.cpu = cpu
def to(x, *args, **kargs):
    "\n    将 Jittor变量 x转换为指定类型或将其移动到指定计算设备上。\n    这个函数可以对Jittor变量进行类型转换(如，将整数类型转换为浮点数类型)，还可以把数据在CPU和GPU设备之间移动(假设你的系统有GPU设备，且已经安装好了CUDA)。\n\n    参数:\n        - x (jt.Var): 要转移或转换的 Jittor 变量。\n        - \\*args: 可变长度参数列表,第一个参数可以是: \n           - jt.NanoString 或可调用对象：在这种情况下，'x' 将被转换为指定的数据类型。\n           - str: 指定设备的字符串（'cuda' 或 'cpu'）。如果字符串中包含 'cuda'，Jittor 将使用 CUDA（GPU）进行未来的操作。如果字符串中包含 'cpu'，Jittor 将使用 CPU。\n        - \\**kargs: 任意关键字参数。\n\n    代码示例:\n        >>> var = jt.array([1, 2, 3])\n        >>> var_cuda = to(var, 'cuda') # 移动到 GPU\n        >>> var_float = to(var, jt.float32) # 转换为 float32\n\n    返回值:\n        jt.Var: 一个新的 Jittor 变量，已移至指定设备或转换为新的数据类型。如果未指定设备或数据类型，则返回 'x' 的克隆。\n\n"
    if len(args) >= 1:
        s = args[0]
        if isinstance(s, jt.NanoString) or callable(s):
            return x.cast(s)
        s = str(s)
        if "cuda" in s:
            jt.flags.use_cuda = 1
        elif "cpu" in s:
            jt.flags.use_cuda = 0
    return x.clone()
jt.Var.to = to

def rsqrt(x):
    "\n    def rsqrt(x)\n    计算并返回 Jittor 变量的平方根的倒数。\n    此函数对给定的 Jittor 变量执行平方根的倒数运算。它等效于 1 / sqrt(x)。\n\n    参数:\n        x (jt.Var): 输入的 Jittor 变量。\n\n    代码示例:\n        >>> var = jt.array([4, 9, 16])\n        >>> result = rsqrt(var)\n        >>> print(result)\n            [0.5, 0.33333333, 0.25]  # 分别是 2, 3, 4 的平方根的倒数\n\n    返回值:\n        jt.Var: 输入变量的平方根的倒数。返回结果的形状和数据类型与输入参数一致。\n"
    return 1/jt.sqrt(x)
jt.Var.rsqrt = rsqrt

def from_torch(x):
    '''
    将PyTorch Tensor转化为Jittor Var。

    参数:
        x (torch.Tensor): 需要转化的PyTorch Tensor。

    代码示例:
        >>> import torch
        >>> import jittor as jt
        >>> a = torch.tensor([1.0, 2.0, 3.0])
        >>> b = from_torch(a)
        >>> print(b)
        jt.Var([1.0, 2.0, 3.0])

    返回值:
        jt.Var: 转化后的Jittor Variable。


'''
    return jt.Var(x.cpu().numpy())

def triu(input: jt.Var, diagonal:int=0):
    '''
    返回输入的矩阵 (即2-D张量) 或者批量矩阵的上三角部分，其他元素被设置为0。

    参数:
        - input: 输入张量，类型为jt.Var.
        - diagonal: 需要计算的对角线，类型为int，默认为0。
  
    代码示例:
        a = jt.ones(3, 3)
        b = jt.triu(a)
        assert jt.all_equal(b, [[1,1,1],[0,1,1],[0,0,1]])

        b = jt.triu(a, diagonal=1)
        assert jt.all_equal(b, [[0,1,1],[0,0,1],[0,0,0]])
        
        b = jt.triu(a, diagonal=-1)
        assert jt.all_equal(b, [[1,1,1],[1,1,1],[0,1,1]])

    返回值:
        返回上三角的张量，类型为jt.Var.


'''
    index = input.index()
    mask = index[-2] <= index[-1] - diagonal
    return jt.ternary(mask, input, jt.zeros_like(input))
jt.Var.triu = triu
jt.Var.triu_ = lambda x: x.assign(x.triu())

def tril(input: jt.Var, diagonal:int=0):
    '''
    返回一个矩阵（2-D张量）或批量矩阵输入的下三角部分，结果张量out的其它元素被设为0. 

    参数:
        - input: 输入张量，类型为jt.Var。
        - diagonal: 考虑的对角线，类型为int，默认值为0。

    代码示例:
        a = jt.ones(3, 3)
        b = jt.tril(a)
        assert jt.all_equal(b, [[1,0,0],[1,1,0],[1,1,1]])
        
        b = jt.tril(a, diagonal=1)
        assert jt.all_equal(b, [[1,1,0],[1,1,1],[1,1,1]])
        
        b = jt.tril(a, diagonal=-1)
        assert jt.all_equal(b, [[0,0,0],[1,0,0],[1,1,0]])

    返回值:
        下三角矩阵，类型为jt.Var。

'''
    index = input.index()
    mask = index[-2] >= index[-1] - diagonal
    return jt.ternary(mask, input, jt.zeros_like(input))
jt.Var.tril = tril
jt.Var.tril_ = lambda x: x.assign(x.tril())

def all_equal(a: jt.Var, b: jt.Var):
    "\n    检查两个Jittor变量（jt.Var）是否完全相等。如果每一个元素都一致，函数则返回True，否则返回False。\n    这个函数通过逐元素的比较来判断两个变量是否相等。首先，使用`a == b`操作进行元素级的比较。然后，通过`.all()`操作来验证所有元素是否与`b`一致。最后，`.item()`操作将结果转化为Python的bool值。\n    传入变量应该是Jittor的Var类型，a和b的尺寸应该是一样的。如果不一样，函数将抛出ValueError`。\n\n\n    参数：\n        - a : jt.Var  第一个Jittor变量，用于比较的输入。\n        - b : jt.Var 第二个Jittor变量，用于比较的输入。\n\n    代码示例：\n    a = jt.array([1, 2, 3])\n    b = jt.array([1, 2, 3])\n    print(all_equal(a, b))  # 输出：True\n\n    a = jt.array([1, 2, 3])\n    b = jt.array([1, 2, 4])\n    print(all_equal(a, b))  # 输出：False\n\n    返回值：\n        bool类型，如果a和b所有元素相等，返回True，否则返回False。\n\n"
    return (a == b).all().item()
jt.all_equal = all_equal

def _to_float(x: jt.Var):
    "\n    将指定的jt.Var变量转化为'float32'类型。\n\n    参数:\n        - x: 需要转换的jt.Var变量。\n    示例代码：\n          >>> x = jt.Var([1])\n          >>> y = x._to_float()\n          >>> x,y\n          jt.Var([1], dtype=int32) jt.Var([1.], dtype=float32)\n   \n 返回值:\n        转换类型后的jt.Var变量。\n\n\n"
    if x.dtype != "float64": x = x.float()
    return x
jt.Var._to_float = _to_float

def index_select(x: jt.Var, dim:int, index: jt.Var):
    '''
    该方法返回一个新的张量，该张量通过使用索引在dim维度上对输入张量x进行索引。
    返回的张量具有与原始张量（x）相同的维度数。dim维度的大小与索引的长度相同; 其他维度的大小与原始张量中的大小相同。

    参数:
        - x: jt.Var类型，输入的张量。
        - dim: int类型，需要进行索引的维度。
        - index: jt.Var类型，包含要索引的索引的1-D张量。

    返回值: 
        jt.Var类型，新的通过索引获得的张量。

    代码示例::
        >>> x = jt.randn(3, 4)
        >>> indices = torch.tensor([2, 1])
        >>> y = jt.index_select(x, 0, indices)
        >>> assert jt.all_equal(y, x[indices])
        >>> y = jt.index_select(x, 1, indices)
        >>> assert jt.all_equal(y, x[:, indices])

    返回值: 
        jt.Var类型，新的通过索引获得的张量。

    '''
    return x.getitem(((slice(None),)*dim)+(index,))
jt.index_select = index_select

def multinomial(weights: jt.Var, num_samples: int, replacement: bool=False):
    '''
    返回一个var，其中每一行包含从对应输入权重行的多项分布中抽取的num_samples个索引。

    参数:
        - weights: 输入概率，类型为jt.Var。（如：weights = jt.float32([0, 10, 3, 0])）
        - num_samples: 抽取的样本数量，类型为整数（如：num_samples = 4）。这个参数必须小于等于weights的个数。换句话说如果你在不放回的情况下设置num_samples为比weights个数还多，将会报错。
        - replacement: 是否放回地抽样，类型为布尔值，默认为False。如果设置为True，那么每一次抽样权重不变，可以多次抽到同一个。反之，每抽一次，权重会进行相应改变，不会抽到同一个。

    代码示例:
        weights = jt.float32([0, 10, 3, 0])
        x = jt.multinomial(weights, 2)
        assert jt.all_equal(x, [1, 2]) or jt.all_equal(x, [2, 1])
        x = jt.multinomial(weights, 4, replacement=True)
        assert x.shape == (4, )

        weights = jt.float32([[0,0,2],[0,1,0], [0.5,0,0]])
        x = jt.multinomial(weights, 1)
        assert jt.all_equal(x, [[2],[1],[0]])

    返回值: 
        返回一个var，其中每一行包含从对应输入权重行的多项分布中抽取的num_samples个索引。

'''
    if replacement:
        cum_probs = jt.cumsum(weights)[..., None, :]
        cum_probs_l = cum_probs[..., :-1]
        cum_probs_r = cum_probs[..., 1:]
        shape = weights.shape[:-1] + (num_samples, 1)
        rand = jt.rand(shape) * cum_probs[..., :1, -1:]
        one_hot = jt.logical_and(cum_probs_l < rand, rand <= cum_probs_r)
        index = one_hot.index(one_hot.ndim - 1) + 1
        return (one_hot * index).sum(-1)
    else:
        # A-Res algorithm
        # Pavlos S. Efraimidis and Paul G. Spirakis, 2006, Weighted random sampling with a reservoir
        assert num_samples <= weights.shape[-1], "num_samples larger than the input"
        # prevent rand generate 1, 1^inf = 1, with override other result
        a = jt.rand(weights.shape).minimum(0.999999)
        rand = a ** (1/weights)
        _, indices = jt.topk(rand, num_samples)
        return indices

def histc(input, bins, min=0., max=0.):
    ''' Return the histogram of the input N-d array.

    :param input: the input array.
    :param bins: number of bins.
    :param min: min of the range.
    :param max: max of the range.

    Example::

        inputs = jt.randn((40,40))
        joup = jt.histc(x, bins=10)
        
    '''
    if min == 0 and max == 0:
        min, max = input.min(), input.max()
    assert min < max
    bin_length = (max - min) / bins
    histc = jt.floor((input[jt.logical_and(input >= min, input <= max)] - min) / bin_length).int().reshape(-1)
    hist = jt.ones_like(histc).float().reindex_reduce("add", [bins,], ["@e0(i0)"], extras=[histc])
    if hist.sum() != histc.shape[0]:
        hist[-1] += 1
    return hist

def peek_s(x):
    if isinstance(x, jt.Var):
        return x.peek()
    if isinstance(x, (list, tuple)):
        res = "["
        for a in x:
            res += peek_s(a)
            res += ", "
        res += "]"
        return res
    if isinstance(x, dict):
        res = "{"
        for a in x:
            res += a
            res += ":"
            res += peek_s(x[a])
            res += ", "
        res += "}"
        return res
    if isinstance(x, str):
        return x
    return x.__class__.__name__

def peek(x):
    print(peek_s(x))

class Finfo:
    pass
bfloat16_finfo = Finfo()
bfloat16_finfo.min = -1e38
bfloat16_finfo.max = 1e38

def finfo(dtype):
    "\n根据指定的数据类型，返回其numpy浮点数类型的机器限制信息。numpy为每个浮点数提供了一个finfo对象，该对象提供了关于机器限制的信息，例如浮点数类型可以表示的最大和最小的正值、精度和“数字”的数量等，参见 ``https://numpy.org/doc/stable/reference/generated/numpy.finfo.html``\n\n    参数：\n        - dtype (str) : 指定的数据类型。例如：'float16'。\n\n    返回值：\n        返回对应指定数据类型的 numpy finfo 对象或者 'bfloat16_finfo'对象。\n\n    代码示例：\n        >>> from jittor import misc\n        >>> misc.finfo('float32')\n        finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)\n\n"
    if dtype == "bfloat16":
        return bfloat16_finfo
    return np.finfo(str(dtype).split('.')[-1])

def iinfo(dtype):
    "\n    此函数返回给定数据类型可表示的最大/最小整数的信息。此函数仅支持整数类型，如果dtype不是整数类型，会引发ValueError。    \n\n    参数:\n        - dtype (str)：期望获得其信息的数据类型的名称。如'int32'、'int16'等。\n\n    返回值:\n        返回一个iinfo对象，它有以下属性：min, max, dtype等。\n    \n    代码示例：\n        >>> from jittor import misc\n        >>> misc.iinfo('int32')\n        iinfo(min=-2147483648, max=2147483647, dtype=int32)\n\n"
    return np.iinfo(str(dtype).split('.')[-1])


def index_select(input,dim,indices):
    return input[(None,)*dim+(indices,)]

jt.Var.index_select = index_select

def cuda(x):
    "\n这是一个用于将Jittor的全局标记`use_cuda`设置为1，并返回输入值的函数。当`use_cuda`被设置为1时，这意味着Jittor将使用CUDA进行加速。\n\n    参数\n        - x : 输入值, 类型不限。\n\n    返回值：\n        x : 传入该函数的输入值。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> jt.flags.use_cuda\n        0\n        >>> jt.cuda(5)\n        5\n        >>> jt.flags.use_cuda\n        1\n\n"
    jt.flags.use_cuda = 1
    return x
jt.Var.cuda = cuda
jt.Var.npu = cuda
