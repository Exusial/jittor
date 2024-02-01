# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#     Zheng-Ning Liu <lzhengning@gmail.com>
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from abc import abstractmethod
import jittor as jt
from jittor import flatten, init, Module
import numpy as np
import collections
import math
from collections import OrderedDict
from jittor.pool import *
from jittor.optim import *
from jittor.misc import _pair, _triple
from jittor_utils import LOG


def matmul_transpose(a, b):
    "\n对于给定的两个矩阵 ``a`` 和 ``b`` ，这个函数首先将 ``b`` 转置，然后再对 ``a`` 和转置后的 ``b`` 进行矩阵乘法运算，返回该运算结果 :math:`ab^T`\n\n    参数:\n        - a (jittor_core.NdArray) : 输入的第一个矩阵\n        - b (jittor_core.NdArray) : 输入的第二个矩阵\n\n    返回值:\n        矩阵 ``a`` 与转置后的矩阵 ``b`` 的矩阵乘法运算结果。如果``a``的最后一个维度与``b``的最后一个维度不同，将引发 ``AssertionError`` ；如果 ``a`` 的形状与 ``b`` 的形状不同，这个函数会使用Jittor的广播规则来进行矩阵乘法运算。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> nn.matmul_transpose(jt.ones(3,4),jt.ones(3,4))\n        jt.Var([[4. 4. 4.]\n                [4. 4. 4.]\n                [4. 4. 4.]], dtype=float32)\n\n\n"
    assert a.shape[-1] == b.shape[-1], (a.shape, b.shape)
    if len(a.shape) != 2:
        aa = a.reshape((-1, a.shape[-1]))
        cc = matmul_transpose(aa, b)
        return cc.reshape(a.shape[:-1]+(-1,))
    assert len(a.shape) == 2 and len(b.shape) == 2

    shape = list(a.shape)[:-1] + list(b.shape)
    with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
        a = a.broadcast(shape, [len(shape)-2])
        b = b.broadcast(shape)
        return (a*b).sum(len(shape)-1)


def bmm_transpose(a, b):
    '''
    returns a * b^T
    '''
    if jt.flags.use_cuda and jt.compile_extern.cublas_ops:
        return jt.compile_extern.cublas_ops.cublas_batched_matmul(a, b, 0, 1)
    t = list(range(b.ndim))
    t[-1], t[-2] = t[-2], t[-1]
    return bmm(a, b.transpose(t))


def bmm(a, b):
    ''' batch matrix multiply, 
shape of input a is [batch, n, m],
shape of input b is [batch, m, k],
return shape is [batch, n, k]

Example::

    import jittor as jt
    from jittor import nn

    batch, n, m, k = 100, 5, 6, 7

    a = jt.random((batch, n, m))
    b = jt.random((batch, m, k))
    c = nn.bmm(a, b)
    '''
    assert len(a.shape) > 2 and len(b.shape) > 2
    return matmul(a, b)

def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    res = bmm(batch1, batch2)
    if alpha != 1: res = res * alpha
    if beta == 0: return res
    return beta * input + res

def matmul(a, b):
    "\n矩阵乘法。此函数接收两个参数，执行矩阵乘法操作，并且返回结果。输入矩阵a，b的尺寸必须匹配。具体来说，a的最后一维的大小必须和b的倒数第二维的大小相同。\n\n    参数：\n        - a : Var，形状为 (..., M, N)的第一个输入矩阵.\n        - b : Var, 形状为 (..., N, K)的第二个输入矩阵.\n\n    返回值：\n        结果矩阵, 形状为 (..., M, K)\n\n    代码示例：\n        >>> a = jt.random([3])\n        >>> b = jt.random([3])\n        >>> c = jt.matmul(a, b)\n        >>> c.shape\n        [1]\n        >>> a = jt.random([3, 4])\n        >>> b = jt.random([4])\n        >>> c = jt.matmul(a, b)\n        >>> c.shape \n        [3]\n        >>> a = jt.random([10, 3, 4])\n        >>> b = jt.random([4])\n        >>> c = jt.matmul(a, b)\n        >>> c.shape \n        [10, 3]\n        >>> a = jt.random([10, 3, 4])\n        >>> b = jt.random([4, 5])\n        >>> c = jt.matmul(a, b)\n        >>> c.shape \n        [10, 3, 5]\n        >>> a = jt.random([10, 3, 4])\n        >>> b = jt.random([10, 4, 5])\n        >>> c = jt.matmul(a, b)\n        >>> c.shape \n        [10, 3, 5]\n        >>> a = jt.random([8, 1, 3, 4])\n        >>> b = jt.random([10, 4, 5])\n        >>> c = jt.matmul(a, b)\n        >>> c.shape\n        [8, 10, 3, 5]\n    \n\n"
    with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
        len_a = len(a.shape)
        len_b = len(b.shape)
        if len_b == 1:
            # a: [n, m], b:[m], c:[n]
            return (a*b).sum(-1)
        if len_a == 1:
            # a: [n], b:[n,k], c:[k]
            return (a.broadcast(b, [-1]) * b).sum(0)
        if len_a>=3 and len_a==len_b:
            # bmm
            # a: [..., n, m], b: [..., m, k], c:[..., n, k]
            if jt.flags.use_cuda and jt.compile_extern.cublas_ops:
                return jt.compile_extern.cublas_ops.cublas_batched_matmul(a, b, 0, 0)
        shape = []
        len_c = max(len_a, len_b)
        (n, m), (m_, k) = a.shape[-2:], b.shape[-2:]
        assert m == m_, f"dimension not match, a.shape:{a.shape}, b.shape:{b.shape}"
        # a: [..., n, m]
        # b: [..., m, k]
        # cc:[..., n, m, k]
        #     -->
        #     012
        if len_b == 2 and len_a>2:
            # TODO:ugly implementation for tuner
            aa = a.reshape((-1, m))
            cc = matmul(aa, b)
            # print(a.shape, b.shape, cc.shape) 
            return cc.reshape(a.shape[:-1] + [k])
        for i in range(len_c-2):
            ai = len_a-(len_c-i)
            bi = len_b-(len_c-i)
            an = a.shape[ai] if ai>=0 else 1
            bn = b.shape[bi] if bi>=0 else 1
            if an!=1 and bn!=1:
                assert an == bn, f"dimension not match, a.shape:{a.shape}, b.shape:{b.shape}"
            cn = max(an, bn)
            shape.append(cn)
        shape.extend([n, m, k])
        a = a.broadcast(shape, [-1])
        b = b.broadcast(shape, [-3])
        return (a*b).sum(-2)
jt.Var.matmul = jt.Var.__matmul__ = matmul
jt.Var.__imatmul__ = lambda a,b: a.assign(matmul(a,b))

def get_init_var_rand(shape, dtype):
    return jt.array(np.random.normal(0.0, 1.0, shape).astype(np.float32))

def relu(x): 
    " \n该函数为Jittor的ReLU激活函数（修正线性单元），在神经网络中应用广泛。ReLU函数在输入值 x > 0 时，返回 x；当输入值 x <= 0 时，返回 0，即 :math:`\\text{ReLU}(x) = \\max(0,x)`\n\n    参数：\n        - x (Var) : 输入的Var张量\n\n    返回值：\n        输入的张量x应用ReLU激活的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0.5, -0.5, -0.7])\n        >>> nn.relu(x)\n        jt.Var([0.5 0.  0. ], dtype=float32)\n\n"
    cond = x>0.0
    return jt.ternary_out_hint(cond, x, 0.0)


def leaky_relu(x, scale=0.01): 
    "\n该函数为Jittor的Leaky ReLU激活函数，与ReLU函数不同的是，输入值 x < 0 时，不直接返回 0，而是返回输入值进行scale之后的结果：\n\n    .. math::\n        \\text{LeakyRELU}(x) =\n        \\begin{cases}\n        x, &  x \\geq 0 \\\\\\\\\n        \\text{scale} * x, & x < 0\n        \\end{cases}\n\n    参数：\n        - x(Var): 输入的Var张量\n        - scale(float，可选)：x<0情况下的放缩比例。默认值：0.01\n\n    返回值：\n        输入的张量x应用Leaky ReLU激活的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0.5, -0.5, -0.7])\n        >>> nn.leaky_relu(x)\n        jt.Var([ 0.5   -0.005 -0.007], dtype=float32)\n\n"
    return jt.ternary(x>0, x, x*scale)

def relu6(x): 
    "\n该函数为Jittor的ReLU6激活函数，这是一个元素级别（element-wise）函数。与ReLU函数不同的是，输入值 x >= 6 时，不直接返回 x，而是返回6:\n\n    .. math::\n        \\text{ReLU6}(x) = \\min(\\max(0,x), 6)\n\n    参数：\n        - x(Var): 输入的Var张量\n\n    返回值：\n        Var张量。张量x应用ReLU6激活的结果，每个元素的值在0和6之间\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0.5, 6.5, -0.7])\n        >>> nn.relu6(x)\n        jt.Var([0.5 6.  0. ], dtype=float32)\n\n\n"
    return jt.minimum(jt.maximum(x, 0.0), 6.0)

def elu(x: jt.Var, alpha: float = 1.0) -> jt.Var:
    "\n该函数为Jittor的ELU激活函数，这是一个元素级别（element-wise）函数。与ReLU函数不同的是，输入值 x <= 0 时，不直接返回 0，而是返回关于x的自然指数的一个线性函数:\n\n    .. math::\n        \\text{ELU}(x) = \\begin{cases}\n        x, & \\text{当} x > 0\\\\\\\\\n        \\alpha * (\\exp(x) - 1), & \\text{当} x \\leq 0\n        \\end{cases}\n\n    参数：\n        - x(Var): 输入的Var张量\n        - alpha(float，可选)：x<=0时公式中的 :math:`\\alpha` 值。默认值：1.0\n\n    返回值：\n        Var张量。张量x应用ELU激活的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0.5, 6.5, -0.7])\n        >>> nn.elu(x)\n        jt.Var([ 0.5        6.5       -0.5034147], dtype=float32)\n        >>> nn.elu(x, 0.1)\n        jt.Var([ 0.5         6.5        -0.05034147], dtype=float32)\n\n"
    return jt.ternary(x>0,x,alpha*(x.exp()-1))

def sign(x: jt.Var) -> jt.Var:
    "\n这是一个元素级别（element-wise）函数。对输入变量x对应索引的元素应用符号函数。具体来说：当元素的值大于0时，返回1；当元素的值等于0时，返回0；当元素的值小于0时，返回-1，即：\n\n    .. math::\n        sign(x) = \n        \\begin{cases}\n        1, & \\text {if } x > 0 \\\\\n        0, & \\text {if } x = 0 \\\\\n        -1, & \\text {if } x < 0 \n        \\end{cases}\n\n    参数：\n        - x(Var): 输入的Var张量\n\n    返回值：\n        Var张量。张量x应用符号函数的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0, 6.5, -0.7])\n        >>> nn.sign(x)\n        jt.Var([ 0.  1. -1.], dtype=float32)\n\n\n"
    one = jt.ones(x.shape)
    x = jt.ternary(x>0, one, x)
    return jt.ternary(x<0, -one, x)

def gelu(x):
    "\n该函数为Jittor的GELU激活函数，它是非线性激活函数，在输入张量的每个元素乘上对应的高斯分布CDF（累积分布函数）：:math:`\\text{GELU}(x) = x * \\Phi(x)` ，其中 :math:`\\Phi(x)` 是高斯分布的累积分布函数。这是一个元素级别（element-wise）函数。\n\n    参数：\n        - x(Var): 输入的Var张量\n\n    返回值：\n        Var张量。张量x应用GELU激活的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0, 6.5, -0.7])\n        >>> nn.gelu(x)\n        jt.Var([ 0.          6.5        -0.16937456], dtype=float32)\n\n"
    _sqrt2 = 1.4142135623730951
    erf = jt.erf(x/_sqrt2)+1
    r = erf*x*.5
    return r

def silu(x):
    "\n该函数为Jittor的SILU(Sigmoid线性单元)激活函数。这是一个元素级别（element-wise）函数。具体来说，其计算过程如下：\n    .. math::\n        \\text{SILU}(x) = x\\ *\\ \\text{Sigmoid}(x)\n\n    参数：\n        - x(Var): 输入的Var张量\n\n    返回值：\n        Var张量。张量x应用SILU激活的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0, 6.5, -0.7])\n        >>> nn.silu(x)\n        jt.Var([ 0.          6.490242   -0.23226856], dtype=float32)\n\n\n"
    return x * x.sigmoid()

class ELU(Module):
    r''' Applies the element-wise function:

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param x: the input var
    :type x: jt.Var

    :param alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
    :param alpha: float, optional

    Example:
        >>> a = jt.randn(3)
        >>> a
        jt.Var([-0.38380373 -1.1338731   2.128115  ], dtype=float32)
        >>> nn.elu(a)
        jt.Var([-0.31873488 -0.6782155   2.128115  ], dtype=float32)
    '''
    def __init__(self,alpha=1.0):
        self.alpha=alpha
    
    def execute(self,x):
        return elu(x,self.alpha)

class PReLU(Module):
    r''' Applies the element-wise function:

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    :param x: the input var
    :type x: jt.Var

    :param num_parameters: number of :math:`a` to learn, can be either 1 or the number of channels at input. Default: 1
    :type num_parameters: int, optional

    :param init: the initial value of :math:`a`. Default: 0.25
    :param init: float, optional

    Example:
        >>> a = jt.randn(3)
        >>> prelu = nn.PReLU()
        >>> prelu(a)
        jt.Var([-0.09595093  1.1338731   6.128115  ], dtype=float32)
    '''

    def __init__(self, num_parameters=1, init_=0.25):
        self.num_parameters = num_parameters
        self.weight = init.constant((num_parameters,), "float32", init_)

    def execute(self, x):
        if self.num_parameters != 1:
            assert self.num_parameters == x.size(1), f"num_parameters does not match input channels in PReLU"
            return jt.maximum(0, x) + self.weight.broadcast(x, [0,2,3]) * jt.minimum(0, x)
        else:
            return jt.maximum(0, x) + self.weight * jt.minimum(0, x)

#TODO dims is 4 will cause slowly execution
def cross_entropy_loss(output, target, weight=None, ignore_index=None,reduction='mean'):
    "\n计算交叉熵损失。对于给定的类别标签`target`和网络输出结果`output`，计算交叉熵损失。该函数主要用于分类问题的损失计算。其具体计算公式如下，其中, *x* 是分类预测的输出，*class* 是真实的类别标签：\n\n    .. math::\n        L = -log( {e^{x[class]}}/{\\sum_{i}^{j}e^{x[i]}})\n\n    参数:\n        - output (jt.Var): 网络输出的结果，形状为(N, C)，这里N表示的是batchsize，而C为类别的数量，也可以为(N, H, W, C)形状，H，W分别为高和宽。\n        - target (jt.Var): 真实的类别标签，形状为(N,)或者(N, 1)，也可以为(N, H, W)形状。\n        - weight (jt.Var, 可选): 各个类别的权重，用于权衡不同类别对损失的贡献。默认值：None，如果提供此项，其形状应为(C,)。\n        - ignore_index (int, 可选): 需要忽略的类别标签。默认值：空，即不忽略任何标签。\n        - reduction (str, 可选): 指定减小损失的方式，可选值为`'none'`、`'mean'`或`'sum'`。`'none'`表示不执行任何减小损失，`'sum'`表示通过所有元素求和将损失减少，并且`'mean'`表示将损失通过所有元素平均值减小。默认值： `'mean'`\n\n    返回值:\n        jt.Var: 计算得到的交叉熵损失。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> output = jt.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0]])\n        >>> label = jt.array([1, 3, 2, 2])\n        >>>nn.cross_entropy_loss(output, label)\n        jt.Var([1.2181114], dtype=float32)\n\n"
    target_shape = target.shape
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))

    target = target.reshape((-1, ))
    target_weight = ((target >= 0) & (target < output.shape[1])).float32() 
    if weight is not None:
        target_weight = weight[target]
    if ignore_index is not None:
        target_weight = jt.ternary(
            target==ignore_index,
            jt.array(0).broadcast(target_weight),
            target_weight
        )
    
    target = target.broadcast(output, [1])
    target = target.index(1) == target
    
    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output*target).sum(1)) * target_weight
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean() / target_weight.mean()
    else:
        return loss.reshape(target_shape) 

def mse_loss(output, target, reduction="mean"):
    "\n计算均方误差（Mean Squared Error）损失。对于给定的类别标签`target`和网络输出结果`output`，计算均方误差损失。该函数主要用于分类问题的损失计算。可以选择损失函数输出方式，例如如果选择 `'mean'` ，则计算方式如下：\n\n    .. math::\n        L = \\frac{1}{n} \\sum_i^n (output[i] - target[i])^2\n\n    参数:\n        - output (jt.Var): 预测值，模型输出。如果output和target的类型不是jt.Var，会抛出TypeError。\n        - target (jt.Var): 目标值，实际值或者是标签。\n        - reduction (str, optional): 控制损失函数的输出方式，可以是'mean' （输出平均损失）, 'sum', 'none'之一。如果reduction的值不是'mean', 'sum', 'none'之一，会抛出ValueError。默认值：'mean'\n\n    返回值:\n        jt.Var: 均方误差损失。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> output = jt.array([1.0, 1.0, 1.0])\n        >>> target = jt.array([0.5, 0.6, -2.0])\n        >>> nn.mse_loss(output, target)\n        jt.Var([3.1366668], dtype=float32)\n\n"
    return (output-target).sqr().reduce(reduction)

def bce_loss(output, target, weight=None, size_average=True):
    "\n计算二分类交叉熵(Binary Cross Entropy)损失。对于给定的类别标签`target`和网络输出结果`output`，计算二分类交叉熵损失，如果指定了权重，则每个样本的损失会乘以权重：\n\n    .. math::\n        loss(target, output) = -[target * log(max(output, 1e^{-20})) + (1 - target) * log(max(1 - output, 1e{-20}))]\n\n    参数:\n        - output (jt.Var): 预测值，模型输出。形状为(batch_size, num_classes)，元素数据类型为float32\n        - target (jt.Var): 目标值，实际值或者是标签。形状应与output保持一致，元素数据类型为float32\n        - weight (float，可选): 每个样本的权重，如果指定，应该是一个1-D张量，长度等于batch_size。默认值: None\n        - size_average (bool): 是否对损失求平均。如果是True, 返回损失的平均值；否则，返回损失之和。默认值: True\n\n    返回值:\n        jt.Var: 二分类交叉熵(Binary Cross Entropy)损失。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> output = jt.array([1.0, 1.0, 1.0])\n        >>> target = jt.array([0.5, 0.6, -2.0])\n        >>> nn.bce_loss(output, target)\n        jt.Var([59.867214], dtype=float32)\n\n"
    loss = - (target * jt.log(jt.maximum(output, 1e-20)) + (1 - target) * jt.log(jt.maximum(1 - output, 1e-20)))

    if weight is not None:
        loss *= weight
    
    if size_average:
        return loss.mean()
    else:
        return loss.sum()

def l1_loss(output, target):
    "\n计算给定输出和目标之间的L1损失（平均绝对误差）。L1损失是预测值output和真实值target之间差值的绝对值的均值：\n\n    .. math::\n        L = \\frac{1}{n} ∑|output_i - target_i|\n\n    参数:\n        - output (jt.Var): 预测值，模型输出。可以是任意形状的张量\n        - target (jt.Var): 目标值，实际值或者是标签。形状应与output保持一致\n\n    返回值:\n        jt.Var: 一个标量张量，表示L1损失。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> output = jt.array([1.0, 1.0, 1.0])\n        >>> target = jt.array([0.5, 0.6, -2.0])\n        >>> nn.l1_loss(output, target)\n        jt.Var([1.3], dtype=float32)\n\n"
    return (output-target).abs().mean()


def smooth_l1_loss(y_true, y_pred,reduction="mean"):
    "\n计算给定输出和目标之间的Smooth-L1损失：\n    .. math::\n        Smooth-L1(x, y) =\\left{\\{}\\begin{array}{11}0.5 * (y-x)^2    & abs(y − x) < 1\\\\\n                 abs(y − x) - 0.5  & otherwise\\end{array}\n\n    参数:\n        - y_true (jt.Var): 真实值，通常为[N, 4]的形状，但是也可以是其他形状。\n        - y_pred (jt.Var): 预测值，通常为[N, 4]的形状，但是也可以是其他形状。\n        - reduction (str, 可选): 计算损失的方式，其值只能在['mean', 'sum', 'none']中选择。默认值： ``'mean'``\n    返回值:\n        jt.Var: 一个标量张量，表示Smooth-L1损失。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> output = jt.array([1.0, 1.0, 1.0])\n        >>> target = jt.array([0.5, 0.6, -2.0])\n        >>> nn.smooth_l1_loss(target, output) \n        jt.Var([0.9016667], dtype=float32)\n\n"
    diff = jt.abs(y_true - y_pred)
    less_than_one = (diff<1.0).float32()
    loss = (less_than_one * 0.5 * diff.sqr()) + (1 - less_than_one) * (diff - 0.5)
    if reduction=="mean":
        return loss.mean()
    elif reduction=="sum":
        return loss.sum()
    elif reduction=="none":
        return loss
    else:
        raise ValueError(f'not support {reduction}')

def nll_loss(output,target,weight=None,ignore_index=-100,reduction='mean'):
    "\n根据给定的目标概率密度函数(target),计算负对数似然损失(negative log likelihood loss)。对于输入中的每个元素，计算该元素在目标处的负对数似然概率： :math:`loss(x, class) = -weight[class] * log(x[class])` ，其中log(x[class]) 是预测为class这一类的概率取对数。如果 ``reduction=='mean'`` ，那么 :math:`Out(n, c) = -weight[c]/\\sum(weight) * log(input_{n, c})`\n\n    参数:\n        - output (jt.Var) : 输出张量，具体维度或形状取决于具体的损失函数。对于nll_loss，input的形状是(minibatch, C)。\n        - target (jt.Var) : 目标张量。其定义在[0，class_num-1]之间的取值表示对应输入数据的类别标签。\n        - weight (jt.Var, 可选) : 一个手动指定每个类别的权重的张量。默认值：None\n        - ignore_index (int, 可选) : 指定一个值，在计算损失函数时忽略目标值中等于指定值的元素。默认值：-100\n        - reduction (string, 可选) : 指定如何减少损失：'none' (不减少) | 'mean' (加权减少) | 'sum' (简单元素相加)。默认值： ``'mean'``\n\n    返回值:\n        jt.Var: 一个标量张量，表示负对数似然损失。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> output = jt.array([[0.1, 0.1, 0.8], [0.2, 0.4, 0.4], [0.2, 0.6, 0.2]])\n        >>> target = jt.array([3, 2, 2])\n        >>> nn.nll_loss(output, target)\n        jt.Var([-0.3], dtype=float32)\n\n"
    assert output.ndim<=2 and output.ndim>0 and target.ndim==1
    n_classes = output.shape[-1]
    assert weight is None or weight.numel()==n_classes
    assert ignore_index<0 or ignore_index<n_classes
    if weight is None:
        weight = jt.ones((n_classes,))
    if ignore_index>0:
        weight[ignore_index]=0
    if output.ndim==2:
        index = jt.index((output.shape[0],),dim=0)
        loss = -output[index,target]*weight[target]
    else:
        loss = -output[target[0]]*weight[target[0]]
    if reduction=="mean":
        total_weight  = weight[target].sum() if output.ndim==2 else weight[target[0]].sum()
        return loss.sum()/total_weight
    elif reduction=="sum":
        return loss.sum()
    elif reduction=="none":
        return loss
    else:
        raise ValueError(f'not support {reduction}')
    
class CrossEntropyLoss(Module):
    def __init__(self, weight=None, ignore_index=None):
        self.weight = weight
        self.ignore_index = ignore_index
        
    def execute(self, output, target):
        return cross_entropy_loss(output, target, self.weight, self.ignore_index)

class MSELoss(Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    def execute(self, output, target):
        return mse_loss(output, target, self.reduction)

class BCELoss(Module):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average
    def execute(self, output, target):
        return bce_loss(output, target, self.weight, self.size_average)

class L1Loss(Module):
    def __init__(self):
        pass
    def execute(self, output, target):
        return l1_loss(output, target)

def binary_cross_entropy_with_logits(output, target, weight=None, pos_weight=None, size_average=True):
    "\n该函数用于计算具有logits的二元交叉熵损失，基于Sigmoid激活函数实现。该函数对于解决数据不平衡问题是非常有效的。\n\n    参数:\n        - output (jt.Var) : 网络的输出张量，元素float32/float64，形状用[batch_size,*]表示。其中'*'代表任意的其他尺寸。\n        - target (jt.Var) : 目标张量，类型同output，形状与output相同。\n        - weight (jt.Var, 可选) : 一个手动指定每个类别的权重的张量。默认值：None\n        - pos_weight (jt.Var, 可选): 正样本的权重。如果给定，必须为张量，而与output形状相同，且类型与output类型相同。默认值：None\n        - size_average (bool, 可选): 如果为True，返回交叉熵损失的平均值。否则，返回交叉熵损失的总和。默认值：True\n\n    返回值:\n        jt.Var: 与output形状相同的二元交叉熵损失张量，具有同样的数据类型。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> output = jt.array([0.2, 1, -0.3])\n        >>> target = jt.array([3, 2, 2])\n        >>> nn.binary_cross_entropy_with_logits(output, target)\n        jt.Var([0.22191863], dtype=float32)\n\n"
    max_val = jt.clamp(-output,min_v=0)
    if pos_weight is not None:
        log_weight = (pos_weight-1)*target + 1
        loss = (1-target)*output+(log_weight*(((-max_val).exp()+(-output - max_val).exp()).log()+max_val))
    else:
        loss = (1-target)*output+max_val+((-max_val).exp()+(-output -max_val).exp()).log()
    if weight is not None:
        loss *=weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()

class BCEWithLogitsLoss(Module):
    def __init__(self, weight=None, pos_weight=None, size_average=True):
        self.pos_weight = pos_weight
        self.weight = weight
        self.size_average = size_average

    def execute(self, output, target):
        return binary_cross_entropy_with_logits(output,target,self.weight,self.pos_weight,self.size_average)

def softmax(x, dim=None, log=False):
    "\n此函数接收一个输入张量x，并在指定的维度上应用softmax函数。如果参数\"log\"为True，则应用log_softmax。具体公式如下：\n\n    给定输入 :math:`x` ， softmax函数可以计算为：\n\n    .. math::\n        \\text{softmax}(x_i) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}\n\n    当参数\"log\"为True时，log_softmax函数可以计算为：\n\n    .. math::\n        \\text{log\\_softmax}(x_i) = \\log(\\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)})\n\n    参数:\n        - x (jt.Var): 输入张量，可以是任意维度。\n        - dim (int or tuple(int), 可选): 指定softmax操作的维度。不指定时，操作会应用于所有的元素上。默认值：None\n        - log (bool, optional): 如果设为True，则应用log_softmax操作，否则应用普通的softmax操作。默认值：False\n\n    返回值:\n        jt.Var: 与输入 x 形状相同的张量，其各元素 :math:`y_i` 等于输入x在dim维度上的softmax或者log_softmax的结果。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0.8, 0.2, 0.5])\n        >>> nn.softmax(x)\n        jt.Var([0.4367518  0.23969446 0.32355368], dtype=float32)\n\n"
    import jittor.other.code_softmax as code_softmax
    if code_softmax.can_softmax_v1(x, dim) and jt.compiler.is_cuda:
        return code_softmax.softmax_v1(x, log)
    if dim is None: dim = ()
    dtype, x = x.dtype, x._to_float()
    if log:
        a = x - jt.max(x, dim, keepdims=True)
        ret = a - a.exp().sum(dim, keepdims=True).log()
    else:
        x = (x - jt.max(x, dim, keepdims=True)).exp()
        ret = x / x.sum(dim, keepdims=True)
    return ret.cast(dtype)
jt.Var.softmax = softmax

def log_softmax(x,dim=None):
    "\n对输入的张量进行softmax操作并取对数:\n\n    .. math::\n        \\text{log\\_softmax}(x_i) = \\log(\\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)})\n\n    参数:\n        - x (jt.Var): 输入张量，可以是任意维度。\n        - dim (int or tuple(int), 可选): 指定softmax操作的维度。不指定时，操作会应用于所有的元素上。默认值：None\n\n    返回值:\n        jt.Var: 与输入 x 形状相同的张量，其各元素 :math:`y_i` 等于输入x在dim维度上的log_softmax的结果。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0.8, 0.2, 0.5])\n        >>> nn.log_softmax(x)\n        jt.Var([-0.8283902 -1.4283903 -1.1283902], dtype=float32)\n\n"
    return softmax(x,dim=dim, log=True)
jt.Var.log_softmax = log_softmax

def log_sigmoid(x):
    "\n将输入的张量x传入sigmoid函数后，然后对其求对数。使用此函数可以帮助我们在进行深度学习或者神经网络计算中，更好地平滑输入值。\n\n    .. math::\n         log\\_sigmoid(x) = log\\left(\\frac{1}{1+e^{-x}}\\right)\n\n    参数:\n        - x (jt.Var): 输入张量，可以是任意维度。\n\n    返回值:\n        jt.Var: 与输入 x 形状相同的张量，其各元素等于输入x在dim维度上的log_sigmoid的结果。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([0.8, 0.2, 0.5])\n        >>> nn.log_sigmoid(x)\n        jt.Var([-0.37110066 -0.5981389  -0.47407696], dtype=float32)\n\n"
    return jt.log(jt.sigmoid(x))
jt.Var.log_sigmoid = log_sigmoid

def logsumexp(x, dim, keepdims=False, keepdim=False):
    "\n计算输入张量x在给定维度上的对数和指数。实现的是下列公式，其中，:math:`x_{i}` 是 `x` 的元素：\n\n    .. math::\n        log(\\sum_{i}exp(x_{i}))\n\n    参数:\n        - x (jt.Var): 输入张量，可以是任意维度。\n        - dim (int 或 tuple of int): 用于指定计算logsumexp的行或列轴的编号或元组\n        - keepdims (Boolean, 可选): 如果此选项设为 True，那么求和张量的选定维度将被保留。默认值：False\n        - keepdim (Boolean, 可选): 与keepdims相同。默认值：False\n\n    返回值:\n        jt.Var: x在给定维度上计算对数和指数得到的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.ones(3,3)\n        >>> nn.logsumexp(x, dim=0)\n        jt.Var([2.0986123 2.0986123 2.0986123], dtype=float32)\n        >>> nn.logsumexp(x, dim=0, keepdims=True)\n        jt.Var([[2.0986123 2.0986123 2.0986123]], dtype=float32)\n\n"
    return x.exp().sum(dim, keepdim or keepdims).log()
jt.Var.logsumexp = logsumexp

class Identity(Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def execute(self, input):
        return input

def identity(input): return input

class Dropout(Module):
    def __init__(self, p=0.5, is_train=False):
        assert p >= 0 and p <= 1, "dropout probability has to be between 0 and 1, but got {}".format(p)
        self.p = p
        self.is_train = is_train
        #TODO: test model.train() to change self.is_train
    def execute(self, input):
        output = input
        if self.p > 0 and self.is_train:
            if self.p == 1:
                noise = jt.zeros(input.shape)
                output = output * noise
            else:
                noise = jt.random(input.shape)
                noise = (noise > self.p).int()
                output = output * noise / (1.0 - self.p) # div keep prob
        return output

def dropout(x,p=0.5,is_train=False):
    "\n该函数实现了dropout操作。此函数会在训练阶段随机将输入张量x中约p比例的元素设置为0，从而防止过拟合。在测试阶段，此函数会返回原始的输入张量x，而不进行dropout操作。在数学上，dropout操作表示如下，这里的 :math:`\\frac{x}{1-p}` 操作确保了在训练和测试阶段，该层的输出的期望保持不变：\n\n    .. math::\n        y = \\begin{cases}\n        0 & 概率 p \\\\\n        \\frac{x}{1-p} & 概率1-p\n        \\end{cases}\n\n    参数:\n        - x (jt.Var): 输入张量\n        - p (float, 可选): dropout的概率。它是一个介于0和1之间的float值，表示每个元素被设置为0的概率。例如，如果p=0.5，那么输入张量中约有一半的元素会被设置为0。默认值: 0.5\n        - is_train (bool, 可选): 一个布尔值，表示是否在训练模式下运行。如果 ``is_train==True`` ，那么会执行dropout操作；如果is_train=False，那么返回原始的输入张量x。默认值: False\n\n    返回值:\n        jt.Var: 一个和输入张量x相同形状的张量，为x执行完dropout操作的结果。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.ones(3,3)\n        >>> nn.dropout(x)\n        jt.Var([[1. 1. 1.]\n                [1. 1. 1.]\n                [1. 1. 1.]], dtype=float32)\n        >>> nn.dropout(x, is_train=True) \n        jt.Var([[0. 0. 0.]\n                [0. 2. 2.]\n                [0. 2. 2.]], dtype=float32)\n\n"
    return Dropout(p=p,is_train=is_train)(x)

class Dropout2d(Module):
    def __init__(self, p=0.5, is_train=False):
        '''
        Randomly zero out entire channels, from "Efficient Object Localization Using Convolutional Networks"
        input:
            x: [N,C,H,W] or [N,C,L]
        output:
            y: same shape as x
        '''
        assert p >= 0 and p <= 1, "dropout probability has to be between 0 and 1, but got {}".format(p)
        self.p = p
        self.is_train = is_train
        #TODO: test model.train() to change self.is_train
    def execute(self, input):
        output = input
        shape = input.shape[:-2]
        if self.p > 0 and self.is_train:
            if self.p == 1:
                output = jt.zeros(input.shape)
            else:
                noise = jt.random(shape)
                noise = (noise > self.p).int()
                output = output * noise.broadcast(input.shape, dims=[-2,-1]) / (1.0 - self.p) # div keep prob
        return output

def dropout2d(x,p=0.5,is_train=False):
    "\n对2D输入数据执行Dropout操作。此函数会在训练阶段随机将输入张量x中约p比例的元素设置为0，从而防止过拟合。通过这种方式，Dropout可以等同于对大量不同的神经网络进行模型平均。\n\n    在测试阶段，此函数会返回原始的输入张量x，而不进行dropout操作。在数学上，dropout操作表示如下，这里的 :math:`\\frac{x}{1-p}` 操作确保了在训练和测试阶段，该层的输出的期望保持不变，y是输出结果，x是输入数据：\n\n    .. math::\n        y = \\begin{cases}\n        0 & 概率 p \\\\\n        \\frac{x}{1-p} & 概率1-p\n        \\end{cases}\n\n    参数:\n        - x (jt.Var): 输入张量，应为2D的Jittor数组。\n        - p (float, 可选): dropout的概率。它是一个介于0和1之间的float值，表示每个元素被设置为0的概率。例如，如果p=0.5，那么输入张量中约有一半的元素会被设置为0。默认值: 0.5\n        - is_train (bool, 可选): 一个布尔值，表示是否在训练模式下运行。如果is_train=True ，那么会执行dropout操作；如果is_train=False，那么返回原始的输入张量x。默认值: False\n\n    返回值:\n        jt.Var: 一个和输入张量x相同形状的张量，为x执行完dropout操作的结果。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([[0, 0.5, 1.0], [-0.3, 0.5, 0.8]])\n        >>> nn.dropout2d(x, 0.2, True) \n        jt.Var([[[ 0.     0.625  1.25 ]\n                [-0.375  0.625  1.   ]]], dtype=float32)\n        >>> nn.dropout2d(x)            \n        jt.Var([[ 0.   0.5  1. ]\n                [-0.3  0.5  0.8]], dtype=float32)\n\n"
    return Dropout2d(p=p,is_train=is_train)(x)

class DropPath(Module):
    '''Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    '''
    def __init__(self, p=0.5, is_train=False):
        '''
            :param p: Specifies the probability of each batch retention. Defaults to 0.5.
            :type p: float dtype
            :param is_train: Specify whether it is a training model. Defaults to False.
            :type is_train: bool
        '''
        self.p = p
        self.is_train = is_train
        #TODO: test model.train() to change self.is_train
    def execute(self, x):
        if self.p == 0. or not self.is_train:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
        output = x.divide(keep_prob) * random_tensor.floor()
        return output

def droppath(x,p=0.5,is_train=False):
    "\n这个函数通过DropPath算法实现对输入的影响，其中概率参数p和训练状态is_train可以被调整。其数学表示如下，其中 :math:`y_i` 是输出张量, :math:`x_i` 是输入张量, :math:`z_i` 是从均匀分布[0, 1]中随机采样的一个随机值：\n\n    .. math::\n        y_i = \\begin{cases}\n        0, & \\text{if } z_i < p \\text{ and } \\text{is\\_train} = \\text{True} \\\\\n        x_i, & \\text{otherwise}\n        \\end{cases}\n\n    参数:\n        - x (jt.Var): 输入张量\n        - p (float, 可选): DropPath算法的概率参数。默认值: 0.5\n        - is_train (bool, 可选): 一个布尔值，表示是否在训练模式下运行。如果is_train=True ，那么会执行dropout操作；如果is_train=False，那么返回原始的输入张量x。默认值: False\n\n    返回值:\n        jt.Var: 一个和输入张量x相同形状的张量，为x执行完DropPath操作的结果。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([[0, 0.5, 1.0], [-0.3, 0.5, 0.8]])\n        >>> nn.droppath(x, is_train=True) \n        jt.Var([[ 0.  1.  2.]\n                [-0.  0.  0.]], dtype=float32)\n        >>> nn.droppath(x)  \n        jt.Var([[ 0.   0.5  1. ]\n                [-0.3  0.5  0.8]], dtype=float32)\n\n"
    return DropPath(p=p,is_train=is_train)(x)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init.invariant_uniform((out_features, in_features), "float32")
        bound = 1.0/math.sqrt(in_features)
        self.bias = init.uniform((out_features,), "float32",-bound,bound) if bias else None

    def execute(self, x):
        x = matmul_transpose(x, self.weight)
        if self.bias is not None:
            return x + self.bias
        return x
    
def linear(x, weight, bias=None):
    "\n对输入x进行线性变换。此函数返回x与权重weight的矩阵乘法的结果，如果传入了偏置bias，则在进行矩阵乘法后还会加上偏置: :math:`x\\ *\\ weight^T + bias`\n\n    参数:\n        - x (jittor.Var): 输入张量，大小可以是(batch_size, input_dim)\n        - weight (jittor.Var): 权重矩阵，大小可以是(batch_size, input_dim)\n        - bias (jittor.Var, 可选): 偏置向量，大小可以是(output_dim,)。默认值: None\n\n    返回值:\n        jittor.Var: 线性变换后的结果，大小可以是(batch_size, output_dim)\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([[0, 0.5, 1.0], [-0.3, 0.5, 0.8]])\n        >>> weight = jt.ones(3,3)     \n        >>> nn.linear(x, weight)   \n        jt.Var([[1.5 1.5 1.5]\n                [1.  1.  1. ]], dtype=float32)\n\n"
    ''' Returns x * weight^T
    '''
    x = matmul_transpose(x, weight)
    if bias is not None:
        return x + bias
    return x

class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, is_train=True, sync=True):
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()

    def execute(self, x):
        dims = [0]+list(range(2,x.ndim))
        if self.is_train:
            xmean = jt.mean(x, dims=dims)
            x2mean = jt.mean(x*x, dims=dims)
            if self.sync and jt.in_mpi:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = (x2mean-xmean*xmean).maximum(0.0)
            w = self.weight / jt.sqrt(xvar+self.eps)
            b = self.bias - xmean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)

            self.running_mean.update(self.running_mean +
                (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
            self.running_var.update(self.running_var +
                (xvar.reshape((-1,))-self.running_var)*self.momentum)
            return norm_x
        else:
            w = self.weight / jt.sqrt(self.running_var+self.eps)
            b = self.bias - self.running_mean * w
            norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
            return norm_x

BatchNorm3d = BatchNorm2d = BatchNorm1d = BatchNorm

def batch_norm(x, running_mean, running_var, weight=1, bias=0, training=False, momentum=0.1, eps=1e-05):
    "\n对输入的张量进行批量归一化。批量归一化是一种用于提高神经网络性能和稳定性的技术。该操作可以使网络处理的内部协变量变化变得更小。 在正向传播时，该层会将输入中心化（均值为0）和规范化（方差为1），然后将结果乘以比例因子 gamma 并加上偏移量 beta。在反向传播时，本层将计算当前批次的均值和方差的梯度。计算方式如下:\n\n    .. math::\n\n        norm_x = \\frac{x - running\\_mean} {\\sqrt{running\\_var + eps}} * weight + bias\n\n\n    参数:\n\n        - `x` (jittor.Var) : 输入张量。维度 (batch_size, num_channels, ..)。\n        - `running_mean` (jittor.Var) : 运行期间的平均值，用于将输入中心化。维度和 num_channels 相同。\n        - `running_var` (jittor.Var) :  运行期间的方差，用于将输入规范化。维度和 num_channels 相同。\n        - `weight` (float，可选) : 批量归一化的缩放系数。默认值: 1\n        - `bias` (float，可选) : 批量归一化的平移系数。默认值: 0\n        - `training` (bool，可选) : 为 True 时， 化函数使用在线定义的计数方法对batch数据进行归一化。只有当你使用的模型在每次训练时修改输入的统计数据,并保存更新后的统计数据以备后续使用时，才需要使用该参数。如果训练数据在每一次训练过程中都保持不变，就无需使用该参数。默认值: False\n        - `momentum` (float, 可选) : 动量值，有效值为 [0,1] 。默认值: 0.1\n        - `eps` (float，可选) : 用作分母以增加数值稳定性的项。默认值: 1e-5\n\n    返回值:\n        jittor.Var: 线性变换后的结果，大小可以是(batch_size, output_dim)\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([[0, 0.5, 1.0], [-0.3, 0.5, 0.8]])\n        >>> running_mean = jt.mean(x, dim=0) # 实际应用时，自行计算\n        >>> running_var = jt.var(x, dim=0) # 实际应用时，自行计算\n        >>> nn.batch_norm(x, running_mean, running_var) \n        jt.Var([[ 0.99977785  0.          0.9995003 ]\n                [-0.99977785  0.         -0.99950075]], dtype=float32\n\n"
    dims = [0]+list(range(2,x.ndim))
    assert not training
    w = weight / jt.sqrt(running_var+eps)
    b = bias - running_mean * w
    norm_x = x * w.broadcast(x, dims) + b.broadcast(x, dims)
    return norm_x


class InstanceNorm(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, is_train=True, sync=True):
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum

        self.affine = affine
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0

    def execute(self, x):
        dims = list(range(2,x.ndim))
        xmean = jt.mean(x, dims=dims)
        x2mean = jt.mean(x*x, dims=dims)

        xvar = (x2mean-xmean*xmean).maximum(0.0)
        w = self.weight / jt.sqrt(xvar+self.eps)
        b = self.bias - xmean * w
        return x * w.broadcast(x, dims) + b.broadcast(x, dims)

InstanceNorm3d = InstanceNorm2d = InstanceNorm1d = InstanceNorm

def fp32_guard(func):
    def wrapper(*args, **kw):
        if jt.flags.amp_level == 0:
            return func(*args, **kw)
        new_args = []
        need_cast = False
        dtype = None
        for a in args:
            if isinstance(a, jt.Var) and (a.dtype == "float16" or a.dtype == "bfloat16"):
                dtype = a.dtype
                new_args.append(a.float32())
                need_cast = True
            else:
                new_args.append(a)
        with jt.flag_scope(amp_level=0):
            a = func(*new_args, **kw)
            if need_cast and isinstance(a, jt.Var) and a.dtype == "float32":
                a = a.cast(dtype)
        return a
    return wrapper

def instance_norm(x, 
    running_mean = None,
    running_var = None,
    weight = 1,
    bias = 0,
    momentum = 0.1,
    eps = 1e-5):
    dims = list(range(2,x.ndim))
    xmean = jt.mean(x, dims=dims)
    x2mean = jt.mean(x*x, dims=dims)

    xvar = (x2mean-xmean*xmean).maximum(0.0)
    w = weight / jt.sqrt(xvar+eps)
    b = bias - xmean * w
    return x * w.broadcast(x, dims) + b.broadcast(x, dims)

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = init.constant(normalized_shape, "float32", 1.0) if elementwise_affine else 1.0
        self.bias = init.constant(normalized_shape, "float32", 0.0) if elementwise_affine else 0.0

    @fp32_guard
    def execute(self, x):
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]
        xmean = jt.mean(x, dims=dims, keepdims=1)
        x2mean = jt.mean(x*x, dims=dims, keepdims=1)

        xvar = (x2mean-xmean*xmean).maximum(0.0)
        w = self.weight / jt.sqrt(xvar+self.eps)
        b = self.bias - xmean * w
        return x * w + b


LayerNorm3d = LayerNorm2d = LayerNorm1d = LayerNorm

@fp32_guard
def layer_norm(x, 
    normalized_shape, 
    weight = 1,
    bias = 0,
    eps: float = 1e-5, 
    elementwise_affine: bool = True):
    dims = [-i for i in range(len(normalized_shape), 0, -1)]
    xmean = jt.mean(x, dims=dims, keepdims=1)
    x2mean = jt.mean(x*x, dims=dims, keepdims=1)

    xvar = (x2mean-xmean*xmean).maximum(0.0)
    w = weight / jt.sqrt(xvar+eps)
    b = bias - xmean * w
    return x * w + b

class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, is_train=True):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.affine = affine
        self.weight = init.constant((num_channels,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_channels,), "float32", 0.0) if affine else 0.0

    def execute(self, x):
        N = x.shape[0]
        C = self.num_channels
        output_shape = (N,-1)
	    # TODO: 3d group norm
        if x.ndim==4:
            output_shape = x.shape
        assert C % self.num_groups == 0
        x = x.reshape((N, self.num_groups, C//self.num_groups, -1))
        xmean = jt.mean(x, dims=[2,3]).reshape((N, self.num_groups, 1))
        x2mean = jt.mean(x*x, dims=[2,3]).reshape((N, self.num_groups, 1))
        xvar = (x2mean-xmean*xmean).maximum(0.0)

        if self.affine:
            w = self.weight.reshape((1, self.num_groups, -1))
            b = self.bias.reshape((1, self.num_groups, -1))
        else:
            w = 1
            b = 0
        w = w / jt.sqrt(xvar+self.eps)
        b = b - xmean * w
        x = x * w.broadcast(x, [3]) + b.broadcast(x, [3])
        return x.reshape(output_shape)

def group_norm(x, 
    num_groups, 
    weight = 1,
    bias = 0,
    eps=1e-05):
    N = x.shape[0]
    C = x.shape[1]
    output_shape = (N,-1)
    # TODO: 3d group norm
    if x.ndim==4:
        output_shape = x.shape
    assert C % num_groups == 0
    x = x.reshape((N, num_groups, C//num_groups, -1))
    xmean = jt.mean(x, dims=[2,3]).reshape((N, num_groups, 1))
    x2mean = jt.mean(x*x, dims=[2,3]).reshape((N, num_groups, 1))
    xvar = (x2mean-xmean*xmean).maximum(0.0)

    if isinstance(weight, jt.Var):
        weight = weight.reshape((1, num_groups, -1))
    if isinstance(bias, jt.Var):
        bias = bias.reshape((1, num_groups, -1))
    weight = weight / jt.sqrt(xvar+eps)
    bias = bias - xmean * weight
    x = x * weight.broadcast(x, [3]) + bias.broadcast(x, [3])
    return x.reshape(output_shape)


Relu = jt.make_module(relu)
ReLU = Relu
Leaky_relu = jt.make_module(leaky_relu, 2)
LeakyReLU = Leaky_relu
ReLU6 = jt.make_module(relu6)
Softmax = jt.make_module(softmax, 2)
GELU = jt.make_module(gelu)

class Flatten(Module):
    ''' Flattens the contiguous range of dimensions in a Var.

    :param start_dim: the first dimension to be flattened. Defaults: 1.
    :type start_dim: int

    :param end_dim: the last dimension to be flattened. Defaults: -1.
    :type end_dim: int
    '''
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def execute(self, x) -> jt.Var:
        return x.flatten(self.start_dim, self.end_dim)


from jittor.depthwise_conv import DepthwiseConv

class Conv(Module):
    ''' Applies a 2D convolution over an input signal composed of several input planes.

    :param in_channels: Number of channels in the input feature map
    :type in_channels: int

    :param out_channels: Number of channels in the output feature map
    :type out_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    Example:

    >>> conv = nn.Conv2d(24, 32, 3)
    >>> conv = nn.Conv2d(24, 32, (3,3))
    >>> conv = nn.Conv2d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv2d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 100, 100)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.is_depthwise_conv = self.groups == self.out_channels and self.groups == self.in_channels
        if self.is_depthwise_conv and jt.flags.use_cuda and jt.compiler.is_cuda:
            self.depthwise_conv = DepthwiseConv(stride, padding, dilation)
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        Kh, Kw = self.kernel_size

        # self.weight = init.relu_invariant_gauss([out_channels, in_channels//groups, Kh, Kw], dtype="float", mode="fan_out")
        self.weight = init.invariant_uniform([out_channels, in_channels//groups, Kh, Kw], dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        if hasattr(self, 'depthwise_conv'):
            y = self.depthwise_conv(x, self.weight)
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        elif self.groups == 1:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            assert C==self.in_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            assert oh>0 and ow>0
            with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
                xx = x.reindex([N,self.out_channels,C,oh,ow,Kh,Kw], [
                    'i0', # Nid
                    'i2', # Cid
                    f'i3*{self.stride[0]}-{self.padding[0]}+i5*{self.dilation[0]}', # Hid+Khid
                    f'i4*{self.stride[1]}-{self.padding[1]}+i6*{self.dilation[1]}', # Wid+KWid
                ])
                ww = self.weight.broadcast(xx.shape, [0,3,4])
                yy = xx*ww
                y = yy.sum([2,5,6]) # Kc, Kh, Kw
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        else:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            G = self.groups
            CpG = C // G # channels per group
            assert C==self.in_channels
            oc = self.out_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            assert oh>0 and ow>0
            xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            # w: [oc, CpG, Kh, Kw]
            ww = self.weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
            ww.compile_options = xx.compile_options = {"G":G,"C":C}
            yy = xx*ww
            y = yy.reindex_reduce('add', [N, oc, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y          

Conv2d = Conv

class Conv1d(Module):
    ''' Applies a 1D convolution over an input signal composed of several input planes.

    :param in_channels: Number of channels in the input feature map
    :type in_channels: int

    :param out_channels: Number of channels in the output feature map
    :type out_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    Example:

    >>> conv = nn.Conv1d(24, 32, 3)
    >>> conv = nn.Conv1d(24, 32, (3,3))
    >>> conv = nn.Conv1d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv1d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 100)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, 1)
        self.stride = (stride, 1)
        self.padding = (padding, 0)
        self.dilation = (dilation, 1)
        self.groups = groups
        self.bias = bias
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        # using list to escape module dfs
        self._conv = [Conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)]
        self.weight = self._conv[0].weight.squeeze(-1)
        self.bias = self._conv[0].bias

    def execute(self, x):
        N,C,D = x.shape
        assert C==self.in_channels
        self._conv[0].weight = self.weight.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = self._conv[0](x)
        y = x.squeeze(-1)
        return y

class Conv3d(Module):
    ''' Applies a 3D convolution over an input signal composed of several input planes.

    :param in_channels: Number of channels in the input feature map
    :type in_channels: int

    :param out_channels: Number of channels in the output feature map
    :type out_channels: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int or tuple

    :param stride: Stride of the convolution. Default: 1
    :type stride: int or tuple, optional

    :param padding: Padding added to all four sides of the input. Default: 0
    :type padding: int or tuple, optional

    :param dilation: Spacing between kernel elements. Default: 1
    :type dilation: int or tuple, optional

    :param groups: Number of blocked connections from input channels to output channels. Default: 1
    :type groups: int, optional

    :param bias: If True, adds a learnable bias to the output. Default: True
    :type bias: bool, optional

    Example:

    >>> conv = nn.Conv3d(24, 32, 3)
    >>> conv = nn.Conv3d(24, 32, (3,3))
    >>> conv = nn.Conv3d(24, 32, 3, stride=2, padding=1)
    >>> conv = nn.Conv3d(24, 32, 3, dilation=(3, 1))
    >>> input = jt.randn(4, 24, 50, 50, 50)
    >>> output = conv(input)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        Kh, Kw, Kd = self.kernel_size
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        self.weight = init.invariant_uniform([out_channels, in_channels//groups, Kh, Kw, Kd], dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        return conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv1d_sp(Linear):
    def __init__(self, inchannels, outchannels, kernel_size=1, bias=True):
        super().__init__(inchannels, outchannels, bias=bias)
        assert kernel_size == 1

    def execute(self, x):
        x = x.transpose(0, 2, 1)
        x = super().execute(x)
        x = x.transpose(0, 2, 1)
        return x

def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    "\n将一个2D卷积应用于由多个输入平面组成的输入信号。\n\n    参数:\n        - `x` (jittor.Var) : 输入张量\n        - `weight` (jittor.Var) : 卷积核\n        - `bias` (jittor.Var，可选) : 卷积后的偏置。默认值：None\n        - `stride` (int或tuple，可选) : 卷积的步长。默认值: 1\n        - `padding` (int或tuple, 可选) : 输入的四周添加的填充长度。默认值: 0\n        - `dilation` (int或tuple，可选) : 卷积核元素之间的间距。默认值: 1\n        - `groups` (int，可选) : 输入通道和输出通道之间的阻塞连接数。默认值: 1\n\n    返回值:\n        jittor.Var: 执行2D卷积之后的结果张量\n\n    代码示例：\n        >>> x = jt.randn(4, 24, 100, 100)\n        >>> w = jt.randn(32, 24, 3, 3)\n        >>> y = nn.conv2d(x, w)\n\n"
    padding = _pair(padding)
    stride = _pair(stride)
    dilation = _pair(dilation)
    out_channels = weight.shape[0]

    if groups == 1:
        N,C,H,W = x.shape
        Kh, Kw = weight.shape[-2:]
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        with jt.flag_scope(amp_reg = jt.flags.amp_reg | 36):
            xx = x.reindex([N,out_channels,C,oh,ow,Kh,Kw], [
                    'i0', # Nid
                    'i2', # Cid
                    f'i3*{stride[0]}-{padding[0]}+i5*{dilation[0]}', # Hid+Khid
                    f'i4*{stride[1]}-{padding[1]}+i6*{dilation[1]}', # Wid+KWid
                ])
            ww = weight.broadcast(xx.shape, [0,3,4])
            yy = xx*ww
            y = yy.sum([2,5,6]) # Kc, Kh, Kw
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y
    else:
        N,C,H,W = x.shape
        Kh, Kw = weight.shape[-2:]
        G = groups
        CpG = C // G # channels per group
        oc = out_channels
        oh = (H+padding[0]*2-Kh*dilation[0]+dilation[0]-1)//stride[0]+1
        ow = (W+padding[1]*2-Kw*dilation[1]+dilation[1]-1)//stride[1]+1
        xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
                f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
            ])
        xx.compile_options = {"G":G}
        # w: [oc, CpG, Kh, Kw]
        ww = weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
        yy = xx*ww
        y = yy.reindex_reduce('add', [N, oc, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y
conv = conv2d

def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    "\n该函数实现了3D卷积运算。此函数首先将padding、stride、dilation参数进行处理。然后计算输出通道的数量。如果支持cuda并且使用cudnn，则进行cuda加速的卷积运算。如果不支持cuda或者groups参数不为1，则进行常规的卷积运算。最后，如果提供了bias参数，则在卷积后加上偏差。最终返回卷积后的结果。\n\n    参数:\n        - `x` (jittor.Var) : 输入张量\n        - `weight` (jittor.Var) : 卷积核\n        - `bias` (jittor.Var，可选) : 卷积后的偏置，如未输入默认无偏置。默认值：None\n        - `stride` (int或tuple，可选) : 卷积的步长。默认值: 1\n        - `padding` (int或tuple, 可选) : 输入的四周添加的填充长度。默认值: 0\n        - `dilation` (int或tuple，可选) : 卷积核元素之间的间距。默认值: 1\n        - `groups` (int，可选) : 输入通道和输出通道之间的阻塞连接数。默认值: 1\n\n    返回值:\n        jittor.Var: 执行2D卷积之后的结果张量\n\n    代码示例：\n        >>> x = jt.randn(4, 24, 50, 50, 50)\n        >>> w = jt.randn(32, 24, 3, 3, 3)\n        >>> y = nn.conv3d(x, w)\n\n"
    padding = _triple(padding)
    stride = _triple(stride)
    dilation = _triple(dilation)
    out_channels = weight.shape[0]

    if jt.flags.use_cuda and jt.cudnn:
        y = jt.cudnn.ops.cudnn_conv3d(x, weight, *stride, *padding, *dilation, groups)
    elif groups == 1:
        N,C,D,H,W = x.shape
        Kd, Kh, Kw = weight.shape[-3:]
        od = (D+padding[0]*2-Kd*dilation[0]+dilation[0]-1)//stride[0]+1
        oh = (H+padding[1]*2-Kh*dilation[1]+dilation[1]-1)//stride[1]+1
        ow = (W+padding[2]*2-Kw*dilation[2]+dilation[2]-1)//stride[2]+1
        xx = x.reindex([N,out_channels,C,od,oh,ow,Kd,Kh,Kw], [
                'i0', # Nid
                'i2', # Cid
                f'i3*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
                f'i4*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
                f'i5*{stride[2]}-{padding[2]}+i8*{dilation[2]}', # Did+KDid
            ])
        ww = weight.broadcast(xx.shape, [0,3,4,5])
        yy = xx*ww
        y = yy.sum([2,6,7,8]) # Kc, Kh, Kw,Kd
    else:
        N,C,D,H,W = x.shape
        Kd, Kh, Kw = weight.shape[-3:]
        G = groups
        CpG = C // G # channels per group
        oc = out_channels
        od = (D+padding[0]*2-Kd*dilation[0]+dilation[0]-1)//stride[0]+1
        oh = (H+padding[1]*2-Kh*dilation[1]+dilation[1]-1)//stride[1]+1
        ow = (W+padding[2]*2-Kw*dilation[2]+dilation[2]-1)//stride[2]+1
        xx = x.reindex([N,G,oc//G,CpG,od,oh,ow,Kd,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{stride[0]}-{padding[0]}+i7*{dilation[0]}', # Hid+Khid
                f'i5*{stride[1]}-{padding[1]}+i8*{dilation[1]}', # Wid+KWid
                f'i6*{stride[2]}-{padding[2]}+i9*{dilation[2]}', # Did+KDid
            ])
        xx.compile_options = {"G":G}
        # w: [oc, CpG, Kh, Kw, Kd]
        ww = weight.reindex([N, G, oc//G, CpG, oh, ow, od, Kh, Kw, Kd], [
                f'i1*{oc//G}+i2',
                'i3',
                'i7',
                'i8',
                'i9'
            ])
        yy = xx*ww
        y = yy.reindex_reduce('add', [N, oc, od, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5',
                'i6'
            ])

    if bias is not None:
        b = bias.broadcast(y.shape, [0,2,3,4])
        y = y + b
    return y

class ConvTranspose(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # added
        self.dilation = dilation
        self.groups = groups

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        # added
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.real_padding = (self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
            self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1])
        self.output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
        assert self.output_padding[0] < max(self.stride[0], self.dilation[0]) and \
            self.output_padding[1] < max(self.stride[1], self.dilation[1]), \
            "output padding must be smaller than max(stride, dilation)"
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        self.weight = init.invariant_uniform((in_channels, out_channels//groups) + self.kernel_size, dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        if self.groups == 1:
            N,C,H,W = x.shape
            i,o,h,w = self.weight.shape
            assert C==i
            stride_h, stride_w = self.stride
            padding_h, padding_w = self.padding
            dilation_h, dilation_w = self.dilation

            h_out = (H-1) * stride_h + self.output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
            w_out = (W-1) * stride_w + self.output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
            out_shape = (N, o, h_out, w_out)
            shape = (N, i, o, H, W, h, w)
            xx = x.broadcast(shape, (2, 5, 6)) # i,h,w
            ww = self.weight.broadcast(shape, (0, 3, 4)) # N,H,W
            y = (ww*xx).reindex_reduce("add", out_shape, [
                'i0', # N
                'i2', # o
                f'i3*{stride_h}-{padding_h}+i5*{dilation_h}', # Hid+Khid
                f'i4*{stride_w}-{padding_w}+i6*{dilation_w}', # Wid+KWid
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        else:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            i,o,h,w = self.weight.shape
            oc = self.out_channels
            G = self.groups
            CpG = C // G # channels per group
            assert C==self.in_channels
            stride_h, stride_w = self.stride
            padding_h, padding_w = self.padding
            dilation_h, dilation_w = self.dilation

            oh = (H-1) * stride_h + self.output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
            ow = (W-1) * stride_w + self.output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
            out_shape = (N, oc, oh, ow)
            shape = [N,G,oc//G,CpG,oh,ow,Kh,Kw]
            xx = x.reindex(shape, [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
            ww = self.weight.reindex(shape, [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
            ww.compile_options = xx.compile_options = {"G":G,"C":C}
            y = (ww*xx).reindex_reduce("add", out_shape, [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
ConvTranspose2d = ConvTranspose

class ConvTranspose3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # added
        self.dilation = dilation
        self.group = groups
        assert groups==1, "Group conv not supported yet."

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        # added
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.real_padding = (
            self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0],
            self.dilation[1] * (self.kernel_size[1] - 1) - self.padding[1],
            self.dilation[2] * (self.kernel_size[2] - 1) - self.padding[2])
        self.output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding, output_padding)
        assert self.output_padding[0] < max(self.stride[0], self.dilation[0]) and \
            self.output_padding[1] < max(self.stride[1], self.dilation[1]) and \
            self.output_padding[2] < max(self.stride[2], self.dilation[2]), \
            "output padding must be smaller than max(stride, dilation)"

        self.weight = init.invariant_uniform((in_channels, out_channels) + self.kernel_size, dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        return conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.group, self.dilation)

def conv_transpose(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    "\n对输入的数据进行转置卷积操作。对输入的每个channel进行转置卷积操作，也就是通过在输入的每个channel上作卷积，得到若干子矩阵，将这些子矩阵线性映射并相加后得到的结果。\n\n    参数:\n        - `input` (jittor.Var) : 输入张量，shape为(N, C, H, W)， N为batch size(每次处理的大小)，C为通道数，H和W为高和宽\n        - `weight` (jittor.Var) : 卷积核权重，shape为(i, o, h, w)， i为输入通道数，o为输出通道数，h和w分别为卷积核的高和宽\n        - `bias` (jittor.Var，可选) : 卷积后的偏置，除非为None，否则其shape应当为(o,)。默认值：None\n        - `stride` (int或tuple，可选) : 卷积的步长，单位为像素。默认值: 1\n        - `padding` (int或tuple, 可选) : 输入的四周添加的填充长度，单位为像素。默认值: 0\n        - `output_padding` (int或tuple, 可选) : 输出的四周添加的填充长度，表示在转置卷积操作结束后添加到输出的零填充的数额，单位为像素。默认值: 0\n        - `groups` (int，可选) : 输入通道和输出通道之间的阻塞连接数。默认值: 1\n        - `dilation` (int或tuple，可选) : 卷积核元素之间的间距。默认值: 1\n\n    返回值:\n        jittor.Var: 对输入的数据进行转置卷积操作后的结果，shape为(N, o, h_out, w_out)\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.ones(2,3,4,5)            \n        >>> w = jt.ones(3,6,3,3)             \n        >>> nn.conv_transpose(x,w).shape\n        [2,6,6,7,]\n\n"
    if groups == 1:
        x = input
        N,C,H,W = x.shape
        i,o,h,w = weight.shape
        assert C==i
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        # added
        padding = padding if isinstance(padding, tuple) else (padding, padding)
        output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
        assert output_padding[0] < max(stride[0], dilation[0]) and \
            output_padding[1] < max(stride[1], dilation[1]), \
            "output padding must be smaller than max(stride, dilation)"

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        h_out = (H-1) * stride_h + output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
        w_out = (W-1) * stride_w + output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
        out_shape = (N, o, h_out, w_out)
        shape = (N, i, o, H, W, h, w)
        xx = x.broadcast(shape, (2, 5, 6)) # i,h,w
        ww = weight.broadcast(shape, (0, 3, 4)) # N,H,W
        y = (ww*xx).reindex_reduce("add", out_shape, [
            'i0', # N
            'i2', # o
            f'i3*{stride_h}-{padding_h}+i5*{dilation_h}', # Hid+Khid
            f'i4*{stride_w}-{padding_w}+i6*{dilation_w}', # Wid+KWid
        ])
        if isinstance(bias, jt.Var):
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        else:
            assert not bias, "Bias should be none or jittor var"
        return y
    else:
        N,C,H,W = input.shape
        i,o,h,w = weight.shape
        G = groups
        oc = o * G
        CpG = C // G # channels per group
        assert C % G == 0
        assert C==i, (C, i)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        # added
        padding = padding if isinstance(padding, tuple) else (padding, padding)
        output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
        assert output_padding[0] < max(stride[0], dilation[0]) and \
            output_padding[1] < max(stride[1], dilation[1]), \
            "output padding must be smaller than max(stride, dilation)"

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        oh = (H-1) * stride_h + output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
        ow = (W-1) * stride_w + output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
        out_shape = (N, oc, oh, ow)
        shape = [N,G,oc//G,CpG,oh,ow,h,w]
        xx = input.reindex(shape, [
            'i0',
            f'i1*{oc//G}+i2',
            'i4',
            'i5'
        ])
        ww = weight.reindex(shape, [
            f'i1*{oc//G}+i2',
            'i3',
            'i6',
            'i7'
        ])
        ww.compile_options = xx.compile_options = {"G":G,"C":C}
        y = (ww*xx).reindex_reduce("add", out_shape, [
            'i0', # Nid
            f'i1*{CpG}+i3', # Gid
            f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', # Hid+Khid
            f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}', # Wid+KWid
        ])
        if bias is not None:
            b = bias.broadcast(y.shape, [0,2,3])
            y = y + b
        return y
conv_transpose2d = conv_transpose

def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    "\n对输入数据执行三维反卷积操作。此操作也称为转置卷积，它将卷积核应用于输入数据，以创建具有更大尺寸的输出。\n\n    参数:\n        - `input` (jittor.Var) : 形状为(N, C, D, H, W)的输入张量，其中N是批量大小，C是输入通道数，D是深度，H是高度，W是宽度。\n        - `weight` (jittor.Var) : 形状为(i, o, d, h, w)的卷积核权重张量，其中i是输入通道数，o是输出通道数，d，h和w分别是深度，高度和宽度。\n        - `bias` (jittor.Var，可选) : 形状为(o,)的偏置张量。如果没有指定，将不使用任何偏置。默认值：None\n        - `stride` (int或tuple，可选) : 卷积的步长，单位为像素。如果是tuple，则按照(d, h, w)的顺序设定步长。默认值: 1\n        - `padding` (int或tuple, 可选) : 输入的四周添加的填充长度，单位为像素。如果是tuple，则按照(d, h, w)的顺序设定填充。默认值: 0\n        - `output_padding` (int或tuple, 可选) : 输出的四周添加的填充长度，表示在转置卷积操作结束后添加到输出的零填充的数额，单位为像素。如果是tuple，则按照(d, h, w)的顺序设定输出填充。默认值: 0\n        - `groups` (int，可选) : 输入通道和输出通道之间的阻塞连接数。默认值: 1\n        - `dilation` (int或tuple，可选) : 卷积核元素之间的间距。如果是元组，则按照(d, h, w)的顺序设定间距。默认值: 1\n\n    返回值:\n        jittor.Var: 对输入的数据进行转置卷积操作后的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> input = jt.ones(1,1,3,3,3) \n        >>> weight = jt.ones(1,1,2,2,2) \n        >>> nn.conv_transpose3d(input, weight).shape\n        [1,1,4,4,4,]\n\n"
    x = input
    N,C,D,H,W = x.shape
    i,o,d,h,w = weight.shape
    assert C==i
    assert groups==1, "Group conv not supported yet."
    stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
    # added
    padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
    output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding, output_padding)
    assert output_padding[0] < max(stride[0], dilation[0]) and \
        output_padding[1] < max(stride[1], dilation[1]) and \
        output_padding[2] < max(stride[2], dilation[2]), \
        "output padding must be smaller than max(stride, dilation)"

    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = dilation

    d_out = (D-1) * stride_d + output_padding[0] - 2*padding_d + 1 + (d-1)*dilation_d
    h_out = (H-1) * stride_h + output_padding[1] - 2*padding_h + 1 + (h-1)*dilation_h
    w_out = (W-1) * stride_w + output_padding[2] - 2*padding_w + 1 + (w-1)*dilation_w
    out_shape = (N, o, d_out, h_out, w_out)
    if jt.flags.use_cuda and jt.cudnn:
        return jt.cudnn.ops.cudnn_conv3d_backward_x(weight, x, *out_shape[2:], *stride, *padding, *dilation, groups)
    shape = (N, i, o, D, H, W, d, h, w)
    xx = x.broadcast(shape, (2, 6, 7, 8)) # i,h,w
    ww = weight.broadcast(shape, (0, 3, 4, 5)) # N,H,W
    y = (ww*xx).reindex_reduce("add", out_shape, [
        'i0', # N
        'i2', # o
        f'i3*{stride_d}-{padding_d}+i6*{dilation_d}', # Did+Kdid
        f'i4*{stride_h}-{padding_h}+i7*{dilation_h}', # Hid+Khid
        f'i5*{stride_w}-{padding_w}+i8*{dilation_w}', # Wid+KWid
    ])
    if isinstance(bias, jt.Var):
        b = bias.broadcast(y.shape, [0,2,3,4])
        y = y + b
    else:
        assert not bias, "Bias should be none or jittor var"
    return y

conv_transpose2d = conv_transpose

def pad(x,padding, mode='constant', value=0):
    "\n对给定的输入张量进行填充。填充方法有四种方式: \n\n    - 常数填充（'constant'，在每个维度的两侧用固定值填充）\n    - 复制填充（'replicate'，在每个维度的两侧用输入张量的反射（无重复）进行填充）\n    - 反射填充（'reflect'，在每个维度的两侧用输入张量的边缘值进行填充）\n    - 环形填充（'circular'，在每个维度的两侧用输入张量的环形复制进行填充）。\n\n    参数:\n        - `x` (jittor.Var) : 输入的张量\n        - `padding` (list[int]) : 填充的尺寸，list长度必须为偶数，且必须小于等于输入张量的维度的两倍。偶数索引为左填充，奇数索引为右填充。\n        - `mode` (str, 可选) : 填充模式，有‘constant'，‘replicate'，‘reflect'和‘circular'四种选择，默认值: 'constant'\n        - `value` (int或float，可选) : 当填充模式为'constant'时，使用此值进行填充。默认值: 0\n\n    返回值:\n        jittor.Var: 填充后的张量\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.ones(3,3)\n        >>> nn.pad(x, [1,2]) \n        jt.Var([[0. 1. 1. 1. 0. 0.]\n                [0. 1. 1. 1. 0. 0.]\n                [0. 1. 1. 1. 0. 0.]], dtype=float32)\n\n"
    assert mode in ['constant','replicate','reflect','circular'],'only support constant,replicate,reflect,circular pad'
    assert len(padding)%2==0 and len(padding)//2<=x.ndim

    padding = list(padding)
    left = [0]*(x.ndim-len(padding)//2)+padding[::2][::-1]
    right = [0]*(x.ndim-len(padding)//2)+padding[1::2][::-1]

    out_dims = []
    out_shape = []
    for i,n,l,r in zip(range(x.ndim),x.shape,left,right):
        out_shape.append(n+l+r)
        if mode == 'constant':
            out_dims.append(f'i{i}-{l}')
        elif mode == 'replicate':
            out_dims.append(f"i{i}<{l} ? 0 : i{i} > {n+l-1} ? {n-1} : i{i}-{l}")
        elif mode == 'reflect':
            out_dims.append(f"i{i}<{l} ? {l}-i{i} : i{i} > {n+l-1} ? {2*(n-1)+l}-i{i} : i{i}-{l}")
        elif mode == 'circular':
            out_dims.append(f"i{i}<{l} ? {n-l}+i{i} : i{i} > {n+l-1} ? i{i}-{n+l} : i{i}-{l}")

    return x.reindex(out_shape,out_dims,overflow_value=value)


class ReflectionPad2d(Module):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ReflectionPad2d padding just support int or tuple, but found {type(padding)}")

    def execute(self, x):
        n,c,h,w = x.shape
        assert (self.pl < w and self.pr < w), f"padding_left and padding_right should be smaller than input width"
        assert (self.pt < h and self.pb < h), f"padding_top and padding_bottom should be smaller than input height"
        oh=h+self.pt+self.pb
        ow=w+self.pl+self.pr
        l = self.pl
        r = self.pl + w - 1
        t = self.pt
        b = self.pt + h - 1
        return x.reindex([n,c,oh,ow], ["i0","i1",
            f"i2<{t} ? {t}-i2 : i2 > {b} ? {h-1+b}-i2 : i2-{t}",
            f"i3<{l} ? {l}-i3 : i3 > {r} ? {w-1+r}-i3 : i3-{l}",
        ])

class ZeroPad2d(Module):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, (tuple,list)):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ZeroPad2d padding just support int or tuple, but found {type(padding)}")

    def execute(self, x):
        n,c,h,w = x.shape
        return x.reindex([n,c,h+self.pt+self.pb,w+self.pl+self.pr], ["i0","i1",f"i2-{self.pt}",f"i3-{self.pl}"])

class ConstantPad2d(Module):
    def __init__(self, padding, value):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ConstantPad2d padding just support int or tuple, but found {type(padding)}")
        self.value = value

    def execute(self, x):
        assert len(x.shape) >= 2
        shape = x.shape
        tar_shape = shape[0:-2] + [shape[-2]+self.pt+self.pb,shape[-1]+self.pl+self.pr]
        tar_dims = []
        for i in range(len(shape)-2):
            tar_dims.append(f"i{i}")
        tar_dims.append(f"i{i+1}-{self.pt}")
        tar_dims.append(f"i{i+2}-{self.pl}")
        return x.reindex(tar_shape, tar_dims, overflow_value=self.value)

class ReplicationPad2d(Module):
    def __init__(self, padding):
        self.padding = padding
        if isinstance(self.padding, int):
            self.pl = self.padding
            self.pr = self.padding
            self.pt = self.padding
            self.pb = self.padding
        elif isinstance(self.padding, tuple):
            self.pl, self.pr, self.pt, self.pb = self.padding
        else:
            raise TypeError(f"ReplicationPad2d padding just support int or tuple, but found {type(padding)}")

    def execute(self, x):
        n,c,h,w = x.shape
        oh=h+self.pt+self.pb
        ow=w+self.pl+self.pr
        l = self.pl
        r = self.pl + w - 1
        t = self.pt
        b = self.pt + h - 1
        return x.reindex([n,c,oh,ow], ["i0","i1",
            f"i2<{t} ? 0 : i2 > {b} ? {h-1} : i2-{t}",
            f"i3<{l} ? 0 : i3 > {r} ? {w-1} : i3-{l}"
        ])

class Embedding(Module):
    ''' A simple lookup table that stores embeddings of a fixed dictionary and size.

        :param num: size of the dictionary of embeddings
        :type num: int

        :param dim: the size of each embedding vector
        :type dim: int

        Example:
            >>> embedding = nn.Embedding(10, 3)
            >>> x = jt.int32([1, 2, 3, 3])
            >>> embedding(x)
            jt.Var([[ 1.1128596   0.19169547  0.706642]
             [ 1.2047412   1.9668795   0.9932192]
             [ 0.14941819  0.57047683 -1.3217674]
             [ 0.14941819  0.57047683 -1.3217674]], dtype=float32)
    '''
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, dtype="float32"):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = jt.init.gauss([self.num_embeddings, self.embedding_dim], dtype)
        if padding_idx is not None:
            self.weight[padding_idx] = 0

    def execute(self, x):
        res = self.weight[x]
        return res

def embedding(input, weight):
    "\n这个函数实现了编译嵌入。接收一个矩阵索引作为输入，并返回相应的权重。\n\n    参数:\n        - input (jittor.Var) : 索引张量，表示要查询权重的索引\n        - weight (jittor.Var) : 权重张量，表示需要查询的权重\n\n    返回值:\n        jittor.Var: 返回一个同权重数据类型的张量，尺寸为(input.shape[0], weight.shape[1])，表示索引所代表的权重。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> weight = jt.array([[0,0,3,1],[2,0,3,1],[0,0,0,0]])\n        >>> input = jt.array([0,1,2])  \n        >>> nn.embedding(input, weight) \n        jt.Var([[0 0 3 1]\n                [2 0 3 1]\n                [0 0 0 0]], dtype=int32)\n\n"
    return weight[input]

class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def execute(self, x):
        n,c,h,w = x.shape
        r = self.upscale_factor
        assert c%(r*r)==0, f"input channel needs to be divided by upscale_factor's square in PixelShuffle"
        return x.reindex([n,int(c/r**2),h*r,w*r], [
            "i0",
            f"i1*{r*r}+i2%{r}*{r}+i3%{r}",
            f"i2/{r}",
            f"i3/{r}"
        ])

class Tanh(Module):
    def __init__(self):
        super().__init__()
    def execute(self, x) :
        return x.tanh()

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def execute(self, x) :
        return x.sigmoid()

def softplus(x,beta=1.0,threshold=20.0):
    "\nSoftplus函数实现。Softplus函数是一种平滑函数，常用于构建深度神经网络的非线性变化层，对x的过大过小值进行了阈值剪裁以提高数值稳定性。具体的数学形式如下:\n\n    .. math::\n        y=\\left{\\{}\\begin{array}{11}\\frac{1}{\\beta} * log(1 + exp(\\beta * x)), & \\beta * x <= threshold\\\\\n        x - \\frac{threshold}{\\beta}, & \\beta * x > threshold\\end{array}\\right.\n\n    \n\n    参数:\n        - x (jittor.Var) : 输入张量\n        - beta (float, 可选) : 控制函数曲线的剧烈程度，默认值: 1.0\n        - threshold (float, 可选): 阈值，当beta * x大于threshold时，会对x进行截断，默认值: 20.0\n\n    返回值:\n        jittor.Var: x应用Softplus函数之后的结果张量，与输入x形状相同\n\n    代码示例：\n        >>> import jittor as jt\n        >>> x = jt.array([1.0, 2.0, 3.0])\n        >>> jt.nn.softplus(x)\n        jt.Var([1.3132616 2.126928  3.0485873], dtype=float32)\n\n"
    return 1 / beta * jt.log(1 + (beta * x).minimum(threshold).exp()) + \
        (x - threshold/beta).maximum(0.0)

def hardtanh(x,min_val=-1,max_val=1):
    return jt.clamp(x,min_v=min_val,max_v=max_val)


class Softplus(Module):
    r'''
    SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.
    
    Args:
        
        [in] beta (float): the beta value for the Softplus formulation. Default: 1.
        
        [in] threshold (float): values above this revert to a linear function. Default: 20.
    '''
    def __init__(self, beta=1, threshold=20):
        self.beta = beta
        self.threshold = threshold

    def execute(self, x):
        return softplus(x, self.beta, self.threshold)

class Resize(Module):
    def __init__(self, size, mode="nearest", align_corners=False):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
    def execute(self, x):
        return resize(x, self.size, self.mode, self.align_corners)


def _bicubic(x, a, func):
    "\n该函数执行双三次插值计算。依据给定的参数``x``和``a``以及插值函数索引``func``，返回相应的插值结果。\n\n    参数:\n        - x (float) : 插值位置的输入值\n        - a (float) : 插值函数的参数\n        - func (int): 一共有两种插值函数，func用于选择执行哪个插值函数，可能的值为1或2。如果func不是1或2，则返回0\n\n    返回值:\n        float: 双三次插值的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> jt.nn._bicubic(0.2, 0.5, 1) \n        jt.Var([0.88], dtype=float32)\n\n"
    # normal ver
    if func == 1:
        return (a+2)*(jt.abs(x)**3)-(a+3)*(x**2)+1
    if func == 2:
        return a*(jt.abs(x)**3)-5*a*(x**2)+8*a*(jt.abs(x))-4*a
    return 0


def _interpolate(img, x, y, ids, mode):
    "\n这个函数用于对图像进行插值。\n\n    参数:\n        - img (jittor.Var) : 输入图像张量\n        - x (jittor.Var) : 图像的x坐标列表\n        - y (jittor.Var) : 图像的y坐标列表\n        - ids (jittor.Var): 图像的id列表\n        - mode (str, 可选): 插值模式，有'nearest'，'bilinear' 和 'bicubic'三种选择，'nearest'表示最近邻插值，'bilinear'表示双线性插值，'bicubic'表示双三次插值。如果插值模式设置错误，会抛出ValueError。默认值: 'nearest'\n\n    返回值:\n        jittor.Var: 插值后的图像张量\n\n"
    if mode == "nearest":
        return img.reindex([*ids, x.floor_int(), y.floor_int()])
    if mode == "bilinear":
        fx, fy = x.floor_int(), y.floor_int()
        cx, cy = fx + 1, fy + 1
        dx, dy = x - fx, y - fy
        a = img.reindex_var([*ids, fx, fy])
        b = img.reindex_var([*ids, cx, fy])
        c = img.reindex_var([*ids, fx, cy])
        d = img.reindex_var([*ids, cx, cy])
        dnx, dny = 1 - dx, 1 - dy
        ab = dx * b + dnx * a
        cd = dx * d + dnx * c
        o = ab * dny + cd * dy
        return o
    if mode=="bicubic": # ugly ver.
        n,c,h,w = img.shape
        fx, fy = x.floor_int(), y.floor_int()
        dix, diy = x - fx, y - fy
        ax, ay = _bicubic(dix+1,-0.75,2), _bicubic(diy+1,-0.75,2)
        bx, by = _bicubic(dix,-0.75,1), _bicubic(diy,-0.75,1)
        cx, cy = _bicubic(1-dix,-0.75,1), _bicubic(1-diy,-0.75,1)
        dx, dy = _bicubic(2-dix,-0.75,2), _bicubic(2-diy,-0.75,2)
        afx, afy = jt.maximum(jt.minimum(fx-1,h-1),0), jt.maximum(jt.minimum(fy-1,w-1),0)
        bfx, bfy = jt.maximum(jt.minimum(fx,h-1),0), jt.maximum(jt.minimum(fy,w-1),0)
        cfx, cfy = jt.maximum(jt.minimum(fx+1,h-1),0), jt.maximum(jt.minimum(fy+1,w-1),0)
        dfx, dfy = jt.maximum(jt.minimum(fx+2,h-1),0), jt.maximum(jt.minimum(fy+2,w-1),0)
        a = ax*(img.reindex_var([*ids,afx,afy])*ay+img.reindex_var([*ids,afx,bfy])*by+img.reindex_var([*ids,afx,cfy])*cy+img.reindex_var([*ids,afx,dfy])*dy)
        b = bx*(img.reindex_var([*ids,bfx,afy])*ay+img.reindex_var([*ids,bfx,bfy])*by+img.reindex_var([*ids,bfx,cfy])*cy+img.reindex_var([*ids,bfx,dfy])*dy)
        c = cx*(img.reindex_var([*ids,cfx,afy])*ay+img.reindex_var([*ids,cfx,bfy])*by+img.reindex_var([*ids,cfx,cfy])*cy+img.reindex_var([*ids,cfx,dfy])*dy)
        d = dx*(img.reindex_var([*ids,dfx,afy])*ay+img.reindex_var([*ids,dfx,bfy])*by+img.reindex_var([*ids,dfx,cfy])*cy+img.reindex_var([*ids,dfx,dfy])*dy)
        o = a + b + c + d
        return o
    raise (f"Not support interpolation mode: {mode}")

# TODO: tf_mode to another function
def resize(img, size, mode="nearest", align_corners=False, tf_mode=False):
    n, c, h, w = img.shape
    H, W = size
    nid, cid, hid, wid = jt.index((n, c, H, W))
    if align_corners:
        x = hid * ((h - 1) / max(1, H - 1))
        y = wid * ((w - 1) / max(1, W - 1))
    elif mode == "bicubic":
        x = (hid + 0.5) * (h / H) - 0.5
        y = (wid + 0.5) * (w / W) - 0.5
    elif mode == 'nearest':
        x = hid * (h / H)
        y = wid * (w / W)
    elif mode == "area":
        '''
        Area interpolation uses AdaptivePool2D to resize origin images.
        '''
        stride = (h // H, w // W)
        assert stride[0] > 0 and stride[1] > 0
        x, y = jt.meshgrid(jt.arange(0, H, 1), jt.arange(0, W, 1))
        startH = jt.floor(x*h/H).int32()
        endH = jt.ceil((x+1)*h/H).int32()
        maxH = int(jt.max(endH - startH).data)
        startW = jt.floor(y*w/W).int32()
        endW = jt.ceil((y+1)*w/W).int32()
        maxW = int(jt.max(endW - startW).data)
        pixel_count = (endH - startH) * (endW - startW)
        adaptive_output = img.reindex([img.shape[0], img.shape[1], H, W, maxH, maxW], ["i0", "i1", "@e0(i2, i3) + i4", "@e2(i2, i3) + i5"], extras=[startH, endH, startW, endW], overflow_conditions=["i4 >= @e1(i2, i3) - @e0(i2, i3)", "i5 >= @e3(i2, i3) - @e2(i2, i3)"], overflow_value=0)
        adaptive_output = adaptive_output.reduce("sum", [4,5]) / pixel_count[None, None, ...]
        return adaptive_output
    else:
        if (tf_mode):
            x = hid * (h / H)
            if H > h: x = x.clamp(0, h - 1)
            y = wid * (w / W)
            if W > w: y = y.clamp(0, w - 1)
        else:
            x = hid * (h / H) + (h / H * 0.5 - 0.5)
            if H > h: x = x.clamp(0, h - 1)
            y = wid * (w / W) + (w / W * 0.5 - 0.5)
            if W > w: y = y.clamp(0, w - 1)
    return _interpolate(img, x, y, (nid, cid), mode)

upsample = resize


def interpolate(X, size=None, scale_factor=None, mode='bilinear', align_corners=False, tf_mode=False):
    if scale_factor is not None:
        size = [int(X.shape[-2] * scale_factor), int(X.shape[-1] * scale_factor)]
    if isinstance(size, int):
        size = (size, size)
    if scale_factor is not None and scale_factor > 1:
        return upsample(X, size, mode, align_corners, tf_mode)
    else:
        return resize(X, size, mode, align_corners, tf_mode)


def grid_sample_v0(input, grid, mode='bilinear', padding_mode='zeros'):
    r'''
    Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.

    grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. For example, values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.

    Args:

        [in] input (var): the source input var, whose shape is (N, C, Hi, Wi)

        [in] grid (var): the pixel locations, whose shape is (N, Ho, Wo, 2)

        [in] mode (string): the interpolate way, default: bilinear.

        [in] padding_mode (string): the padding way, default: zeros.

        [out] output (var): the output var, whose shape is (N, C, Ho, Wo)

    Example:

        >>> x = jt.array([[[[1,2],[3,4]]]])
        >>> print(x)
        [[[[1 2]
        [3 4]]]] 

        >>> grid = jt.array([[[[0.5, 0.5]]]])
        >>> print(x.shape, grid.shape)
        [1,1,2,2,], [1,1,2,2,]

        >>> nn.grid_sample(x, grid)
        [[[[3.25]]]]
    '''
    assert padding_mode == 'zeros'
    Ni, Ci, Hi, Wi = input.shape
    No, Ho, Wo, D = grid.shape
    assert D == 2
    assert Ni == No
    assert len(input.shape) == 4 and len(grid.shape)

    nid, cid, hid, wid = jt.index((Ni, Ci, Ho, Wo))
    x = ((grid[:, :, :, 1].unsqueeze(1).repeat([1, Ci, 1, 1]) + 1) / 2) * (Hi - 1)
    y = ((grid[:, :, :, 0].unsqueeze(1).repeat([1, Ci, 1, 1]) + 1) / 2) * (Wi - 1)
    return _interpolate(input, x, y, (nid, cid), mode)


def linspace_from_neg_one(grid,num_steps,align_corners):
    if  num_steps <= 1:
        return jt.array([],dtype=grid.dtype)
    # TODO: use jt.index
    ra = np.linspace(-1,1,num_steps)
    if not align_corners:
        ra = ra*(num_steps-1)/num_steps
    return jt.array(ra,dtype=grid.dtype)

def make_base_grid_4D(theta,N,C,H,W,align_corners):
    base_grid = jt.zeros((N, H, W, 3), dtype=theta.dtype)
    base_grid[...,0] = linspace_from_neg_one(theta, W, align_corners)
    base_grid[...,1] = jt.unsqueeze(linspace_from_neg_one(theta, H, align_corners),-1)
    base_grid[...,-1] = 1
    return base_grid

def make_base_grid_5D(theta,N,C,D,H,W,align_corners):
    base_grid = jt.zeros((N, D, H, W, 4), dtype=theta.dtype)
    base_grid[...,0] = linspace_from_neg_one(theta, W, align_corners)
    base_grid[...,1] = jt.unsqueeze(linspace_from_neg_one(theta, H, align_corners),-1)
    base_grid[...,2] = jt.unsqueeze(jt.unsqueeze(linspace_from_neg_one(theta, D, align_corners),-1),-1)
    base_grid[...,-1] = 1
    return base_grid

def affine_grid_generator_4D(theta,N,C,H,W,align_corners):
     base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners)
     grid = jt.nn.bmm(base_grid.reshape(N, H * W, 3),theta.transpose(0,2,1))
     return grid.reshape(N, H, W, 2)

def affine_grid_generator_5D(theta,N,C,D,H,W,align_corners):
    "用于生成一个五维的仿射网格。该函数首先通过对给定的仿射矩阵和基础网格进行矩阵乘法，然后通过变换生成新的仿射网格。\n\n参数:\n    theta (Var): 仿射变换矩阵，形状为 (N, 3, 4)\n    N (int): 输出网格的批次大小\n    C (int): 输出网格的通道数\n    D (int): 输出网格的深度\n    H (int): 输出网格的高度\n    W (int): 输出网格的宽度\n    align_corners (bool): 是否对齐角点\n\n返回:\n    生成的五维仿射网格 (Var)，大小为 (N, D, H, W, 3)\n\n示例:\n  .. code-block:: python\n\n    theta = jt.array([[[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0]]])\n    N = 1\n    C = 3\n    D = 4\n    H = 5\n    W = 5\n    align_corners = True\n    grid = affine_grid_generator_5D(theta, N, C, D, H, W, align_corners)\n    print(grid.shape)  # Output: (1, 4, 5, 5, 3)\n\n"
    base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners)
    grid = jt.nn.bmm(base_grid.reshape(N, D * H * W, 4),theta.transpose(0,2,1))
    return grid.reshape(N, D, H, W, 3)

def affine_grid(theta, size, align_corners=False):
    "根据给定的尺寸生成一个4D或5D的仿射网格。\n\n参数:\n    theta (Var): 仿射变换矩阵，对于4D网格，形状为 (N, 2, 3)；对于5D网格，形状为 (N, 3, 4)\n    size (list of int): 定义输出网格的尺寸，长度为4时表示4D网格，长度为5时表示5D网格\n    align_corners (bool): 是否对齐角点，默认为 False\n    \n代码示例:\n .. code-block:: python\n\n    import jittor as jt\n\n    # 4D网格示例\n    theta_4d = jt.array([[[1., 0, 0], [0, 1., 0]]])\n    size_4d = [1, 3, 5, 5]\n    grid_4d = affine_grid(theta_4d, size_4d)\n    print(grid_4d.shape)  # Output: (1, 5, 5, 2)\n\n    # 5D网格示例\n    theta_5d = jt.array([[[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0]]])\n    size_5d = [1, 3, 4, 5, 5]\n    grid_5d = affine_grid(theta_5d, size_5d)\n    print(grid_5d.shape)  # Output: (1, 4, 5, 5, 3)\n    \n返回值:\n    Var: 生成的仿射网格，形状为 (N, C, H, W, 2) 或 (N, C, D, H, W, 3)，具体取决于 `size` 的长度\n\n"
    assert str(theta.dtype) in ['float','float32','float64']
    assert min(size)>0
    assert len(size) in [4,5]    
    if len(size)== 4:
        assert theta.ndim == 3 and theta.shape[-2] == 2 and theta.shape[-1] == 3
        return affine_grid_generator_4D(theta, size[0], size[1], size[2], size[3], align_corners)
    elif len(size)==5:
        assert theta.ndim == 3 and theta.shape[-2] == 3 and theta.shape[-1] == 4
        return affine_grid_generator_5D(theta, size[0], size[1], size[2], size[3], size[4], align_corners)


def grid_sampler_unnormalize(coord,size,align_corners):
    "将归一化坐标转换为未归一化的坐标。该函数根据 `align_corners` 参数的值来决定坐标转换的方式。\n\n当 `align_corners` 为 True 时，转换公式为 `((coord + 1) / 2) * (size - 1)`。\n当 `align_corners` 为 False 时，转换公式为 `((coord + 1) * size - 1) / 2`。\n\n参数:\n    coord (Var): 归一化坐标，可以是任意形状的张量\n    size (int 或 list/tuple of int): 目标尺寸，可以是单个整数或整数列表/元组\n    align_corners (bool): 控制坐标转换方式的布尔值\n\n返回值:\n    Var: 转换后的未归一化坐标，形状与输入 `coord` 相同\n\n代码示例:\n.. code-block:: python\n\n    import jittor as jt\n\n    coord = jt.array([-1., 0, 1])\n    size = 100\n    align_corners = True\n    unnormalized_coord = grid_sampler_unnormalize(coord, size, align_corners)\n    print(unnormalized_coord)  # Output: [  0.  49.5  99. ]\n\n"
    if align_corners:
        #unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1) / 2) * (size - 1)
    else:
        #unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1) * size - 1) / 2


def clip_coordinates(x,clip_limit):
    "将输入坐标 `x` 裁剪到 `[0, clip_limit - 1]` 的范围内。\n\n参数:\n    x (Var): 需要裁剪的坐标\n    clip_limit (int): 裁剪的上限\n\n返回值:\n    裁剪后的坐标(Var):其形状与输入 `x` 相同。\n\n代码示例:\n    >>> import jittor as jt\n    >>> x = jt.array([-5, 0, 5, 10, 15])\n    >>> clip_limit = 10\n    >>> clipped_x = clip_coordinates(x, clip_limit)\n    >>> print(clipped_x)  \n    jt.Var([0 0 5 9 9], dtype=int32)\n\n"
    return jt.clamp(x,min_v=0,max_v=clip_limit-1)

def reflect_coordinates(x,twice_low,twice_high):
    "对输入坐标 `x` 进行反射操作。\n\n首先将输入`x`减去`twice_low / 2`，使其落在零和`twice_high - twice_low`的范围内，然后使用绝对值，取余数和取反操作来反射。\n .. math::\n\n    m = \\frac{twice\\_low}{2}\n   \n    span = \\frac{twice\\_high - twice\\_low}{2}\n    \n    flips = \\left\\lfloor \\frac{x - m}{span} \\right\\rfloor\n    \n    result1 = |(x - m) \\mod span| + m\n    \n    result2 = span - |(x - m) \\mod span| + m\n\n参数：\n    x (Var)：需要进行反射操作的坐标\n    twice_low (float)：反射区间两倍低点的值\n    twice_high (float)：反射区间两倍高点的值\n\n返回值：\n    反射后的坐标(Var):其形状与输入 `x` 相同。\n\n代码示例：\n  >>> import jittor as jt\n  >>> x = jt.array([1, 2, 3, 4, 5])\n  >>> twice_low = 2\n  >>> twice_high = 8\n  >>> reflected_x = reflect_coordinates(x, twice_low, twice_high)\n  >>> print(reflected_x)  \n  jt.Var([1. 2. 3. 4. 3.], dtype=float32)\n\n"
    if twice_low == twice_high:
        return jt.zeros_like(x)
    m = twice_low / 2
    span = (twice_high - twice_low) / 2
    x = (x - m).abs()
    #`fmod` returns same sign as `in`, which is positive after the `fabs` above.
    extra = x.mod(span)
    flips = (x / span).floor_int()
    result1 = extra+m
    result2 = span-extra+m
    con = flips%2==0
    not_con = flips%2!=0
    result1[not_con]=0.0
    result2[con]=0.0
    return result1+result2


def grid_sampler_compute_source_index(coord,size,padding_mode,align_corners):
    "计算网格采样器的源索引。\n\n首先将标准化坐标转化为原始坐标，然后进行裁剪或反射并裁剪以适应原始图像的边界。\n\n参数：\n    coord(Var)：归一化的坐标\n    size(int)：目标尺寸\n    padding_mode(str)：填充模式，'border' 或 'reflection'\n    align_corners(bool)：是否对齐角点\n\n返回值：\n    调整后的坐标(Var)，其形状与输入 `coord` 相同\n\n"
    coord = grid_sampler_unnormalize(coord, size, align_corners)
    if padding_mode == 'border':
        #clip coordinates to image borders
        coord = clip_coordinates(coord, size)
    elif padding_mode == 'reflection':
        #reflect coordinates by image borders
        if align_corners:
           coord = reflect_coordinates(coord, 0, 2*(size - 1))
        else:
           coord = reflect_coordinates(coord, -1, 2*size - 1)
        #clip coordinates to image borders
        coord = clip_coordinates(coord, size)
    return coord



def grid_sampler_3d(X,grid,mode,padding_mode,align_corners):
    "对三维数据张量 `X` 进行基于网格的采样。\n\n根据提供的 `grid`，以及指定的采样模式、填充模式和对齐角点选项，对 `X` 进行采样。\n\n参数：\n    X(Var)：输入的三维数据张量，形状为 [N, C, inp_D, inp_H, inp_W]\n    grid(Var)：采样网格，形状为 [N, D, H, W, 3]\n    mode(str)：采样模式，'nearest' 或 'bilinear'\n    padding_mode(str)：填充模式，'border' 或 'reflection'\n    align_corners(bool)：是否对齐角点\n\n返回值：\n    采样后的数据张量(Var)，形状为 [N, C, D, H, W]\n\n代码示例：\n >>> import jittor as jt\n >>> from jittor.nn import grid_sampler_3d\n >>> N, C, D, H, W = 1, 1, 2, 2, 2\n >>> X = jt.array([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]])\n >>> grid = jt.array([[[[[0, 0, 0], [1, 1, 1]], [[0, 1, 0], [1, 0, 1]]]]])\n >>> mode = 'bilinear'\n >>> padding_mode = 'border'\n >>> align_corners = True\n >>> sampled_X = grid_sampler_3d(X, grid, mode, padding_mode, align_corners)\n >>> print(sampled_X)\n jt.Var([[[[[0.16508093 0.25176302]\n        [0.12954207 0.27189574]]   \n        [[0.24352701 0.32800347]\n        [0.30734745 0.38710332]]]]], dtype=float32)\n >>> print(sampled_X) \n jt.Var([[[[[4.5 8. ]\n        [5.5 7. ]]]]], dtype=float32)\n >>> print(sampled_X.shape)\n [1, 1, 1, 2, 2]\n\n"
    N = X.shape[0]
    C = X.shape[1]
    inp_D = X.shape[2]
    inp_H = X.shape[3]
    inp_W = X.shape[4]

    D  = grid.shape[1]
    H = grid.shape[2]
    W = grid.shape[3]
    x = grid[:,:,:,:,0]
    y = grid[:,:,:,:,1]
    z = grid[:,:,:,:,2]
    shape = [N,C,D,H,W]
    cid = jt.index(shape, dim=1)
    nid = jt.index(shape, dim=0)

    x = grid_sampler_compute_source_index(x,inp_W,padding_mode,align_corners)
    y = grid_sampler_compute_source_index(y,inp_H,padding_mode,align_corners)
    z = grid_sampler_compute_source_index(z,inp_D,padding_mode,align_corners)
    xid = x.reindex(shape,['i0','i2','i3','i4'])
    yid = y.reindex(shape,['i0','i2','i3','i4'])
    zid = z.reindex(shape,['i0','i2','i3','i4'])

    if mode=='nearest':
        return X.reindex([nid,cid,zid.round_int(),yid.round_int(),xid.round_int()])
    elif mode=='bilinear':
        fx,fy,fz = xid.floor_int(),yid.floor_int(),zid.floor_int()
        cx,cy,cz = fx+1,fy+1,fz+1
        dx,dy,dz = xid-fx,yid-fy,zid-fz
        dnx,dny,dnz = cx-xid,cy-yid,cz-zid
        a = X.reindex([nid,cid,fz,fy,fx])
        b = X.reindex([nid,cid,cz,fy,fx])
        c = X.reindex([nid,cid,fz,cy,fx])
        d = X.reindex([nid,cid,fz,fy,cx])
        e = X.reindex([nid,cid,fz,cy,cx])
        f = X.reindex([nid,cid,cz,fy,cx])
        g = X.reindex([nid,cid,cz,cy,fx])
        h = X.reindex([nid,cid,cz,cy,cx])
        o = a*dnx*dny*dnz+b*dnx*dny*dz+c*dnx*dy*dnz+d*dx*dny*dnz+e*dx*dy*dnz+f*dx*dny*dz+g*dnx*dy*dz+h*dx*dy*dz
        return o

def grid_sampler_2d(X,grid,mode,padding_mode,align_corners):
    "对二维数据张量 `X` 进行基于网格的采样。\n\n根据提供的 `grid`，以及指定的采样模式、填充模式和对齐角点选项，对 `X` 进行采样。\n\n参数：\n    X(Var)：输入的二维数据张量，形状为 [N, C, inp_H, inp_W]。\n    grid(Var)：采样网格，形状为 [N, H, W, 2]。\n    mode(str)：采样模式，'nearest' 或 'bilinear'。\n    padding_mode(str)：填充模式，'border' 或 'reflection'。\n    align_corners(bool)：是否对齐角点。\n\n返回值：\n    采样后的数据张量(Var)，形状为 [N, C, H, W]。\n\n代码示例：\n >>> import jittor as jt\n >>> from jittor.nn import grid_sampler_2d\n >>> N, C, H, W = 1, 1, 2, 2\n >>> X = jt.array([[[[1, 2], [3, 4]]]])\n >>> grid = jt.array([[[[0, 0], [1, 1]], [[0, 1], [1, 0]]]])\n >>> mode = 'bilinear'\n >>> padding_mode = 'border'\n >>> align_corners = True\n >>> sampled_X = grid_sampler_2d(X, grid, mode, padding_mode, align_corners)\n >>> print(sampled_X)\n jt.Var([[[[[4.5 8. ]\n        [5.5 7. ]]]]], dtype=float32)\n >>> print(sampled_X.shape)\n [1, 1, 2, 2]\n\n"
    N = X.shape[0]
    C = X.shape[1]
    inp_H = X.shape[2]
    inp_W = X.shape[3]

    H  = grid.shape[1]
    W = grid.shape[2]
    x = grid[:,:,:,0]
    y = grid[:,:,:,1]
    shape = [N,C,H,W]
    cid = jt.index(shape, dim=1)
    nid = jt.index(shape, dim=0)

    x = grid_sampler_compute_source_index(x,inp_W,padding_mode,align_corners)
    y = grid_sampler_compute_source_index(y,inp_H,padding_mode,align_corners)
    xid = x.reindex(shape,['i0','i2','i3'])
    yid = y.reindex(shape,['i0','i2','i3'])

    if mode=='nearest':
        return X.reindex([nid,cid,yid.round_int(),xid.round_int()])
    elif mode=='bilinear':
        #xid,yid = (xid+0.00001),(yid+0.00001)
        fx,fy = (xid).floor_int(),(yid).floor_int()
        cx,cy = fx+1,fy+1
        dx,dy = xid-fx,yid-fy
        dnx,dny = cx-xid,cy-yid

        a = X.reindex([nid,cid,fy,fx],overflow_value=0.0)
        b = X.reindex([nid,cid,cy,fx],overflow_value=0.0)
        c = X.reindex([nid,cid,fy,cx],overflow_value=0.0)
        d = X.reindex([nid,cid,cy,cx],overflow_value=0.0)
        o = a*dnx*dny+b*dnx*dy+c*dx*dny+d*dx*dy
        return o


def grid_sampler(X, grid, mode, padding_mode, align_corners):
    "对数据张量 ``X`` 进行基于网格的采样。该函数根据输入张量的维度自动选择使用二维或三维采样。\n\n参数：\n    X(Var)：输入的数据张量，维度为4或5，形状为 [N, C, H, W]（二维）或 [N, C, D, H, W]（三维）\n    grid(Var)：采样网格，维度与 ``X`` 相同，形状为 [N, H, W, 2]（二维）或 [N, D, H, W, 3]（三维）\n    mode(str)：采样模式，'nearest' 或 'bilinear'。\n    padding_mode(str)：填充模式，'border' 或 'reflection'\n    align_corners(bool)：是否对齐角点。\n\n返回值：\n    采样后的数据张量(Var)，形状为 [N, C, H, W]（二维）或 [N, C, D, H, W]（三维）\n\n给定输入张量 ``X`` 和流场 ``grid``，使用 ``X`` 的值和 ``grid`` 中指定的像素位置计算输出。\n\n当前仅支持空间（二维）和体积（三维）的 ``X``。\n\n在空间（二维）情况下，对于形状为 ``(N, C, inp_H, inp_W)`` 的 ``X`` 和形状为 ``(N, H, W, 2)`` 的 ``grid``，输出将具有形状 ``(N, C, H, W)``\n\n对于每个输出位置 ``output[n, :, h, w]``，大小为 2 的向量 ``grid[n, h, w]`` 指定了 ``X`` 中的像素位置 ``x`` 和 ``y``，这些位置用于插值计算输出值 ``output[n, :, h, w]``。在 5D 输入的情况下，``grid[n, d, h, w]`` 指定了用于插值计算 ``output[n, :, d, h, w]`` 的 ``x``、``y``、``z`` 像素位置。``mode`` 参数指定了用于采样输入像素的 ````nearest```` 或 ````bilinear```` 插值方法。\n\n如果 ``grid`` 的值超出 ``[-1, 1]`` 范围，相应的输出将按照 ``padding_mode`` 定义的方式处理。选项包括：\n\n* ``padding_mode=\"zeros\"``：对于越界的网格位置使用 `0`，\n* ``padding_mode=\"border\"``：对于越界的网格位置使用边界值\n* ``padding_mode=\"reflection\"``：对于越界的网格位置使用边界反射的值。对于远离边界的位置，将继续反射直到变为界内\n\n代码示例：\n >>> import jittor as jt\n >>> from jittor.nn import grid_sampler\n >>> N, C, D, H, W = 1, 1, 2, 2, 2\n >>> X = jt.array([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]])\n >>> grid = jt.array([[[[[0, 0, 0], [1, 1, 1]], [[0, 1, 0], [1, 0, 1]]]]])\n >>> mode = 'bilinear'\n >>> padding_mode = 'border'\n >>> align_corners = True\n >>> sampled_X = grid_sampler(X, grid, mode, padding_mode, align_corners)\n >>> print(sampled_X)\n jt.Var([[[[[0.16508093 0.25176302]\n         [0.12954207 0.27189574]]  \n         [[0.24352701 0.32800347]\n         [0.30734745 0.38710332]]]]], dtype=float32)\n >>> print(sampled_X) \n jt.Var([[[[[4.5 8. ]\n        [5.5 7. ]]]]], dtype=float32)\n >>> print(sampled_X.shape)\n [1, 1, 1, 2, 2]\n\n >>> N, C, H, W = 1, 1, 2, 2\n >>> X = jt.array([[[[1, 2], [3, 4]]]])\n >>> grid = jt.array([[[[0, 0], [1, 1]], [[0, 1], [1, 0]]]])\n >>> mode = 'bilinear'\n >>> padding_mode = 'border'\n >>> align_corners = True\n >>> sampled_X = grid_sampler_2d(X, grid, mode, padding_mode, align_corners)\n >>> print(sampled_X)\n jt.Var([[[[[4.5 8. ]\n        [5.5 7. ]]]]], dtype=float32)\n >>> print(sampled_X.shape)\n [1, 1, 2, 2]\n\n"
    assert X.dtype==grid.dtype
    assert ((X.ndim==4 or X.ndim==5) and X.ndim==grid.ndim)
    assert X.shape[0]==grid.shape[0] and grid.shape[-1]==X.ndim-2
    assert X.numel()>0
    if X.ndim == 4:
        return grid_sampler_2d(X, grid, mode, padding_mode, align_corners)
    else:
        return grid_sampler_3d(X, grid, mode, padding_mode, align_corners)


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    assert mode in ['bilinear','nearest']
    assert padding_mode in ['zeros','border','reflection']
    return grid_sampler(input, grid, mode, padding_mode, align_corners)


class Upsample(Module):
    def __init__(self, scale_factor=None, mode='nearest'):
        self.scale_factor = scale_factor if isinstance(scale_factor, tuple) else (scale_factor, scale_factor)
        self.mode = mode
    
    def execute(self, x):
        return upsample(x,
            size=(
                int(x.shape[2]*self.scale_factor[0]), 
                int(x.shape[3]*self.scale_factor[1])),
            mode=self.mode)

class UpsamplingBilinear2d(Upsample):
    def __init__(self, scale_factor=None):
        Upsample.__init__(self, scale_factor, 'bilinear')

class UpsamplingNearest2d(Upsample):
    def __init__(self, scale_factor=None):
        Upsample.__init__(self, scale_factor, 'nearest')

class Sequential(Module):
    def __init__(self, *args):
        self.layers = collections.OrderedDict()
        for mod in args:
            if isinstance(mod, collections.OrderedDict):
                for k, m in mod.items():
                    self.add_module(k, m)
            elif isinstance(mod,list):
                for m in mod:
                    self.append(m)
            else:
                self.append(mod)
    def __getitem__(self, idx):
        if isinstance(idx, slice) or idx not in self.layers:
            return list(self.layers.values())[idx]

        return self.layers[idx]
    def __iter__(self):
        return self.layers.values().__iter__()
    def keys(self):
        return self.layers.keys()
    def values(self):
        return self.layers.values()
    def items(self):
        return self.layers.items()
    def execute(self, x):
        for k, layer in self.layers.items():
            x = layer(x)
        return x
    def dfs(self, parents, k, callback, callback_leave, recurse=True):
        n_children = len(self.layers)
        ret = callback(parents, k, self, n_children)
        if ret == False:
            return
        parents.append(self)
        if recurse:
            for k,v in self.layers.items():
                if isinstance(v, Module):
                    v.dfs(parents, k, callback, callback_leave)
        parents.pop()
        if callback_leave:
            callback_leave(parents, k, self, n_children)
    def append(self, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(len(self.layers))]=mod
    def add_module(self, name, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(name)]=mod

    def __len__(self):
        return len(self.layers)
    
    def named_children(self,):
        return list(self.layers.items())
    

    def __getattr__(self, key):
        if key in self.layers:
            return self.layers[key]
        return super().__getattr__(key)


class ParameterList(Module):
    def __init__(self, *args):
        self.params = collections.OrderedDict()
        for var in args:
            if isinstance(var, (collections.OrderedDict, dict)):
                for k, v in var.items():
                    self.add_param(k, v)
            elif isinstance(var, list):
                for v in var:
                    self.append(v)
            else:
                self.append(var)
    def __getitem__(self, idx):
        if idx not in self.params:
            return list(self.params.values())[idx]

        return self.params[idx]
    def __iter__(self):
        return self.params.values().__iter__()
    def keys(self):
        return self.params.keys()
    def values(self):
        return self.params.values()
    def items(self):
        return self.params.items()
    def execute(self, x):
        raise NotImplementedError("Parameters is not executable")
    def append(self, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        self.params[len(self.params)] = var
    def add_param(self, name, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        self.params[name]=var
    def __setitem__(self, name, var):
        self.add_param(name, var)

    def __len__(self):
        return len(self.params)

ParameterDict = ParameterList

def Parameter(data, requires_grad=True):
    ''' The `Parameter` interface isn't needed in Jittor, this interface
does nothings and it is just used for compatible.
    
A Jittor Var is a Parameter
when it is a member of Module, if you don't want a Jittor
Var menber is treated as a Parameter, just name it startswith
underscore `_`.
    '''
    LOG.w(Parameter.__doc__)
    data = data.clone()
    data.requires_grad = requires_grad
    return data

def backward(v, *args, **kw):
    ''' The `backward` variable interface doesn't exist in Jittor.
please use `optimizer.backward(loss)` or 
`optimizer.step(loss)` instead.
For example, if your code looks like this::

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

It can be changed to this::

    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()

Or more concise::

    optimizer.step(loss)

The step function will automatically zero grad and backward.
    '''
    LOG.f(backward.__doc__)

jt.Var.backward = backward

def unfold(X, kernel_size, dilation=1, padding=0, stride=1):
    assert X.ndim == 4
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    n, c, h, w = X.shape
    shape = X.shape
    area = kernel_size[0] * kernel_size[1]
    block_nums = []
    for i in range(2, 4):
        block_nums.append(
            (shape[i] + 2 * padding[i - 2] - dilation[i - 2] * (kernel_size[i - 2] - 1) - 1) // stride[i - 2] + 1)
    if padding[0] != 0 or padding[1] != 0:
        X = X.reindex([n, c, h + padding[0] * 2, w + padding[1] * 2],
                      ["i0", "i1", f"i2-{padding[0]}", f"i3-{padding[1]}"])
    output = X.reindex([n, c * area, block_nums[0] * block_nums[1]], ["i0", f"i1/{area}",
                                                                      f"i2/{block_nums[1]}*{stride[0]}+(i1%{area})/{kernel_size[1]}*{dilation[0]}",
                                                                      f"i2%{block_nums[1]}*{stride[1]}+(i1%{area})%{kernel_size[1]}*{dilation[1]}"])
    return output


def fold(X,output_size,kernel_size,dilation=1,padding=0,stride=1):
    assert X.ndim==3
    if not isinstance(kernel_size,tuple):
        kernel_size = (kernel_size,kernel_size)
    if not isinstance(dilation,tuple):
        dilation = (dilation,dilation)
    if not isinstance(padding,tuple):
        padding = (padding,padding)
    if not isinstance(stride,tuple):
        stride = (stride,stride)
    n,cl,num = X.shape
    area = kernel_size[0] * kernel_size[1]
    block_nums = []
    for i in range(2,4):
        block_nums.append((output_size[i-2]+2*padding[i-2]-dilation[i-2]*(kernel_size[i-2]-1)-1) // stride[i-2]+1)
    output = X.reindex_reduce("add",[n,cl // area,output_size[0]+2*padding[0],output_size[1]+2*padding[1]],["i0",f"i1/{area}",f"i2/{block_nums[1]}*{stride[0]}+(i1%{area})/{kernel_size[1]}*{dilation[0]}",f"i2%{block_nums[1]}*{stride[1]}+(i1%{area})%{kernel_size[1]}*{dilation[1]}"])
    return output[:,:,padding[0]:padding[0]+output_size[0],padding[1]:padding[1]+output_size[1]]

ModuleList = Sequential


class LSTMCell(jt.Module):
    ''' A long short-term memory (LSTM) cell.

    :param input_size: The number of expected features in the input
    :type input_size: int

    :param hidden_size: The number of features in the hidden state
    :type hidden_size: int

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    Example:

    >>> rnn = nn.LSTMCell(10, 20) # (input_size, hidden_size)
    >>> input = jt.randn(2, 3, 10) # (time_steps, batch, input_size)
    >>> hx = jt.randn(3, 20) # (batch, hidden_size)
    >>> cx = jt.randn(3, 20)
    >>> output = []
    >>> for i in range(input.shape[0]):
            hx, cx = rnn(input[i], (hx, cx))
            output.append(hx)
    >>> output = jt.stack(output, dim=0)
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = init.uniform((4 * hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = init.uniform((4 * hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = init.uniform((4 * hidden_size,), 'float32', -k, k)
            self.bias_hh = init.uniform((4 * hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            zeros = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)
            h, c = zeros, zeros
        else:
            h, c = hx

        y = matmul_transpose(input, self.weight_ih) + matmul_transpose(h, self.weight_hh)

        if self.bias:
            y = y + self.bias_ih + self.bias_hh
        
        i = y[:, :self.hidden_size].sigmoid()
        f = y[:, self.hidden_size : 2 * self.hidden_size].sigmoid()
        g = y[:, 2 * self.hidden_size : 3 * self.hidden_size].tanh()
        o = y[:, 3 * self.hidden_size:].sigmoid()

        c = f * c + i * g
        h = o * c.tanh()

        return h, c


class RNNCell(jt.Module):
    ''' An Elman RNN cell with tanh or ReLU non-linearity.

    :param input_size: The number of expected features in the input
    :type input_size: int

    :param hidden_size: The number of features in the hidden state
    :type hidden_size: int

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    :param nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'.
    :type nonlinearity: str, optional

    Example:

    >>> rnn = nn.RNNCell(10, 20)
    >>> input = jt.randn((6, 3, 10))
    >>> hx = jt.randn((3, 20))
    >>> output = []
    >>> for i in range(6):
            hx = rnn(input[i], hx)
            output.append(hx)
    '''
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity = "tanh"):
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = init.uniform((hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = init.uniform((hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = init.uniform((hidden_size,), 'float32', -k, k)
            self.bias_hh = init.uniform((hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            hx = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        y = matmul_transpose(input, self.weight_ih)+matmul_transpose(hx, self.weight_hh)

        if self.bias:
            y= y + self.bias_ih + self.bias_hh

        if self.nonlinearity == 'tanh':
            y = y.tanh()
        elif self.nonlinearity == 'relu':
            y = relu(y) 
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))

        return y


class GRUCell(jt.Module):
    ''' A gated recurrent unit (GRU) cell.

    :param input_size: The number of expected features in the input
    :type input_size: int

    :param hidden_size: The number of features in the hidden state
    :type hidden_size: int

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    Example:

    >>> rnn = nn.GRUCell(10, 20)
    >>> input = jt.randn((6, 3, 10))
    >>> hx = jt.randn((3, 20))
    >>> output = []
    >>> for i in range(6):
            hx = rnn(input[i], hx)
            output.append(hx)
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.bias = bias

        k = math.sqrt(1 / hidden_size)
        self.weight_ih = init.uniform((3*hidden_size, input_size), 'float32', -k, k)
        self.weight_hh = init.uniform((3*hidden_size, hidden_size), 'float32', -k, k)

        if bias:
            self.bias_ih = init.uniform((3*hidden_size,), 'float32', -k, k)
            self.bias_hh = init.uniform((3*hidden_size,), 'float32', -k, k)

    def execute(self, input, hx = None):
        if hx is None:
            hx = jt.zeros((input.shape[0], self.hidden_size), dtype=input.dtype)

        gi = matmul_transpose(input, self.weight_ih)
        gh = matmul_transpose(hx, self.weight_hh)

        if self.bias:
            gi += self.bias_ih
            gh += self.bias_hh
            
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        
        resetgate = jt.sigmoid(i_r + h_r)
        inputgate = jt.sigmoid(i_i + h_i)
        newgate = jt.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        return hy

class RNNBase(Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int, 
            num_layers: int = 1, bias: bool = True, batch_first: bool = False, 
            dropout: float = 0, bidirectional: bool = False, 
            proj_size: int = 0, nonlinearity: str = None) -> None:
        super().__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.nonlinearity = nonlinearity

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        num_directions = 1 + bidirectional
        k = math.sqrt(1 / hidden_size)

        def build_unit(name, in_channels, out_channels=None):
            if out_channels is not None:
                shape = (in_channels, out_channels)
            else:
                shape = (in_channels,)
            setattr(self, name, init.uniform(shape, 'float32', -k, k))
            if self.bidirectional:
                setattr(self, name + '_reverse', init.uniform(shape, 'float32', -k, k))

        for layer in range(num_layers):
            if layer == 0:
                build_unit(f'weight_ih_l{layer}', gate_size, input_size)
            else:
                if proj_size > 0:
                    build_unit(f'weight_ih_l{layer}', gate_size, num_directions * proj_size)
                else:
                    build_unit(f'weight_ih_l{layer}', gate_size, num_directions * hidden_size)

            if proj_size > 0:
                build_unit(f'weight_hh_l{layer}', gate_size, proj_size)
                build_unit(f'weight_hr_l{layer}', proj_size, hidden_size)
            else:
                build_unit(f'weight_hh_l{layer}', gate_size, hidden_size)

            if bias:
                build_unit(f'bias_ih_l{layer}', gate_size)
                build_unit(f'bias_hh_l{layer}', gate_size)

    def _cudnn_flatten_weights(self, cudnn_mode):
        def copy_to_flatten_weight(param_name, offset_idx, num_gates):
            def copy_to(param_name, offset_idx, idx):
                cur_offset = self._cudnn_weight_offset[offset_idx]
                param = getattr(self, param_name)
                param = param[self.hidden_size * idx: self.hidden_size * (idx + 1)]
                ft_weight[cur_offset:cur_offset + param.numel()] = param.flatten()
                
            if self.bias:
                for idx in range(num_gates):
                    copy_to('weight' + param_name, offset_idx + idx * 2, idx)
                    copy_to('bias' + param_name, offset_idx + idx * 2 + 1, idx)
                return num_gates * 2
            else:
                for idx in range(num_gates):
                    copy_to('weight' + param_name, offset_idx + idx, idx)
                return num_gates

        if jt.flags.use_cuda and jt.cudnn and jt.compiler.is_cuda:
            if getattr(self, '_cudnn_weight_size', None) is None:                
                offset_array = jt.cudnn.cudnn_rnn_weight_offset(
                    cudnn_mode,
                    self.input_size,
                    self.hidden_size, 
                    self.num_layers,
                    self.proj_size,
                    self.bias,
                    self.bidirectional
                )
                self._cudnn_weight_size = offset_array[0]
                self._cudnn_weight_offset = offset_array[1:]
            
            num_gates = {
                "RNN": 1, "LSTM": 4, "GRU": 3
            }[self.mode]
            ft_weight = jt.zeros(self._cudnn_weight_size, dtype=jt.float32)

            cnt = 0
            for layer in range(self.num_layers):
                suffix = ''
                cnt += copy_to_flatten_weight(f'_ih_l{layer}' + suffix, cnt, num_gates)
                cnt += copy_to_flatten_weight(f'_hh_l{layer}' + suffix, cnt, num_gates)
                if self.bidirectional:
                    suffix = '_reverse'
                    cnt += copy_to_flatten_weight(f'_ih_l{layer}' + suffix, cnt, num_gates)
                    cnt += copy_to_flatten_weight(f'_hh_l{layer}' + suffix, cnt, num_gates)
            return ft_weight
        else:
            raise RuntimeError("Not Cudnn found")

    @abstractmethod
    def call_rnn_cell(self, input, hidden, suffix):
        pass

    def call_rnn_sequence(self, input, hidden, suffix):
        if 'reverse' in suffix:
            input = input[::-1]

        output = []
        for s in range(input.shape[0]):
            out, hidden = self.call_rnn_cell(input[s], hidden, suffix)
            output.append(out)

        if 'reverse' in suffix:
            output = output[::-1]
        output = jt.stack(output, dim=0)

        return output, hidden

    def _execute_cudnn_rnn(self, input, hx):
        cudnn_mode = {
            ('RNN', 'tanh'): 'tanh',
            ('RNN', 'relu'): 'relu',
            ('LSTM', None): 'lstm',
            ('GRU', None): 'gru'
        }[(self.mode, self.nonlinearity)]
        ft_weight = self._cudnn_flatten_weights(cudnn_mode)

        if self.mode == 'LSTM':
            ret = jt.cudnn.ops.cudnn_rnn(input, hx[0], hx[1], ft_weight,
                cudnn_mode, self.input_size, self.hidden_size, self.num_layers, 0,
                self.dropout, self.bias, self.bidirectional, self.is_training()
            )
            return ret[0], (ret[1], ret[2])
        else:
            ret = jt.cudnn.ops.cudnn_rnn(input, hx, ft_weight,
                cudnn_mode, self.input_size, self.hidden_size, self.num_layers, 0,
                self.dropout, self.bias, self.bidirectional, self.is_training()
            )
            return ret[0], ret[1]

    def execute(self, input, hx=None):
        if self.batch_first:
            input = input.permute(1, 0, 2)

        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            if self.mode in ['RNN', 'GRU']:
                hx = jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype)
            elif self.mode == 'LSTM':
                hx = (jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype),
                      jt.zeros((num_directions * self.num_layers, input.shape[1], self.hidden_size), dtype=input.dtype))

        if jt.flags.use_cuda and jt.cudnn and self.proj_size == 0 and jt.compiler.is_cuda:
            return self._execute_cudnn_rnn(input, hx)
        else:
            hidden_n = []

            for l in range(self.num_layers):
                output = []

                if isinstance(hx, tuple):
                    hidden = [h[l * num_directions] for h in hx]
                else:
                    hidden = hx[l * num_directions]

                output, _hidden = self.call_rnn_sequence(input, hidden, f'l{l}')
                hidden_n.append(_hidden)

                if self.bidirectional:
                    if isinstance(hx, tuple):
                        hidden = [h[l * num_directions + 1] for h in hx]
                    else:
                        hidden = hx[l * num_directions + 1]

                    output_b, _hidden = self.call_rnn_sequence(input, hidden, f'l{l}_reverse')
                    output = jt.concat([output, output_b], dim=-1)
                    hidden_n.append(_hidden)

                if self.dropout > 0:
                    input = dropout(output, p=self.dropout)
                else:
                    input = output

            if isinstance(hx, tuple):
                hidden_n = tuple(jt.stack(hn, dim=0) for hn in zip(*hidden_n))
            else:
                hidden_n = jt.stack(hidden_n, dim=0)

            return output, hidden_n


class RNN(RNNBase):
    ''' Applies a multi-layer Elman RNN with tanh ReLU non-linearity to an input sequence.

    :param input_size: The number of expected features in the input.
    :type input_size: int

    :param hidden_size: The number of features in the hidden state.
    :type hidden_size: int

    :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
    :type num_layers: int, optinal

    :param nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
    :type nonlinearity: str, optional

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
    :type bias: bool, optional

    :param dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
    :type dropout: float, optional

    :param bidirectional: If True, becomes a bidirectional RNN. Default: False
    :type bidirectional: bool, optional    

    Example:
        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = jt.randn(5, 3, 10)
        >>> h0 = jt.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    '''
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
        nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False, 
        dropout: float = 0, bidirectional: bool = False) -> None:
        super().__init__('RNN', input_size, hidden_size, num_layers=num_layers, 
            bias=bias, batch_first=batch_first, dropout=dropout, 
            bidirectional=bidirectional)

        if not nonlinearity in ['tanh', 'relu']:
            raise ValueError('Unrecognized nonlinearity: ' + nonlinearity)
        self.nonlinearity = nonlinearity

    def call_rnn_cell(self, input, hidden, suffix):
        y = matmul_transpose(input, getattr(self, f'weight_ih_{suffix}')) 
        y = y + matmul_transpose(hidden, getattr(self, f'weight_hh_{suffix}'))
        
        if self.bias:
            y = y + getattr(self, f'bias_ih_{suffix}') + getattr(self, f'bias_hh_{suffix}')

        if self.nonlinearity == 'tanh':
            h = jt.tanh(y)
        else:
            h = jt.nn.relu(y)

        return h, h


class LSTM(RNNBase):
    ''' Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

    :param input_size: The number of expected features in the input.
    :type input_size: int

    :param hidden_size: The number of features in the hidden state.
    :type hidden_size: int

    :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
    :type num_layers: int, optinal

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
    :type bias: bool, optional

    :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
    :type dropout: float, optional

    :param bidirectional: If True, becomes a bidirectional LSTM. Default: False
    :type bidirectional: bool, optional

    :param proj_size: If > 0, will use LSTM with projections of corresponding size. Default: 0
    :type proj_size: int, optional

    Example:
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = jt.randn(5, 3, 10)
        >>> h0 = jt.randn(2, 3, 20)
        >>> c0 = jt.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    '''

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
            batch_first=False, dropout=0, bidirectional=False, proj_size=0):
        super().__init__('LSTM', input_size, hidden_size, num_layers=num_layers, 
            bias=bias, batch_first=batch_first, dropout=dropout, 
            bidirectional=bidirectional, proj_size=proj_size)

    def call_rnn_cell(self, input, hidden, suffix):
        h, c = hidden
        y = matmul_transpose(input, getattr(self, f'weight_ih_{suffix}')) 
        y = y + matmul_transpose(h, getattr(self, f'weight_hh_{suffix}'))
        
        if self.bias:
            y = y + getattr(self, f'bias_ih_{suffix}') + getattr(self, f'bias_hh_{suffix}')

        i = y[:, :self.hidden_size].sigmoid()
        f = y[:, self.hidden_size : 2 * self.hidden_size].sigmoid()
        g = y[:, 2 * self.hidden_size : 3 * self.hidden_size].tanh()
        o = y[:, 3 * self.hidden_size:].sigmoid()
        c = f * c + i * g
        h = o * c.tanh()

        if self.proj_size > 0:
            h = matmul_transpose(h, getattr(self, f'weight_hr_{suffix}'))

        return h, (h, c)


class GRU(RNNBase):
    ''' Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

    :param input_size: The number of expected features in the input.
    :type input_size: int

    :param hidden_size: The number of features in the hidden state.
    :type hidden_size: int

    :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
    :type num_layers: int, optinal

    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
    :type bias: bool, optional

    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
    :type bias: bool, optional

    :param dropout: If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
    :type dropout: float, optional

    :param bidirectional: If True, becomes a bidirectional GRU. Default: False
    :type bidirectional: bool, optional

    Example:
        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = jt.randn(5, 3, 10)
        >>> h0 = jt.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    '''

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
        bias: bool = True, batch_first: bool = False, dropout: float = 0, 
        bidirectional: bool = False) -> None:
        super().__init__('GRU', input_size, hidden_size, num_layers=num_layers, 
            bias=bias, batch_first=batch_first, dropout=dropout, 
            bidirectional=bidirectional)

    def call_rnn_cell(self, input, hidden, suffix):
        ih = matmul_transpose(input, getattr(self, f'weight_ih_{suffix}')) 
        hh = matmul_transpose(hidden, getattr(self, f'weight_hh_{suffix}'))
        
        if self.bias:
            ih = ih + getattr(self, f'bias_ih_{suffix}')
            hh = hh + getattr(self, f'bias_hh_{suffix}')

        hs = self.hidden_size
        r = (ih[:, :hs] + hh[:, :hs]).sigmoid()
        z = (ih[:, hs: 2 * hs] + hh[:, hs: 2 * hs]).sigmoid()
        n = (ih[:, 2 * hs:] + r * hh[:, 2 * hs:]).tanh()
        h = (1 - z) * n + z * hidden

        return h, h

def bilinear(in1, in2, weight, bias):
    "\n该函数对输入的in1, in2, weight和bias参数进行双线性（bilinear）运算。在数学上，双线性运算可以表示为：\n\n    .. math::\n        z = x * y + bias\n\n    其中，\n        - :math:`x` 是第一个输入矩阵和权重矩阵的乘积，形状为 [weight.shape[0], weight.shape[2]]\n        - :math:`y` 是扩展到与 :math:`x` 相同形状的第二个输入矩阵\n        - :math:`*` 表示矩阵乘法\n        - :math:`bias` 是偏置值\n\n    参数:\n        - in1（jittor.Var）：第一个输入张量\n        - in1（jittor.Var）：第二个输入张量\n        - weight（jittor.Var）：用于完成双线性运算的权重张量\n        -  bias（jt.Var，可选）：用于完成双线性运算的偏置值。如果该参数为None，则不使用偏置值。默认值: None\n\n    返回值:\n        jittor.Var: 完成双线性运算后的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> batch_size = 10\n        >>> feature_size = 5\n        >>> out_features = 7\n        >>> in1 = jt.randn(batch_size, feature_size)\n        >>> in2 = jt.randn(batch_size, feature_size)  \n        >>> weight = jt.randn(out_features,  feature_size, feature_size)\n        >>> bias = jt.randn(out_features)\n        >>> jt.nn.bilinear(in1, in2, weight, bias)\n\n"
    w = weight.transpose((1,0,2))
    w = w.reshape((w.shape[0], -1))
    x = jt.matmul(in1, w)
    x = x.reshape(x.shape[:-1]+[weight.shape[0], weight.shape[2]])
    y = in2.broadcast(x, (-2,))
    z = (x*y).sum(-1)
    if bias is not None:
        z += bias
    return z


class Bilinear(Module):
    ''' bilinear transformation $out = in1^T W in2 + bias$, Example::

    m = nn.Bilinear(20, 30, 40)
    input1 = jt.randn(128, 20)
    input2 = jt.randn(128, 30)
    output = m(input1, input2)
    print(output.shape)
    # [128, 40]

    '''
    def __init__(self, in1_features, in2_features, out_features, bias=True, dtype="float32"):
        bound = 1 / math.sqrt(in1_features)
        self.weight = jt.init.uniform([out_features, in1_features, in2_features], dtype, -bound, bound)
        self.bias = bias
        if bias:
            self.bias = jt.init.uniform([out_features], dtype, -bound, bound)

    def execute(self, in1, in2):
        return bilinear(in1, in2, self.weight, self.bias)

#TODO: support FFT2D only now.
def _fft2(x, inverse=False):
    "\n在Jittor中执行2维FFT操作。此函数仅在启用CUDA的条件下可用。输入 `x` 的形状必须为4维，且最后一维的大小为2，通常用于表示复数的实部和虚部。\n\n    数学公式:对于输入 `x` ，如果 `inverse` 为 ``False`` ，该函数执行以下FFT操作:\n        .. math::\n            Y[p, q] = \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1} x[m, n] \\exp\\left(-j\\frac{2 \\pi}{M N} (m p + n q)\\right)\n\n    如果 `inverse` 为 ``True`` ，则执行以下逆FFT操作:\n        .. math::\n            x[m, n] = \\frac{1}{M N} \\sum_{p=0}^{M-1} \\sum_{q=0}^{N-1} Y[p, q] \\exp\\left(j\\frac{2 \\pi}{M N} (m p + n q)\\right)\n\n    参数:\n        - x（jittor.Var）：输入的张量，要求为4维，且最后一维大小为2，通常用于表示复数的实部和虚部\n        - inverse（bool, 可选）：是否执行逆FFT操作。默认为 ``False`` ，表示执行正向FFT。如果为 ``True`` 则执行反向FFT。默认值: ``False``\n\n    返回值:\n        jittor.Var: 与输入 `x` 相同形状的复数张量，表示FFT结果。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> x = jt.random([2,3,4,2])\n        >>> y = jt.nn._fft2(x)\n        >>> y.shape\n        [2,3,4,2,]\n\n"
    assert(jt.flags.use_cuda==1)
    assert(len(x.shape) == 4)
    assert(x.shape[3] == 2)
    y = jt.compile_extern.cufft_ops.cufft_fft(x, inverse)
    if inverse:
        y /= x.shape[1] * x.shape[2]
    return y

class ComplexNumber:
    ''' Applys Complex number class.

        It's saved as jt.stack(real, imag, dim=-1)

        You can construct ComplexNumber with real part and imaginary part like ComplexNumber(real, imag)
        or real part only with ComplexNumber(real)
        or value after jt.stack with ComplexNumber(value, is_concat_value=True)

        add, sub, mul and truediv between ComplexNumber and ComplexNumber, jt.Var, int, float are implemented

        You can use 'shape', 'reshape' etc. as jt.Var

    Example:
        >>> real = jt.array([[[1., -2., 3.]]])
        >>> imag = jt.array([[[0., 1., 6.]]])
        >>> a = ComplexNumber(real, imag)
        >>> a + a
        >>> a / a
        >>> a.norm()                # sqrt(real^2+imag^2)
        >>> a.exp()                 # e^real(cos(imag)+isin(imag))
        >>> a.conj()                # ComplexNumber(real, -imag)
        >>> a.fft2()                # cuda only now. len(real.shape) equals 3
        >>> a.ifft2()               # cuda only now. len(real.shape) equals 3

        >>> a = jt.array([[1,1],[1,-1]])
        >>> b = jt.array([[0,-1],[1,0]])
        >>> c = ComplexNumber(a,b) / jt.sqrt(3)
        >>> c @ c.transpose().conj()
        ComplexNumber(real=jt.Var([[0.99999994 0.        ]
                [0.         0.99999994]], dtype=float32), imag=jt.Var([[0. 0.]
                [0. 0.]], dtype=float32))
    '''
    def __init__(self, real: jt.Var, imag: jt.Var=None, is_concat_value=False):
        if is_concat_value:
            assert real.shape[-1] == 2
            assert imag is None
            self.value = real
        elif imag is None:
            self.value = jt.stack([real, jt.zeros_like(real)], dim=-1)
        else:
            assert real.shape == imag.shape
            assert real.dtype == imag.dtype
            self.value = jt.stack([real, imag], dim=-1)

    @property
    def real(self):
        return self.value[..., 0]

    @property
    def imag(self):
        return self.value[..., 1]

    @property
    def shape(self):
        return self.value.shape[:-1]

    def norm(self):
        return jt.sqrt(jt.sqr(self.real) + jt.sqr(self.imag))

    def stop_grad(self):
        return ComplexNumber(self.value.stop_grad(), is_concat_value=True)

    def start_grad(self):
        return ComplexNumber(self.value.start_grad(), is_concat_value=True)
    
    def detach(self):
        return ComplexNumber(self.value.detach(), is_concat_value=True)

    def unsqueeze(self, dim=0):
        return ComplexNumber(jt.unsqueeze(self.real, dim=dim), jt.unsqueeze(self.imag, dim=dim))

    def squeeze(self, dim=0):
        return ComplexNumber(jt.squeeze(self.real, dim=dim), jt.squeeze(self.imag, dim=dim))

    def reshape(self, shape):
        return ComplexNumber(jt.reshape(self.real, shape), jt.reshape(self.imag, shape))
    
    def permute(self, *axes):
        return ComplexNumber(jt.permute(self.real, *axes), jt.permute(self.imag, *axes))

    def transpose(self, *axes):
        return ComplexNumber(jt.transpose(self.real, *axes), jt.transpose(self.imag, *axes))

    def exp(self):
        er = jt.exp(self.real)
        return ComplexNumber(er * jt.cos(self.imag), er * jt.sin(self.imag))

    def conj(self):
        return ComplexNumber(self.real, -self.imag)

    def __add__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        elif isinstance(other, (jt.Var, int, float)):
            return ComplexNumber(self.real + other, self.imag)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real + self.real, other.imag + self.imag)
        elif isinstance(other, (jt.Var, int, float)):
            return ComplexNumber(other + self.real, self.imag)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        elif isinstance(other, (jt.Var, int, float)):
            return ComplexNumber(self.real - other, self.imag)
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real - self.real, other.imag - self.imag)
        elif isinstance(other, (jt.Var, int, float)):
            return ComplexNumber(other - self.real, self.imag)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real * other.real - self.imag * other.imag,
                                 self.real * other.imag + self.imag * other.real)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.value * other, is_concat_value=True)
        elif isinstance(other, jt.Var):
            return ComplexNumber(self.real * other, self.imag * other)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real * self.real - other.imag * self.imag,
                                 other.imag * self.real + other.real * self.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(other * self.value, is_concat_value=True)
        elif isinstance(other, jt.Var):
            return ComplexNumber(other * self.real, other * self.imag)
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, ComplexNumber):
            norm = jt.sqr(other.real) + jt.sqr(other.imag)
            return ComplexNumber((self.real * other.real + self.imag * other.imag) / norm,
                                 (self.imag * other.real - self.real * other.imag) / norm)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.value / other, is_concat_value=True)
        elif isinstance(other, jt.Var):
            return ComplexNumber(self.real / other, self.imag / other)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        norm = jt.sqr(self.real) + jt.sqr(self.imag)
        if isinstance(other, ComplexNumber):
            return ComplexNumber((other.real * self.real + other.imag * self.imag) / norm,
                                 (other.imag * self.real - other.real * self.imag) / norm)
        elif isinstance(other, (int, float, jt.Var)):
            return ComplexNumber(other * self.real / norm, - other * self.imag / norm)
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real @ other.real - self.imag @ other.imag,
                                 self.real @ other.imag + self.imag @ other.real)
        elif isinstance(other, jt.Var):
            return ComplexNumber(self.real @ other, self.imag @ other)
        else:
            raise NotImplementedError

    def __imatmul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real @ self.real - other.imag @ self.imag,
                                 other.imag @ self.real + other.real @ self.imag)
        elif isinstance(other, jt.Var):
            return ComplexNumber(other @ self.real, other @ self.imag)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return f'ComplexNumber(real={self.real.__repr__()}, imag={self.imag.__repr__()})'
    
    def fft2(self):
        return ComplexNumber(_fft2(self.value, inverse=False), is_concat_value=True)

    def ifft2(self):
        return ComplexNumber(_fft2(self.value, inverse=True), is_concat_value=True)


def one_hot(x: jt.Var, num_classes: int=-1) -> jt.Var:
    "\n返回输入的 one-hot 编码。如果 x 中的值大于 num_class 或小于0，返回的 one-hot 将全为零。\n\n    参数:\n        - x（jittor.Var）：输入的张量，元素类型为bool或者int\n        - num_classes（bool, 可选）：类总数。如果设为-1，那么类的数量将被推断为输入张量中最大的类值加一。默认值: -1\n\n    返回值:\n        jittor.Var: 其形状由输入维度加一得到。在最后一个维度的索引处该 Var 的值为1，其他处为0。\n\n    代码示例：\n        >>> jt.nn.one_hot(jt.arange(5) % 3)\n        jt.Var([[1 0 0]\n            [0 1 0]\n            [0 0 1]\n            [1 0 0]\n            [0 1 0]], dtype=int32)\n        >>> jt.nn.one_hot(jt.arange(5) % 3, num_classes=5)\n        jt.Var([[1 0 0 0 0]\n            [0 1 0 0 0]\n            [0 0 1 0 0]\n            [1 0 0 0 0]\n            [0 1 0 0 0]], dtype=int32)\n        >>> jt.nn.one_hot(jt.arange(6).reshape(3,2) % 3)\n        jt.Var([[[1 0 0]\n            [0 1 0]]\n            [[0 0 1]\n            [1 0 0]]\n            [[0 1 0]\n            [0 0 1]]], dtype=int32)\n\n"

    assert x.dtype in [jt.bool, jt.int8, jt.int16, jt.int32, jt.int64, jt.uint8, jt.uint16, jt.uint32, jt.uint64]
    if num_classes == -1:
        num_classes = x.max().item() + 1

    N = len(x.shape)
    indices = ["i"+str(i) for i in range(N)]
    y = jt.ones_like(x).reindex(
        x.shape + [num_classes],
        indices, 
        extras=[x],
        overflow_conditions=[f"i{N} != @e0({','.join(indices)})"],
        overflow_value=0)
    return y


class KLDivLoss(Module):
    ''' Computes the Kullback-Leibler divergence loss.
    '''

    def __init__(self, reduction: str = 'mean', log_target: bool = False):
        '''
            :param reduction: Specifies the reduction to apply to the output. Can be 'mean', 'sum', 'batchmean', or 'none'. Defaults to 'mean'.
            :type reduction: str, optional
            :param log_target: Specifies whether target is the log space. Defaults to False.
            :type log_target: bool, optional
        '''
        self.reduction = reduction
        self.log_target = log_target

    def execute(self, input: jt.Var, target: jt.Var) -> jt.Var:
        if not self.log_target:
            loss_pointwise = target * (target.log() - input)
        else:
            loss_pointwise = target.exp() * (target - input)

        if self.reduction == "mean":
            loss = loss_pointwise.mean()
        elif self.reduction == "batchmean":
            loss = loss_pointwise.sum() / input.size(0)
        elif self.reduction == "sum":
            loss = loss_pointwise.sum()
        else:
            loss = loss_pointwise
        return loss

class Mish(Module):
    def __init__(self, inplace=False):
        '''
Applies the Mish function, element-wise.
reference: Mish - A Self Regularized Non-Monotonic Neural Activation Function.
        '''
        pass
    def execute(self, x):
        return x * jt.tanh(jt.softplus(x))

def mish(x, inplace=False):
    "\nmish函数是一个在神经网络中使用的激活函数，其定义为 :math:`x * tanh(softplus(x))` 。其中 :math:`softplus` 函数的定义为 :math:`softplus(x) = log(1 + e^x)`\n\n    参数:\n        - x（jittor.Var）：输入的张量\n        - inplace（bool, 可选）：用来决定是否在原位进行运算。当inplace=True时，函数会将结果直接保存在输入张量x上，这将节省存储空间但可能会改变输入的值。 默认值是False。当inplace=False时，函数将会创建一个新的张量来保存运算结果，并且不会影响输入张量x的值。\n\n    返回值:\n        jittor.Var: 一个与输入同形状、同数据类型的张量，代表运算的结果\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import nn\n        >>> x = jt.array([1, 2, 3, 4, 5])\n        >>> y = jt.nn.functional.mish(x, inplace=False)\n        >>> y = jt.nn.mish(x, inplace=False)            \n        >>> y\n        jt.Var([0.86509836 1.9439589  2.986535   3.997413   4.9995522 ], dtype=float32)\n\n"
    return x * jt.tanh(jt.nn.softplus(x))

def skip_init(module_cls, *args, **kw):
    "\n跳过初始化模块类的函数。这个函数主要用来直接返回需要实例化的模块类，而不对其进行初始化，可用于在实例化模块类后进行特殊的初始化设置。\n\n    参数:\n        - module_cls（jittor.Var）：需要实例化的模块类\n        - \\*args（tuple）：传递给模块类实例化的非关键字参数\n        - \\*\\*kw （dict）：传递给模块类实例化的关键字参数\n\n    返回值:\n        jittor.Var: 模块类的实例\n\n    代码示例：\n        >>> nn.skip_init(nn.Linear, 20, 30)\n        # 等同于：\n        >>> nn.Linear(20, 30)\n\n"
    return module_cls(*args, **kw)
