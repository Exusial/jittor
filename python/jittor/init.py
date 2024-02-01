# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import NanoVector, Var
import numpy as np
import math
import warnings

def eye(shape, dtype="float32"):
    "\n创建一个形状为 ``shape`` 的单位矩阵。\n\n    单位矩阵的定义为：\n    对于任何i, j (0 <= i, j < var的维度)，当i等于j时，输出为1；否则为0。\n\n    .. math::\n        eye(i, j) =\n        \\begin{cases}\n        1, & \\text{如果} i = j \\\\\n        0, & \\text{否则}\n        \\end{cases}\n\n\n    参数：\n        - shape (int or int...): 规定返回矩阵的形状，如果 ``shape`` 为整数， 那么返回的矩阵的形状为 ``(shape, shape)``。\n        - dtype (var.dtype, optional): 数据类型，默认为 ``float32``\n        \n    代码示例：\n        >>> init.eye(2)\n        jt.Var([[1. 0.]\n                [0. 1.]], dtype=float32)\n        >>> init.eye((2, 3), dtype='int32')\n        jt.Var([[1 0 0]\n                [0 1 0]], dtype=int32)\n\n    返回值：\n        返回一个单位矩阵\n    \n    注意事项：\n        - ``shape`` 如果是整数序列，只能传入两个维度的长度，否则会引起异常\n\n\n"
    if isinstance(shape, int):
        shape = (shape,shape)
    assert len(shape)==2, f"len of shape should be 2, but got {shape}"
    index = jt.index(shape)
    return (index[0]==index[1]).unary(dtype)

def eye_(var):
    "\n原地修改矩阵 ``var``，将其成为单位矩阵。\n\n    单位矩阵的定义为：\n    对于任何i, j (0 <= i, j < var的维度)，当i等于j时，输出为1；否则为0。\n\n    .. math::\n        eye(i, j) =\n        \\begin{cases}\n        1, & \\text{如果} i = j \\\\\n        0, & \\text{否则}\n        \\end{cases}\n    \n\n    参数：\n        - var (Var): Var类型的张量\n\n    代码示例：\n        >>> x = jt.randn((3,2))\n        >>> x\n        jt.Var([[-0.582047    1.1068922 ]\n                [ 0.08440255 -0.86142904]\n                [ 0.06269896 -0.979008  ]], dtype=float32)\n        >>> init.eye_(x)\n        >>> x\n        jt.Var([[1. 0.]\n                [0. 1.]\n                [0. 0.]], dtype=float32)\n\n    返回值：\n        返回一个形状和 ``var`` 相同的单位矩阵\n\n\n"
    return var.assign(eye(var.shape, var.dtype))
Var.eye_ = eye_

def constant(shape, dtype="float32", value=0.0):
    "\n创建一个数值均为 ``value``, 形状由可变参数 ``shape`` 确定， 默认填充 ``float32`` 类型的零值。\n\n    参数：\n\n    - shape (int...): 整数序列，定义了输出的形状\n    - dtype (var.dtype, optional): 数据类型，默认为 ``float32``\n    - value (int or float): 填充的数值\n        \n    代码示例：\n        >>> init.constant(2)\n        jt.Var([0. 0.], dtype=float32)\n        >>> init.constant((2,3), \"float32\", 3.8)\n        >>> x\n        jt.Var([[3.8 3.8 3.8]\n                [3.8 3.8 3.8]], dtype=float32)\n\n    返回值：\n        返回一个填充值为 ``value`` ，形状大小为 ``shape`` 的张量(Var)\n\n\n"
    return jt.array(value).unary(dtype).broadcast(NanoVector(shape))

def constant_(var, value=0.0):
    " \n原地修改张量 ``var`` ，将其修改为数值均为 ``value`` 的张量，默认填充零值。\n\n    参数：\n\n    - var (Var): Var类型的张量\n    - value (int or float): 整数序列，定义了输出的形状\n        \n    代码示例：\n        >>> x = jt.randn((3,2))\n        >>> x\n        jt.Var([[ 0.0335454  -0.18658346]\n                [-0.53283346 -1.2073938 ]\n                [-1.344916   -1.6093305 ]], dtype=float32)\n        >>> init.constant_(x, 3.8)\n        >>> x\n        >jt.Var([[3.8 3.8]\n                 [3.8 3.8]\n                 [3.8 3.8]], dtype=float32)\n\n    返回值：\n        就地修改 ``var`` ，返回一个填充值为 ``value`` ，形状和 ``var`` 相同的张量(Var)\n\n\n"
    return var.assign(constant(var.shape, var.dtype, value))
Var.constant_ = constant_
fill = Var.fill_ = constant_

def zero(shape, dtype="float32"):
    "  \n返回一个全为0的张量(Var)，形状由可变参数 ``shape`` 定义，数据类型由 ``dtype`` 定义。如果不给定 ``dtype`` , 默认类型为 ``float32``。\n\n    参数：\n        - shape (int...): 整数序列，定义了输出的形状\n        - dtype (var.dtype, optional): 数据类型，默认为`float32`\n        \n    代码示例：\n        >>> init.zero((3,2), dtype='int32')\n        jt.Var([[0 0]\n                [0 0]\n                [0 0]], dtype=int32)\n        \n    返回值：\n        全为0的张量(Var)\n\n    注意事项：\n        - 该函数和 ``jt.zeros()`` 用处一致\n\n"
    return constant(shape, dtype, 0)
def zero_(var):
    " 原地修改张量 ``var`` ，将其修改为数值均为0的张量。\n\n    参数：\n        - var (Var): Var类型的张量\n        \n    代码示例：\n        >>> x = jt.randn((3,2))\n        >>> x\n        jt.Var([[-1.1730903   0.21458259]\n                [ 1.0399616   0.07660236]\n                [-1.8453276  -0.95629567]], dtype=float32)\n        >>> init.zero_(x)\n        >>> x\n        jt.Var([[0. 0.]\n                [0. 0.]\n                [0. 0.]], dtype=float32)\n        \n    返回值：\n        全为0的张量(Var)\n\n"
    return var.assign(zero(var.shape, var.dtype))
Var.zero_ = zero_

def random_(var):
    return var.assign(jt.rand(var.shape, var.dtype))
Var.random_ = random_ 

def one(shape, dtype="float32"):
    "\n返回一个全为1的张量(Var)，形状由可变参数 ``shape`` 定义, 数据类型由可变参数 ``dtype`` 定义，默认类型  ``float32``。\n\n    参数：\n        - shape (int...): 整型序列，定义了输出的形状\n        - dtype (var.dtype, optional): 数据类型，默认为 ``float32``\n        \n    代码示例：\n        >>> init.one(3,4)\n        jt.Var([[1. 1. 1. 1.]\n                [1. 1. 1. 1.]\n                [1. 1. 1. 1.]], dtype=float32)\n        >>> init.one(5)\n        jt.Var([1. 1. 1. 1. 1.], dtype=float32)\n        \n    返回值：\n        全为1的张量(Var)\n\n\n"
    return constant(shape, dtype, 1)
def one_(var):
    "\n原地修改张量 ``var`` ，将其数值全部填充为1，数据类型保持不变。\n    \n    参数：\n        - var (Var): Var类型的张量\n        \n    代码示例：\n        >>> x = jt.randn((3,2))\n        >>> x\n        jt.Var([[ 0.8584159  -1.1204817 ]\n                [ 0.5418147  -0.62170196]\n                [-0.91137475 -0.13982968]], dtype=float32)\n        >>> init.one_(x)\n        >>> x\n        jt.Var([[1. 1.]\n                [1. 1.]\n                [1. 1.]], dtype=float32)\n        \n    返回值：\n        就地修改张量 ``var`` ，返回一个数值全为1的张量\n\n"
    return var.assign(one(var.shape, var.dtype))
Var.one_ = one_

def uniform(shape, dtype="float32", low=0, high=1):
    "\n创建一个张量，其形状由可变参数 ``shape`` 决定， 其数局类型由 ``dtype`` 决定， 其数值满足如下均匀分布：\n    \n    .. math::\n        out_i \\sim \\mathcal{U}(low, high)\n\n\n    参数：\n        - shape (int...): 整型序列，定义了输出的形状\n        - dtype (var.dtype, optional): 数据类型，默认为 ``float32``\n        - low (int or Var, optional): 均匀分布的上界， 默认值为0\n        - high (int or Var, optional): 均匀分布的下界，默认值为1\n        \n    代码示例：\n        >>> init.uniform((3,2))\n        jt.Var([[0.82488173 0.7875814 ]\n                [0.8802127  0.7350983 ]\n                [0.79734576 0.9903714 ]], dtype=float32)\n        >>> init.uniform((2,2), 'float32', 0, 10)\n        jt.Var([[0.35382926 6.8104105 ]\n                [2.5638878  1.2676293 ]], dtype=float32)\n        \n    返回值：\n        返回一个数值符合均匀分布的张量\n\n"
    return jt.random(NanoVector(shape), dtype) * (low - high) + high

def uniform_(var, low=0, high=1):
    "\n原地修改张量 ``var`` ，将其数值填充满足如下均匀分布的随机数：\n\n    .. math::\n        out_i \\sim \\mathcal{U}(low, high)\n\n    参数：\n        - var (Var): Var类型的张量\n        - low (int or Var, optional): 均匀分布的上界， 默认值为0\n        - high (int or Var, optional): 均匀分布的下界，默认值为1\n        \n    代码示例：\n        >>> x = jt.ones((3,2))\n        >>> x\n        jt.Var([[1. 1.]\n                [1. 1.]\n                [1. 1.]], dtype=float32)\n        >>> init.uniform_(x)\n        >>> x\n        jt.Var([[0.82488173 0.7875814 ]\n                [0.8802127  0.7350983 ]\n                [0.79734576 0.9903714 ]], dtype=float32)\n        \n    返回值：\n        就地修改张量 ``var`` ，返回一个数值符合均匀分布的张量\n\n"
    return var.assign(uniform(var.shape, var.dtype, low, high))
Var.uniform_ = uniform_

def gauss(shape, dtype="float32", mean=0.0, std=1.0):
    "\n创建一个张量，其形状由 ``shape`` 指定，数据类型为  ``dtype`` ，元素值遵循均值为 ``mean`` 和标准差为 ``std`` 的高斯分布（正态分布）的张量。\n\n    .. math::\n        out_i \\sim \\mathcal{N}(mean, std^2)\n\n    参数：\n        - shape (int...): 整型序列，定义了输出的形状\n        - dtype (var.dtype, optional): 数据类型，默认为 ``float32``\n        - mean (float or Var, optional): 高斯分布的均值， 默认值为0.0\n        - std (float or Var, optional): 高斯分布的标准差，默认值为1.0\n        \n    代码示例：\n        >>> init.gauss((3,2))\n        jt.Var([[-1.09277    -0.22924843]\n                [-0.5264394  -0.13242662]\n                [-1.1316705   1.2506602 ]], dtype=float32)\n        >>> init.gauss((2,2), 'float32', 0, 10)\n        jt.Var([[ 3.457805   5.8171034]\n                [-1.6440934  2.1744032]], dtype=float32)\n        \n    返回值：\n        返回一个数值符合高斯分布的张量\n\n    注意事项：\n        - 确保指定的 ``std`` 是非负的，因为标准差不能是负数\n\n"
    return jt.random(NanoVector(shape), dtype, "normal") * std + mean

def gauss_(var, mean=0.0, std=1.0):
    "\n原地修改张量 ``var`` ，将其数值填充满足以均值为 ``mean`` 和标准差为 ``std`` 的高斯分布（正态分布）的随机数。\n\n    .. math::\n        out_i \\sim \\mathcal{N}(mean, std^2)\n\n    参数：\n        - var (Var): Var类型的张量\n        - mean (float or Var, optional): 高斯分布的均值， 默认值为0.0\n        - std (float or Var, optional): 高斯分布的标准差，默认值为1.0\n        \n    代码示例：\n        >>> x = init.zero((3,3))\n        >>> x\n        jt.Var([[0. 0. 0.]\n                [0. 0. 0.]\n                [0. 0. 0.]], dtype=float32)\n        >>> init.gauss_(x, 0, 5.0)\n        >>> x\n        jt.Var([[-0.34221977 -8.056475    2.6251674 ]\n                [-0.81275284  2.00393    -2.4397573 ]\n                [ 0.7867035  -8.159389    7.100675  ]], dtype=float32)\n        \n    返回值：\n        就地修改张量 ``var`` ，返回一个数值符合高斯分布的张量\n\n"
    return var.assign(gauss(var.shape, var.dtype, mean, std))
Var.gauss_ = gauss_
Var.normal_ = gauss_

def invariant_uniform(shape, dtype="float32", mode="fan_in"):
    ''' Return Jittor initialized Var by invariant_uniform.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Example::

        from jittor import init
        from jittor import nn
        a = init.invariant_uniform_((2,2))
        print(a)

    '''
    assert len(shape)>1
    assert mode=="fan_in" or mode=="fan_out", \
        f"mode not supported, should be fan_in or fan_out, but got {mode}"

    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) if mode=="fan_in" else (shape[0] * matsize)
    bound = math.sqrt(1.0/fan)
    return uniform(shape, dtype, -bound, bound)

def invariant_uniform_(var, mode="fan_in"):
    ''' Inplace initialize Jittor Var by invariant_uniform.

    Args:
        var (Jittor Var):
            Var to be initialized by random invariant_uniform
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.invariant_uniform_(linear.weight)
        print(linear.weight)
        linear.weight.invariant_uniform_() # This is ok too

    '''
    var.assign(invariant_uniform(tuple(var.shape), var.dtype, mode))
Var.invariant_uniform_ = invariant_uniform_

def relu_invariant_gauss(shape, dtype="float32", mode="fan_in"):
    ''' Return Jittor Var initialized by relu_invariant_gauss.

    Args:
        shape (int or tuple of int):
            shape of the output Var
        dtype (string):
            dtype of the output Var, default float32
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Example::

        from jittor import init
        from jittor import nn
        a = init.relu_invariant_gauss((2,2))
        print(a)
    
    '''
    assert len(shape)>1
    assert mode=="fan_in" or mode=="fan_out"
    
    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) if mode=="fan_in" else (shape[0] * matsize)
    std = math.sqrt(2.0/fan)
    return gauss(shape, dtype, 0, std)

def relu_invariant_gauss_(var, mode="fan_in"):
    ''' Inplace initialize Jittor Var by relu_invariant_gauss.

    Args:
        var (Jittor Var):
            Var to be initialized by random relu_invariant_gauss
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.relu_invariant_gauss_(linear.weight)
        print(linear.weight)
        linear.weight.relu_invariant_gauss_() # This is ok too

    '''
    return var.assign(relu_invariant_gauss(tuple(var.shape), var.dtype, mode))
Var.relu_invariant_gauss_ = relu_invariant_gauss_

def calculate_std(var, mode, nonlinearity, param=0.01):
    mode = mode.lower()
    assert isinstance(param,(int,float))
    assert var.ndim>=2
    assert mode in ['fan_in', 'fan_out']

    fan = var.shape[1] if mode == 'fan_in' else var.shape[0]
    fan *= var[0][0].numel()

    gains = {
        'linear':1,
        'conv1d':1,
        'conv2d':1,
        'conv3d':1,
        'conv_transpose1d':1,
        'conv_transpose2d':1,
        'conv_transpose3d':1,
        'sigmoid':1,
        'tanh':5.0/3,
        'relu':math.sqrt(2.0),
        'leaky_relu':math.sqrt(2.0 / (1 + param ** 2)),
    }
    gain = gains[nonlinearity]
    std = gain/math.sqrt(fan)
    return std


def kaiming_uniform_(var, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    ''' Inplace initialize Jittor Var by kaiming_uniform.

    Args:
        var (Jittor Var):
            Var to be initialized by random kaiming_uniform
        a (float):
            the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
        nonlinearity (string):
            nonlinearity used after this layer. 
            It can be one of [linear, conv*, sigmoid, tanh, relu, leaky_relu].
            leaky_relu is used by default.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.kaiming_uniform_(linear.weight)
        print(linear.weight)
        linear.weight.kaiming_uniform_() # This is ok too

    '''
    std = calculate_std(var,mode,nonlinearity,a)
    bound = math.sqrt(3.0) * std
    return uniform_(var,-bound, bound)
Var.kaiming_uniform_ = kaiming_uniform_

def kaiming_normal_(var, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    ''' Inplace initialize Jittor Var by kaiming_normal.

    Args:
        var (Jittor Var):
            Var to be initialized by random kaiming_normal
        a (float):
            the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
        mode (string):
            mode selection, should be fan_in or fan_out.
            Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
        nonlinearity (string):
            nonlinearity used after this layer. 
            It can be one of [linear, conv*, sigmoid, tanh, relu, leaky_relu].
            leaky_relu is used by default.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.kaiming_normal_(linear.weight)
        print(linear.weight)
        linear.weight.kaiming_normal_() # This is ok too

    '''
    std = calculate_std(var,mode,nonlinearity,a)
    return gauss_(var,0, std)
Var.kaiming_normal_ = kaiming_normal_


def xavier_uniform(shape, dtype="float32", gain=1.0):
    ''' Inplace initialize Jittor Var by xavier_uniform.
    The resulting var will have values sampled from
    :math:`uniform(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Args:
        shape (int or tuple of int):
            shape of the return Var.
        dtype (string):
            dtype of the return Var, default float32.
        gain (float):
            an optional scaling factor.

    Example::

        from jittor import init
        from jittor import nn
        a = init.xavier_uniform((2,2), gain=init.calculate_gain('relu'))
        print(a)
    '''
    assert len(shape)>1

    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) + (shape[0] * matsize)
    bound = gain * math.sqrt(6.0/fan)
    return uniform(shape, dtype, -bound, bound)

def xavier_uniform_(var, gain=1.0):
    ''' Inplace initialize Jittor Var by xavier_uniform.
    The resulting var will have values sampled from
    :math:`uniform(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Args:
        var (Jittor Var):
            Var to be initialized by random xavier_uniform
        gain (float):
            an optional scaling factor.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.xavier_uniform_(linear.weight, init.calculate_gain('relu'))
        print(linear.weight)
        linear.weight.xavier_uniform_() # This is ok too

    '''
    return var.assign(xavier_uniform(tuple(var.shape), var.dtype, gain))
Var.xavier_uniform_ = xavier_uniform_

def xavier_gauss(shape, dtype="float32", gain=1.0):
    ''' Return Jittor Var initialized by xavier_gauss, a.k.a xavier_normal.
    The resulting var will have values sampled from
    :math:`gauss(-a, a)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Args:
        shape (int or tuple of int):
            shape of the return Var.
        dtype (string):
            dtype of the return Var, default float32.
        gain (float):
            an optional scaling factor.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.xavier_gauss_(linear.weight, init.calculate_gain('relu'))
        print(linear.weight)
        linear.weight.xavier_gauss_() # This is ok too

    '''
    assert len(shape)>1
    
    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) + (shape[0] * matsize)
    std = gain * math.sqrt(2.0/fan)
    return gauss(shape, dtype, 0, std)

def xavier_gauss_(var, gain=1.0):
    ''' Inplace initialize Jittor Var by xavier_gauss, a.k.a xavier_normal.
    The resulting var will have values sampled from
    :math:`gauss(-a, a)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Args:
        var (Jittor Var):
            Var to be initialized by random xavier_gauss
        gain (float):
            an optional scaling factor.

    Example::

        from jittor import init
        from jittor import nn
        linear = nn.Linear(2,2)
        init.xavier_gauss_(linear.weight, init.calculate_gain('relu'))
        print(linear.weight)
        linear.weight.xavier_gauss_() # This is ok too

    '''
    return var.assign(xavier_gauss(tuple(var.shape), var.dtype, gain))
Var.xavier_gauss_ = xavier_gauss_

def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4 
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def trunc_normal_(var, mean=0., std=1., a=-2., b=2.):
    # type: (jt.jittor_core.Var, float, float, float, float) -> jt.jittor_core.Var
    "\n将输入的 :math:`jt.jittor\\_core.Var` 用一个截断正态分布中的值填充。这些值实际上是从正态分布 :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)` 中抽取的，其中超出 :math:`[a, b]` 范围的值会被重新抽取，直到他们在这个范围之内。生成随机值的方法在 :math:`a \\leq \\text{mean} \\leq b` 这个条件下工作得最好。\n\n    具体而言，:math:`trunc\\_normal\\_` 函数采样的方法如下： 首先，根据指定的均值 ``mean`` 和标准差 ``std`` ，从正态分布中采样一个随机值 :math:`x` 。然后，对采样的随机值 :math:`x` 进行截断操作，将其限制在指定的范围[ ``a`` ,  ``b`` ]内。具体而言，如果  :math:`x` 小于下界 ``a`` ，则将它替换为 ``a`` ；如果x大于上界 ``b``，则将它替换为 ``b``。最后，将截断后的值作为初始化值赋给张量 ``var`` 的对应元素。\n\n    参数:\n        - var (jt.jittor_core.Var): 一个n维的变量\n        - mean (float， 可选): 正态分布的均值，默认值：0.0\n        - std (float， 可选): 正态分布的标准差，默认值：1.0\n        - a (float， 可选): 最小截断值，默认值：-2.0\n        - b (float， 可选): 最大截断值，默认值：2.0\n\n    代码示例:\n        >>> from jittor import init\n        >>> from jittor import nn\n        >>> linear = nn.Linear(2,2)\n        >>> init.trunc_normal_(linear.weight, std=.02)\n        >>> linear.weight    \n  \n    返回值：\n        使用截断正态分布填充的输入变量(Var)\n\n\n"
    return var.assign(_no_grad_trunc_normal_(var, mean, std, a, b))
Var.trunc_normal_ = trunc_normal_

def _no_grad_trunc_normal_(var, mean, std, a, b):
    "\n使用截断正态分布初始化给定变量的值。与 :math:`trunc\\_normal\\_` 函数不同的是，该函数在执行初始化时不会进行梯度计算，即不会自动记录梯度。\n\n    该函数使用均值为 ``mean`` , 标准差为 ``std`` , 上下限为 ``a`` ,  ``b`` 的截断正态分布初始化给定的变量 ``var`` .当均值 ``mean`` 超出 ``[a-2*std, b+2*std]`` 的范围时，会弹出警告提示。\n\n    在数学上，截断正态分布的累积分布函数为 :math:`F(x)=\\frac{1+\\text{erf}(x / \\sqrt{2})}{2}`。\n\n    参数:\n        - var (jt.Var): 要初始化的变量。\n        - mean (float): 截断正态分布的均值。\n        - std (float): 截断正态分布的标准差。\n        - a (float): 截断正态分布的下界。\n        - b (float): 截断正态分布的上界。\n\n    返回值:\n        初始化后的变量(Var)。\n\n    代码示例:\n        >>> import jittor as jt\n        >>> var = jt.zeros([3, 3])\n        >>> var = jt._no_grad_trunc_normal_(var, mean=0., std=1., a=-2., b=2.)\n\n"
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)


    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    # var.uniform(2 * l - 1, 2 * u - 1)
    var.uniform_(low=2 * l - 1, high=2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    var = var.erfinv()

    # Transform to proper mean, std
    var = var.multiply(std * math.sqrt(2.))
    var = var.add(mean)

    # Clamp to ensure it's in the proper range
    var = var.clamp(min_v=a, max_v=b)
    return var