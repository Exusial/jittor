# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#   Dun Liang <randonlang@gmail.com>.
#   Xiangli Li <190569238@qq.com>
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
import numpy as np

class SparseVar:
    def __init__(self,indices,values,shape):
        assert isinstance(indices,jt.Var) and isinstance(values,jt.Var) and isinstance(shape,jt.NanoVector)
        self.indices = indices
        self.values = values
        self.shape = shape
        self.ndim = len(shape)
        
    def _indices(self):
        return self.indices
    
    def _values(self):
        return self.values
    
    def t(self):
        indices = list(self.indices.split(1,dim=0))
        indices[-1],indices[-2] = indices[-2],indices[-1]
        indices = jt.concat(indices,dim=0)
        shape = list(self.shape)
        shape[-1],shape[-2] = shape[-2],shape[-1]
        shape = jt.NanoVector(shape)
        return SparseVar(indices,self.values,shape)
        
    def to_dense(self):
        ret = jt.zeros(self.shape,self.values.dtype)
        indices  = tuple(self.indices.split(1,dim=0))
        ret[indices]=self.values
        return ret

def sparse_array(indices,values,shape):
    "\n将给出的稀疏数组的索引、值和形状构建一个Jittor稀疏张量。对于任何给定的索引，其对应的值不应该为零。以下为数学表示，其中 (i_1, i_2, ..., i_N)是indices的某一行，(j) 是该行在indices中的索引：\n\n    .. math::\n        SparseVar[i_1, i_2, ..., i_N] = values[j]\n      \n    参数:\n        - indices (jittor.Var): 稀疏数组的索引，必须是2D整型的Jittor变量。其每一行是代表非零值在最终结果中的位置\n        - values (jittor.Var): 非零值组成的1D浮点型Jittor变量。\n        - shape (jittor.NanoVector): 稀疏张量的形状。长度必须和indices中的列数相同。\n\n    返回值:\n        jt.sparse.SparseVar: 一个Jittor的稀疏张量\n\n    代码示例：\n        >>> import jittor as jt\n        >>> indices = jt.array([[0, 0], [1, 2]])\n        >>> values = jt.array([1, 2])\n        >>> shape = jt.NanoVector([3, 4])\n        >>> jt.sparse.sparse_array(indices, values, shape).shape\n        [3,4,]\n\n"
    return SparseVar(indices,values,shape)

def spmm(spase_x,y):
    "\n稀疏矩阵和密集矩阵的乘法操作。此函数先将稀疏矩阵转化为密集矩阵，然后使用Jittor的矩阵乘法操作将两个矩阵相乘。此函数需要等长的rows和columns，即输入的两个矩阵可以执行矩阵乘法操作。\n      \n    参数:\n        - spase_x (jittor.sparse.SparseVar): 2维稀疏矩阵。假设有N行和M列\n        - y (jittor.Var): 2维密集矩阵。假设有M行和P列\n\n    返回值:\n        jt.Var:  结果矩阵。它是一个N行和P列的密集矩阵\n\n    代码示例：\n        >>> import jittor as jt\n        >>> indices = jt.array([[0, 0], [1, 2]])\n        >>> values = jt.array([1, 2])\n        >>> shape = jt.NanoVector([3, 4])\n        >>> sparse_mat = jt.sparse.sparse_array(indices, values, shape)\n        [3,4,]\n        >>> y = jt.randn(4,5) \n        >>> jt.sparse.spmm(sparse_mat, y).shape \n        [3,5,]\n\n"
    assert isinstance(spase_x,SparseVar) and isinstance(y,jt.Var)
    assert spase_x.ndim==2 and y.ndim==2 and spase_x.shape[-1]==y.shape[0]
    
    # TODO
    x = spase_x.to_dense()
    return jt.matmul(x,y)
    