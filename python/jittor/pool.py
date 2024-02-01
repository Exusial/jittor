# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import init, Module
import numpy as np
import math

pool_use_code_op = True

class Pool(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None or op == "maximum"
        self.return_indices = return_indices
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.op = op
        stride = stride if stride else kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N,C,H,W = x.shape
        if self.ceil_mode == False:
            h = (H+self.padding[0]*2-self.kernel_size[0])//self.stride[0]+1
            w = (W+self.padding[1]*2-self.kernel_size[1])//self.stride[1]+1
            use_code_op = self.op in ['maximum', 'minimum']
            # some second order avg_pool is require, so we don't use code op here  
        else:
            h = (H+self.padding[0]*2-self.kernel_size[0] + self.stride[0] - 1)//self.stride[0]+1
            w = (W+self.padding[1]*2-self.kernel_size[1] + self.stride[1] - 1)//self.stride[1]+1
            use_code_op = self.op in ['maximum', 'minimum', 'mean']

        if use_code_op and pool_use_code_op:
            if self.op == 'mean':
                if self.count_include_pad:
                    count = f"int count = {self.kernel_size[0]*self.kernel_size[1]};"
                else:
                    count = "int count = (k2_ - k2) * (k3_ - k3);"
                count += "float32 rcount = 1.0f / count;"
            else:
                count = ""
            forward_body = f'''
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
            '''
            if not self.return_indices:
                forward_body += f'''
                @out(i0, i1, i2, i3) = @expand_op(init_{self.op}, @out_type);
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        @out(i0, i1, i2, i3) = @expand_op({self.op}, @out_type, @out(i0, i1, i2, i3), @out_type, @in0(i0, i1, p, q), @in0_type);
                '''
            else:
                forward_body += f'''
                auto out_value = @expand_op(init_{self.op}, @out_type);
                int out_index = -1;
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q) 
                        if (out_value < @in0(i0, i1, p, q)) {{
                            out_value = @in0(i0, i1, p, q);
                            out_index = p * in0_shape3 + q;
                        }}
                @out(i0, i1, i2, i3) = out_value;
                @out1(i0, i1, i2, i3) = out_index;
                '''
            backward_body = f'''
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
                int bo=1;
                for (int p = k2; p < k2_ && bo; ++p)
                    for (int q = k3; q < k3_ && bo; ++q) {{
                        {"atomicAdd(&@out(i0,i1,p,q), @dout(i0,i1,i2,i3)/count);"
                            if self.op == "mean" else
                        f"""if (@pout(i0,i1,i2,i3) == @in0(i0,i1,p,q)) {{
                            atomicAdd(&@out(i0,i1,p,q), @dout(i0,i1,i2,i3)),
                            bo=0;
                        }}"""}
                    }}
            '''
            if self.return_indices:
                return_shapes = [[N,C,h,w]] * 2
                return_dtypes = [x.dtype, 'int32']
            else:
                return_shapes = [N,C,h,w]
                return_dtypes = x.dtype
            out = jt.code(return_shapes, return_dtypes, [x],
                cuda_header="""
                    #include <misc/cuda_limits.h>
                """,
                cuda_src=f'''
                    __global__ static void kernel1(@ARGS_DEF) {{
                        @PRECALC
                        int p3 = threadIdx.x;
                        int s3 = blockDim.x;
                        int p2 = threadIdx.y + blockIdx.x * blockDim.y;
                        int s2 = blockDim.y * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i3 = p3; i3 < out_shape3; i3 += s3)
                        for (int i2 = p2; i2 < out_shape2; i2 += s2)
                            {{ {forward_body} }}
                    }}
                    int tx = std::min(1024, out_shape3);
                    int ty = std::min(1024 / tx, out_shape2);
                    int bx = (out_shape2 - 1) / ty + 1;
                    int by = out_shape1;
                    int bz = out_shape0;
                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty);
                    kernel1<<<s1, s2>>>(@ARGS);
                ''',
                cuda_grad_src=[f'''
                    __global__ static void kernel3(@ARGS_DEF) {{
                        @PRECALC
                        int p3 = threadIdx.x;
                        int s3 = blockDim.x;
                        int p2 = threadIdx.y + blockIdx.x * blockDim.y;
                        int s2 = blockDim.y * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i3 = p3; i3 < pout_shape3; i3 += s3)
                            for (int i2 = p2; i2 < pout_shape2; i2 += s2)
                                {{ {backward_body} }}
                    }}
                    cudaMemsetAsync(out_p, 0, out->size);
                    int tx = std::min(1024, pout_shape3);
                    int ty = std::min(1024 / tx, pout_shape2);
                    int bx = (pout_shape2 - 1) / ty + 1;
                    int by = pout_shape1;
                    int bz = pout_shape0;
                    dim3 s1_(bx, by, bz);
                    dim3 s2_(tx, ty);
                    kernel3<<<s1_, s2_>>>(@ARGS);
                '''],
                cpu_header='',
                cpu_src=f'''
                    using namespace std;
                    for (int i0=0; i0<out_shape0; i0++)
                    for (int i1=0; i1<out_shape1; i1++)
                    for (int i2=0; i2<out_shape2; i2++)
                    for (int i3=0; i3<out_shape3; i3++)
                        {{ {forward_body} }}
                ''',
                cpu_grad_src = [f'''
                    using namespace std;
                    std::memset(out_p, 0, out->size);
                    #define atomicAdd(a,b) (*a) += b

                    for (int i0=0; i0<pout_shape0; i0++)
                    for (int i1=0; i1<pout_shape1; i1++)
                    for (int i2=0; i2<pout_shape2; i2++) 
                    for (int i3=0; i3<pout_shape3; i3++)
                        {{ {backward_body} }}
                '''])
            return out
        else:
            # TODO: backward 
            xx = x.reindex([N,C,h,w,self.kernel_size[0],self.kernel_size[1]], [
                "i0", # Nid
                "i1", # Cid
                f"i2*{self.stride[0]}-{self.padding[0]}+i4", # Hid
                f"i3*{self.stride[1]}-{self.padding[1]}+i5", # Wid
            ])
            return xx.reduce(self.op, [4,5])

def _triple(x):
    "\n    将输入 ``x`` 处理为一个三元组。\n    如果输入 ``x`` 是一个包含3个元素的元组，会原样返回。否则，会返回一个新的三元组，其三个元素都为 ``x`` 。\n\n    参数:\n        x (Any): 输入变量，可以为任意类型。\n\n    \n    代码示例:\n        >>> _triple(3) \n        (3, 3, 3)\n        >>>  _triple((1, 2, 3))\n        (1, 2, 3)\n\n    返回值:\n       返回一个三元组(tuple)。如果输入 ``x`` 已经是一个三元组，那么返回原输入。否则返回的三元组中的每个元素都为输入 ``x`` 。\n\n    \n"
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    else:
        return (x,x,x)


class Pool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None or op == "maximum"
        self.return_indices = return_indices
        self.kernel_size = _triple(kernel_size)
        self.op = op
        stride = stride if stride else kernel_size
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N,C,D,H,W = x.shape
        if self.ceil_mode == False:
            d = (D+self.padding[0]*2-self.kernel_size[0])//self.stride[0]+1
            h = (H+self.padding[1]*2-self.kernel_size[1])//self.stride[1]+1
            w = (W+self.padding[2]*2-self.kernel_size[2])//self.stride[2]+1
            use_code_op = self.op in ['maximum', 'minimum']
            # some second order avg_pool is require, so we don't use code op here  
        else:
            d = (D+self.padding[0]*2-self.kernel_size[0] + self.stride[0] - 1)//self.stride[0]+1
            h = (H+self.padding[1]*2-self.kernel_size[1] + self.stride[1] - 1)//self.stride[1]+1
            w = (W+self.padding[2]*2-self.kernel_size[2] + self.stride[2] - 1)//self.stride[2]+1
            use_code_op = self.op in ['maximum', 'minimum', 'mean']

        if use_code_op and pool_use_code_op:
            if self.op == 'mean':
                if self.count_include_pad:
                    count = f"int count = {self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]};"
                else:
                    count = "int count = (k2_ - k2) * (k3_ - k3) * (k4_ - k4);"
                count += "float32 rcount = 1.0f / count;"
            else:
                count = ""
            forward_body = f'''
                int k4 = i4*{self.stride[2]}-{self.padding[2]};
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k4_ = min(k4 + {self.kernel_size[2]}, in0_shape4);
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
            '''
            if not self.return_indices:
                forward_body += f'''
                @out(i0, i1, i2, i3, i4) = @expand_op(init_{self.op}, @out_type);
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        for (int r = k4; r < k4_; ++r)
                            @out(i0, i1, i2, i3, i4) = @expand_op({self.op}, @out_type, @out(i0, i1, i2, i3, i4), @out_type, @in0(i0, i1, p, q, r), @in0_type);
                '''
            else:
                forward_body += f'''
                auto out_value = @expand_op(init_{self.op}, @out_type);
                int out_index = -1;
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q) 
                        for (int r = k4; q < k4_; ++r) 
                        if (out_value < @in0(i0, i1, p, q, r)) {{
                            out_value = @in0(i0, i1, p, q, r);
                            out_index = p * in0_shape3 * in0_shape4 + q * in0_shape4 + r;
                        }}
                @out(i0, i1, i2, i3, i4) = out_value;
                @out1(i0, i1, i2, i3, i4) = out_index;
                '''
            backward_body = f'''
                int k4 = i4*{self.stride[2]}-{self.padding[2]};
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k4_ = min(k4 + {self.kernel_size[2]}, in0_shape4);
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
                int bo=1;
                for (int p = k2; p < k2_ && bo; ++p)
                    for (int q = k3; q < k3_ && bo; ++q) 
                        for (int r = k4; r < k4_ && bo; ++r) {{
                            {"atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)/count);"
                                if self.op == "mean" else
                            f"""if (@pout(i0,i1,i2,i3,i4) == @in0(i0,i1,p,q,r)) {{
                                atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)),
                                bo=0;
                            }}"""}
                        }}
            '''
            if self.return_indices:
                return_shapes = [[N,C,d,h,w]] * 2
                return_dtypes = [x.dtype, 'int32']
            else:
                return_shapes = [N,C,d,h,w]
                return_dtypes = x.dtype
            out = jt.code(return_shapes, return_dtypes, [x],
                cuda_header="""
                    #include <misc/cuda_limits.h>
                """,
                cuda_src=f'''
                    __global__ static void kernel1(@ARGS_DEF) {{
                        @PRECALC
                        int p4 = threadIdx.x;
                        int s4 = blockDim.x;
                        int p3 = threadIdx.y;
                        int s3 = blockDim.y;
                        int p2 = threadIdx.z + blockIdx.x * blockDim.z;
                        int s2 = blockDim.z * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i4 = p4; i4 < out_shape4; i4 += s4)
                        for (int i3 = p3; i3 < out_shape3; i3 += s3)
                        for (int i2 = p2; i2 < out_shape2; i2 += s2)
                            {{ {forward_body} }}
                    }}
                    int tx = std::min(1024, out_shape4);
                    int ty = std::min(1024 / tx, out_shape3);
                    int tz = std::min(1024 / tx / ty, out_shape2);
                    int bx = (out_shape2 - 1) / tz + 1;
                    int by = out_shape1;
                    int bz = out_shape0;
                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty, tz);
                    kernel1<<<s1, s2>>>(@ARGS);
                ''',
                cuda_grad_src=[f'''
                    __global__ static void kernel3(@ARGS_DEF) {{
                        @PRECALC
                        int p4 = threadIdx.x;
                        int s4 = blockDim.x;
                        int p3 = threadIdx.y;
                        int s3 = blockDim.y;
                        int p2 = threadIdx.z + blockIdx.x * blockDim.z;
                        int s2 = blockDim.z * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i4 = p4; i4 < out_shape4; i4 += s4)
                        for (int i3 = p3; i3 < out_shape3; i3 += s3)
                        for (int i2 = p2; i2 < out_shape2; i2 += s2)
                                {{ {backward_body} }}
                    }}
                    cudaMemsetAsync(out_p, 0, out->size);
                    int tx = std::min(1024, pout_shape4);
                    int ty = std::min(1024 / tx, pout_shape3);
                    int tz = std::min(1024 / tx / ty, pout_shape2);
                    int bx = (pout_shape2 - 1) / tz + 1;
                    int by = pout_shape1;
                    int bz = pout_shape0;
                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty, tz);
                    kernel3<<<s1, s2>>>(@ARGS);
                '''],
                cpu_header='',
                cpu_src=f'''
                    using namespace std;
                    for (int i0=0; i0<out_shape0; i0++)
                    for (int i1=0; i1<out_shape1; i1++)
                    for (int i2=0; i2<out_shape2; i2++)
                    for (int i3=0; i3<out_shape3; i3++)
                    for (int i4=0; i4<out_shape4; i4++)
                        {{ {forward_body} }}
                ''',
                cpu_grad_src = [f'''
                    using namespace std;
                    std::memset(out_p, 0, out->size);
                    #define atomicAdd(a,b) (*a) += b

                    for (int i0=0; i0<pout_shape0; i0++)
                    for (int i1=0; i1<pout_shape1; i1++)
                    for (int i2=0; i2<pout_shape2; i2++) 
                    for (int i3=0; i3<pout_shape3; i3++)
                    for (int i4=0; i4<pout_shape4; i4++)
                        {{ {backward_body} }}
                '''])
            return out
        else:
            # TODO: backward 
            xx = x.reindex([N,C,d,h,w,self.kernel_size[0],self.kernel_size[1],self.kernel_size[2]], [
                "i0", # Nid
                "i1", # Cid
                f"i2*{self.stride[0]}-{self.padding[0]}+i5", # Did
                f"i3*{self.stride[1]}-{self.padding[1]}+i6", # Hid
                f"i4*{self.stride[2]}-{self.padding[2]}+i7", # Hid
            ])
            return xx.reduce(self.op, [5,6,7])

class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        if isinstance(self.output_size, int):
            oh = self.output_size
            ow = self.output_size
        elif isinstance(self.output_size, tuple) or isinstance(self.output_size, list):
            oh = x.shape[2] if self.output_size[0] is None else self.output_size[0]
            ow = x.shape[3] if self.output_size[1] is None else self.output_size[1]
        else:
            raise TypeError(f"AdaptiveAvgPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
        if oh == 1 and ow == 1:
            return x.reduce("mean", [2,3], keepdims=True)
        N,C,H,W = x.shape
        self.sh = math.floor(H / oh)
        self.sw = math.floor(W / ow)
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,h,w,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sh}+i4", # Hid
            f"i3*{self.sw}+i5", # Wid
        ])
        return xx.reduce("mean", [4,5])

class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size, return_indices=False):
        self.output_size = output_size
        self.return_indices = return_indices

    def execute(self, x):
        if isinstance(self.output_size, int):
            oh = self.output_size
            ow = self.output_size
        elif isinstance(self.output_size, tuple) or isinstance(self.output_size, list):
            oh = x.shape[2] if self.output_size[0] is None else self.output_size[0]
            ow = x.shape[3] if self.output_size[1] is None else self.output_size[1]
        else:
            raise TypeError(f"AdaptiveMaxPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
        if oh == 1 and ow == 1:
            return x.reduce("maximum", [2,3], keepdims=True)
        N,C,H,W = x.shape
        self.sh = math.floor(H / oh)
        self.sw = math.floor(W / ow)
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        if self.return_indices:
            return MaxPool2d(
                kernel_size=(self.ksh, self.ksw),
                stride=(self.sh, self.sw), return_indices=True)(x)
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,h,w,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sh}+i4", # Hid
            f"i3*{self.sw}+i5", # Wid
        ])
        return xx.reduce("maximum", [4,5])


class AdaptiveAvgPool3d(Module):
    def __init__(self, output_size):
        self.output_size = _triple(output_size)

    def execute(self, x):
        od, oh, ow = self.output_size
        if od == 1 and oh == 1 and ow == 1:
            return x.reduce("mean", [2,3,4], keepdims=True)
        N,C,D,H,W = x.shape
        self.sd = math.floor(D / od)
        self.sh = math.floor(H / oh)
        self.sw = math.floor(W / ow)
        self.ksd = D - (od - 1) * self.sd
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        d = (D-self.ksd)//self.sd+1
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,d,h,w,self.ksd,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sd}+i5", # Did
            f"i3*{self.sh}+i6", # Hid
            f"i4*{self.sw}+i7", # Wid
        ])
        return xx.reduce("mean", [5,6,7])

class AdaptiveMaxPool3d(Module):
    def __init__(self, output_size, return_indices=False):
        self.output_size = _triple(output_size)
        self.return_indices = return_indices

    def execute(self, x):
        od, oh, ow = self.output_size
        if od == 1 and oh == 1 and ow == 1 and not self.return_indices:
            return x.reduce("maximum", [2,3,4], keepdims=True)
        N,C,D,H,W = x.shape
        self.sd = math.floor(D / od)
        self.sh = math.floor(H / oh)
        self.sw = math.floor(W / ow)
        self.ksd = D - (od - 1) * self.sd
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        if self.return_indices:
            return MaxPool3d(
                kernel_size=(self.ksd, self.ksh, self.ksw),
                stride=(self.sd, self.sh, self.sw), return_indices=True)(x)
        d = (D-self.ksd)//self.sd+1
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,d,h,w,self.ksd,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sd}+i5", # Did
            f"i3*{self.sh}+i6", # Hid
            f"i4*{self.sw}+i7", # Wid
        ])
        return xx.reduce("maximun", [5,6,7])

def pool(x, kernel_size, op, padding=0, stride=None):
    "    对输入的张量进行池化操作。此函数将对输入的张量 ``x`` 应用指定的池化操作，池化的方式由参数 ``op`` 指定。\n    数学表达式：\n    对于最大值池化，池化后的值为：\n    \n        y_{ij} = \\max_{m,n \\in [0,kernel\\_size)} x_{i+m, j+n}\n        \n    对于平均池化，池化后的值为：\n\n        y_{ij} = \\frac{1}{kernel\\_size^2} \\sum_{m,n \\in [0,kernel\\_size)} x_{i+m, j+n}\n    \n    参数：\n        - x (jt.Var): 输入的张量。\n        - kernel_size (int 或 tuple of int): 池化窗口的大小。 如果是一个整数，那么窗口将是一个方形，如果是一个元组，则元组的长度必须是2，分别代表窗口的高度和宽度。\n        - op (str): 池化方式， :math:`'max'` 表示最大值池化, :math:`'avg'` 表示平均池化。\n        - padding (int 或 tuple of int, 可选): 在输入的张量的各边填充0的宽度，默认值：0。如果是一个整数，那么所有边都填充同样的宽度，如果是一个元组，则元组的长度必须是2，分别代表高度和宽度方向的填充宽度。\n        - stride (int 或 tuple of int, 可选): 池化窗口移动的步长。默认值：None，即步长等于池化窗口的大小。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> x = jt.ones((1,3,4,4))\n        >>> y = pool(x, (2, 2), 'max')\n        >>> y.shape\n        (1, 3, 2, 2)\n \n    返回值：\n        池化后的张量(Var)。\n\n\n"
    return Pool(kernel_size, stride, padding, op=op)(x)

pool2d = pool

def pool3d(x, kernel_size, op, padding=0, stride=None):
    " 对3维输入进行池化操作。\n    .. math::\n        y_{d,h,w} = op\\{x_{s:t, s+m:t+m, s+n:t+n}\\}\n        \n    其中，:math:`s` 是在对应方向上的开始位置，:math:`t` 是结束位置，``op`` 是池化操作（可能是最大值或者平均值），:math:`m` 和:math:`n` 分别是 :math:`kernel` _ :math:`size` 在高和宽上的尺寸，:math:`s+m` 和 :math:`s+n` 分别是步长为 :math:`stride` 时滑动窗口的结束位置。\n\n    参数：\n        - x (jt.Var or jt.Module): 输入的3维张量。\n        - kernel_size (int or tuple of int): 池化窗口的尺寸。如果是单个整数，表示在所有方向上的窗口尺寸都是相同的。如果是一个长度为3的元组，则分别表示深度、高度和宽度上的窗口尺寸。\n        - op (str): 池化操作的类型。可以是 :math:`\"max\"` 表示最大值池化，或者是 :math:`\"avg\"` 表示平均值池化。\n        - padding (int or tuple of int, 可选): 输入的每一维在每个方向上的补0层数。默认值：0。如果是单个整数，表示在所有方向上的补0层数都是相同的。如果是一个长度为3的元组，则分别表示深度、高度和宽度上的补0层数。\n        - stride (int or tuple of int, 可选): 池化窗口的步长。默认值：等于 :math:`kernel` _ :math:`size` 。如果是单个整数，表示在所有方向上的步长都是相同的。如果是一个长度为3的元组，则分别表示深度、高度和宽度上的步长。 \n\n    代码示例：\n        >>> import jittor as jt\n        >>> x = jt.random([2,3,10,10,10])\n        >>> y = jt.nn.pool3d(x, 2, 'max')\n        >>> y.shape \n        [2,3,5,5,5]\n\n    返回值：\n        池化后的输出(Var)。\n\n\n    \n"
    return Pool3d(kernel_size, stride, padding, op=op)(x)

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.layer = Pool(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, op="mean")
    
    def execute(self, x):
        return self.layer(x)

class AvgPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.layer = Pool3d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, op="mean")
    
    def execute(self, x):
        return self.layer(x)

def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    "    \n    这是一个二维平均池化函数。该函数将给定输入进行二维平均池化操作。\n\n     .. math::\n    \n        H_{out}=\\left\\lfloor \\frac{{H_{in} + 2 \\cdot \\text{{padding}} - \\text{{kernel_size}}}}{\\text{{stride}}} \\right\\rfloor + 1\n\n        W_{out} = \\left\\lfloor \\frac{{W_{in} + 2 \\cdot \\text{{padding}} - \\text{{kernel_size}}}}{\\text{{stride}}} \\right\\rfloor + 1\n\n    参数：\n       - x(jt.Var)：输入张量，输入的维度应为四维 :math:`(Batch Size， Channel， Height， Width)` 。\n       - kernel_size (int or tuple)：池化核的大小，如果是一个单独的整数，则长度和宽度都将为该值。若为元组则将按照给定的值作为高和宽。\n       - stride(int or tuple, optional)：步长，该值决定了滑动窗口滑过输入张量的步幅。默认步长等于核大小。如果是一个单独的整数，则长度和宽度都将为该值。若为元组则将按照给定的值作为高和宽。\n       - padding(int, optional)：在输入张量的所有边界上隐式零填充。默认值：0。\n       - ceil_mode(bool, optional)：当设置为True时，会使用 :math:`ceil` 函数计算输出形状。默认值：False。\n       - count_include_pad(bool, optional)：当设置为True时，将在计算平均值时包含零填充。默认值：True。\n   \n     代码示例:\n        >>> import jittor as jt\n        >>> x = jt.random((1, 1, 4, 4))\n        >>> jt.nn.avg_pool2d(x, kernel_size=2, stride=2)\n    返回值：\n        返回张量，其形状取决于输入和平均池化操作。如果输入的形状是 :math:`(N, C, H_{in}, W_{in})` ，那么在默认 :math:`stride` （即kernel size）下，输出的形状将是 :math:`(N, C, H_{out}, W_{out})` ，其中\n \n\n"
    return AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)(x)

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
        self._layer = Pool(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, op="maximum")
    
    def execute(self, x):
        return self._layer(x)

class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
        self._layer = Pool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, op="maximum")
    
    def execute(self, x):
        return self._layer(x)

def max_pool2d(x, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
    "    对输入的多通道二维输入数据进行最大池化计算。\n\n    参数:\n        - x (Tensor): 输入张量，形状为 :math:`(N,C,H,W)` ，其中 :math:`N` 是批大小， :math:`C` 是通道数， :math:`H` 和 :math:`W` 是输入图片的高和宽。\n        - kernel_size (int 或 tuple(int)): 池化窗口的大小。当给出的是一个整数时，池化窗口的高和宽为同一值。也可以直接以 :math:`(H, W)` 的形式给出。\n        - stride (int 或 tuple(int), optional): 池化操作的步长。当给出的是一个整数时，步长在高和宽方向分别为该值。也可以直接以 :math:`(SH, SW)` 的形式给出。如果未给出，默认值 :math:`kernel_size` 。\n        - padding (int 或 tuple(int), optional): 在输入的边界外扩展0值的填充大小。当给出的是一个整数时，填充在高和宽方向分别为该值。也可以直接以 :math:`(PH, PW)` 的形式给出。默认值：0。\n        - dilation (int 或 tuple(int), optional): 窗口内元素之间的间距。当给出的是一个整数时，间距在高和宽方向分别为该值。也可以直接以 :math:`(DH, DW)` 的形式给出。如果未给出，默认值：1。\n        - return_indices (bool， optional): 如果为True，则返回输出最大值的索引; 默认值：None。\n        - ceil_mode (bool， optional): 如果为True，使用 :math:`ceil` 代替 :math:`floor` 计算输出 :math:`shape` 。默认值：False。\n\n    返回:\n        最大池化后的结果(Tensor)。\n\n    代码示例:\n        >>> import jittor as jt\n        >>> x = jt.random((1, 1, 4, 4))\n        >>> jt.nn.max_pool2d(x, kernel_size=2, stride=2)\n       \n"
    return MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)(x)


def max_pool3d(x, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
    "\n对输入张量进行3D最大池化操作。\n\n    对于一个输入张量的形状为 :math:`(N, C, D, H, W)` ，则输出张量的形状计算公式为： \n\n    .. math::\n\n        \\left\\lfloor \\frac{{\\text{input_length} + 2 \\cdot \\text{padding} - \\text{kernel_size}}}{\\text{stride}} \\right\\rfloor + 1\n\n    参数：\n        - x(Var): 输入的张量。 \n        - kernel_size(int or tuple): 池化核大小。\n        - stride(int or tuple, optional): 池化操作步长。默认值： None\n        - padding(int or tuple, optional): 输入的每一条边补充0的层数。默认值：0\n        - dilation( int or tuple, optional): 控制池化窗口中元素步幅，可以是单个整数或（深度，高度，宽度）的元组。默认值： None\n        - return_indices(bool, optional): 是否返回池化结果对应的输入索引，有些情况下用于计算梯度。默认值： None\n        - ceil_mode(bool, optional): 是否使用 :math:`ceil` 函数计算输出大小。如果为 False，则使用 :math:`floor` 函数。默认值： False\n\n    代码示例：\n        >>> import torch\n        >>> x = torch.randn(1, 1, 4, 4, 4)  # 输入张量的形状为 [batch_size, channels, depth, height, width]\n        >>> torch.nn.functional.max_pool3d(x, kernel_size=3, stride=2, padding=1)\n    \n返回值：\n        3D最大池化操作后的输出张量(jt.Var)\n\n\n"
    return MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)(x)

class MaxUnpool2d(Module):
    ''' MaxUnpool2d is the invert version of MaxPool2d with indices.
    It takes the output index of MaxPool2d as input.
    The element will be zero if it is not the max pooled value.

    Example::

    >>> import jittor as jt
    >>> from jittor import nn

    >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool2d(2, stride=2)
    >>> input = jt.array([[[[ 1.,  2,  3,  4,0],
                            [ 5,  6,  7,  8,0],
                            [ 9, 10, 11, 12,0],
                            [13, 14, 15, 16,0],
                            [0,  0,  0,  0, 0]]]])
    >>> output, indices = pool(input)
    >>> unpool(output, indices, output_size=input.shape)
    jt.array([[[[   0.,  0.,   0.,   0.,   0.],
                [   0.,  6.,   0.,   8.,   0.],
                [   0.,  0.,   0.,   0.,   0.],
                [   0., 14.,   0.,  16.,   0.],
                [   0.,  0.,   0.,   0.,   0.]]]])
    '''
    def __init__(self, kernel_size, stride=None):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if stride is None: stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride

    def execute(self, x, id, output_size=None):
        b, c, ph, pw = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        if output_size:
            h, w = output_size[-2:]
        else:
            h, w = ph * sh, pw * sw
        if self.stride == self.kernel_size:
            x = x.reindex(shape=[b, c, h, w], 
                indexes=['i0', 'i1', f'i2/{kh}', f'i3/{kw}'],
                extras=[id], 
                overflow_conditions=[
                    f'(i2*yshape3+i3) != @e0(i0,i1,i2/{kh},i3/{kw})'],
                overflow_value=0)
        else:
            x = x.reindex_reduce(
                op="add",
                shape=[b, c, h, w], 
                indexes=['i0', 'i1', 
                    f'@e0(i0,i1,i2,i3)/xshape3', 
                    f'@e0(i0,i1,i2,i3)%xshape3'],
                extras=[id], 
            )
        return x

class MaxUnpool3d(Module):
    ''' MaxUnpool3d is the invert version of MaxPool3d with indices.
    It takes the output index of MaxPool3d as input.
    The element will be zero if it is not the max pooled value.
    '''
    def __init__(self, kernel_size, stride=None):
        if stride is None: stride = kernel_size
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def execute(self, x, id, output_size=None):
        b, c, pd, ph, pw = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        if output_size:
            d, h, w = output_size[-3:]
        else:
            d, h, w = pd * sd, ph * sh, pw * sw
        if self.stride == self.kernel_size:
            x = x.reindex(shape=[b, c, d, h, w], 
                indexes=['i0', 'i1', f'i2/{kd}', f'i3/{kh}', f'i4/{kw}'],
                extras=[id], 
                overflow_conditions=[
                    f'(i2*yshape3*yshape4+i3*yshape4+i4) != @e0(i0,i1,i2/{kd},i3/{kh},i4/{kw})'],
                overflow_value=0)
        else:
            x = x.reindex_reduce(
                op="add",
                shape=[b, c, d, h, w], 
                indexes=['i0', 'i1', 
                    f'@e0(i0,i1,i2,i3,i4)/(xshape4*xshape3)', 
                    f'@e0(i0,i1,i2,i3,i4)/xshape4%xshape3', 
                    f'@e0(i0,i1,i2,i3,i4)%xshape4'],
                extras=[id], 
            )
        return x
