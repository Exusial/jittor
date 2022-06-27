// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Wen-Yang Zhou <zhouwy1998@foxmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CudnnFusedConvBnReluOp : Op {
    Var* x, * w, * b, * y;
    int strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
    string xformat, wformat, bformat, yformat;
    /* CudnnConvOp: xformat abcd represents nchw */
    CudnnFusedConvBnReluOp(Var* x, Var* w, Var* b, int strideh, int stridew, int paddingh, int paddingw, int dilationh=1, int dilationw=1, int groups=1, string xformat="abcd", string wformat="oihw", string bformat="abcd", string yformat="");
    
    const char* name() const override { return "cudnn_fused_conv_bn_relu"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
