// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Wen-Yang Zhou <zhouwy1998@foxmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cudnn_fused_conv_bn_relu_op.h"
#include "cudnn_wrapper.h"
#include "executor.h"
#include "ops/op_register.h"

using namespace std;

namespace jittor {

extern int use_tensorcore;

static inline int findc(const char* format, const char& c) {
    if (c==format[0]) return 0;
    if (c==format[1]) return 1;
    if (c==format[2]) return 2;
    ASSERT(c==format[3]) << "Not a valid format" << format << c;
    return 3;
}

#ifndef JIT

static inline void get_shape(Var* x, const char* f, const string& format, int& a, int& b, int &c, int& d) {
    auto& shape = x->shape;
    a = shape[findc(format.c_str(), f[0])];
    b = shape[findc(format.c_str(), f[1])];
    c = shape[findc(format.c_str(), f[2])];
    d = shape[findc(format.c_str(), f[3])];
}

static inline void set_shape(Var* x, const char* f, const string& format, int a, int b, int c, int d) {
    int64 shape[4];
    shape[findc(format.c_str(), f[0])] = a;
    shape[findc(format.c_str(), f[1])] = b;
    shape[findc(format.c_str(), f[2])] = c;
    shape[findc(format.c_str(), f[3])] = d;
    x->set_shape(NanoVector(
        shape[0], shape[1], shape[2], shape[3]));
}

CudnnFusedConvBnReluOp::CudnnFusedConvBnReluOp(Var* x, Var* w, Var* b, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string bformat, string yformat)
    : x(x), w(w), b(b), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), bformat(move(bformat)), yformat(move(yformat)) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_manual_set_vnbb);
    x->flags.set(NodeFlags::_needed_by_backward);
    w->flags.set(NodeFlags::_needed_by_backward);
    y = create_output(nullptr, dtype_infer(x->ns, w->ns));
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

void CudnnFusedConvBnReluOp::infer_shape() {
    ASSERTop(x->shape.size(),==,4);
    ASSERTop(w->shape.size(),==,4);
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    get_shape(w, "oihw", wformat, wco, wci, wh, ww);
    ASSERTop(wci * groups,==,xc);
    yn = xn, yc = wco;
    yh = (xh+paddingh*2-wh*dilationh+dilationh-1)/strideh+1;
    yw = (xw+paddingw*2-ww*dilationw+dilationw-1)/stridew+1;
    set_shape(y, "abcd", yformat, yn, yc, yh, yw);
}

void CudnnFusedConvBnReluOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«Tw:" << w->dtype();
    jk << "«Tb:" << b->dtype();
    jk << "«XFORMAT:" << xformat;
    jk << "«WFORMAT:" << wformat;
    jk << "«BFORMAT:" << bformat;
    jk << "«YFORMAT:" << yformat;
}
static auto make_backwardx = get_op_info("cudnn_conv_backward_x")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string>();
static auto make_backwardw = get_op_info("cudnn_conv_backward_w")
    .get_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int, int, int, string, string, string>();
VarPtr CudnnFusedConvBnReluOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    if (xformat == "ncdhw")
        x->shape.unpack(xn, xc, xh, xw);
    else
        x->shape.unpack(xn, xh, xw, xc);
    w->shape.unpack(wco, wci, wh, ww);
    if (v_index == 0) {
        return make_backwardx(w, dout, xh, xw, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    } else {
        return make_backwardw(x, dout, wh, ww, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    }
}

unordered_map<string, cudnnConvolutionFwdAlgo_t> fwd_algo_cache1;

#else // JIT
#ifdef JIT_cuda

#pragma clang diagnostic ignored "-Wtautological-compare"

EXTERN_LIB unordered_map<string, cudnnConvolutionFwdAlgo_t> fwd_algo_cache1;

template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnFusedConvBnReluOp::jit_run() {
    cudnnHandle_t& handle_ = cudnn_handle;

    cudnnTensorDescriptor_t cudnnIdesc;
    cudnnTensorDescriptor_t cudnnBdesc;
    cudnnFilterDescriptor_t cudnnFdesc;
    cudnnTensorDescriptor_t cudnnOdesc;
    cudnnConvolutionDescriptor_t cudnnConvDesc;
    cudnnActivationDescriptor_t  activationDesc;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
    
    checkCudaErrors(cudnnCreateTensorDescriptor( &cudnnIdesc ));
    checkCudaErrors(cudnnCreateTensorDescriptor( &cudnnBdesc ));
    checkCudaErrors(cudnnCreateFilterDescriptor( &cudnnFdesc ));
    checkCudaErrors(cudnnCreateTensorDescriptor( &cudnnOdesc ));
    checkCudaErrors(cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));
    checkCudaErrors(cudnnSetConvolutionGroupCount( cudnnConvDesc, groups ));
    checkCudaErrors(cudnnCreateActivationDescriptor( &activationDesc ));
    checkCudaErrors(cudnnSetActivationDescriptor( activationDesc, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0 ));

    int dimX[] = {
        (int)x->shape[findc("@XFORMAT", 'a')], // n
        (int)x->shape[findc("@XFORMAT", 'b')], // c
        (int)x->shape[findc("@XFORMAT", 'c')], // h
        (int)x->shape[findc("@XFORMAT", 'd')], // w
    };
    checkCudaErrors(cudnnSetTensor4dDescriptor(
        cudnnIdesc, CUDNN_TENSOR_NCHW, getDataType<Tx>(),
        dimX[0], dimX[1], dimX[2], dimX[3]
    ));

    auto bs = b->shape;
    int dimB[] = {(int)bs[0],(int)bs[1],(int)bs[2],(int)bs[3]};
    checkCudaErrors(cudnnSetTensor4dDescriptor(
        cudnnBdesc, CUDNN_TENSOR_NCHW, getDataType<Tb>(),
        dimB[0], dimB[1], dimB[2], dimB[3]
    ));

    auto ws = w->shape;
    int dimW[] = {(int)ws[0],(int)ws[1],(int)ws[2],(int)ws[3]};
    checkCudaErrors(cudnnSetFilter4dDescriptor(
        cudnnFdesc, getDataType<Tw>(), CUDNN_TENSOR_NCHW,
        dimW[0], dimW[1], dimW[2], dimW[3]
    ));

    int padA[] = {paddingh, paddingw};
    int convstrideA[] = {strideh, stridew};
    int dilationA[] = {dilationh, dilationw};
    // difference between
    // CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION
    // is the kernel rc order
    // currently, No perf difference is observed between
    // this two mode
    checkCudaErrors(cudnnSetConvolution2dDescriptor(
        cudnnConvDesc,
        paddingh, paddingw, strideh, stridew, dilationh, dilationw,
        CUDNN_CROSS_CORRELATION, getDataType<Ty>()
    ));

    // using tensor core
    if(use_tensorcore){
        checkCudaErrors( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) );
    }
    
    if (x->dtype() == ns_float16
        || y->dtype() == ns_float16 || w->dtype() == ns_float16) {
        checkCudaErrors( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) );
    }

    int dimY[] = {
        (int)y->shape[findc("@YFORMAT", 'a')], // n
        (int)y->shape[findc("@YFORMAT", 'b')], // c
        (int)y->shape[findc("@YFORMAT", 'c')], // h
        (int)y->shape[findc("@YFORMAT", 'd')], // w
    };
    checkCudaErrors( cudnnSetTensor4dDescriptor(
        cudnnOdesc, CUDNN_TENSOR_NCHW, getDataType<Ty>(),
        dimY[0], dimY[1], dimY[2], dimY[3]
    ) );

    cudnnConvolutionFwdAlgo_t algo;
    checkCudaErrors( cudnnGetConvolutionForwardAlgorithm(
        handle_,
        cudnnIdesc, cudnnFdesc, cudnnConvDesc, cudnnOdesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo) );
    // TODO: warp work space
    void *workSpace = 0;
    size_t workSpaceSize = 0;
    checkCudaErrors (cudnnGetConvolutionForwardWorkspaceSize(
        handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, 
        cudnnOdesc, algo, &workSpaceSize) );
    size_t allocation;
    if (workSpaceSize > 0) {
        workSpace = exe.temp_allocator->alloc(workSpaceSize, allocation);
    }

    float alpha1=1, alpha2=0, beta=0;
    checkCudaErrors(cudnnConvolutionBiasActivationForward(
        handle_,
        (void*)(&alpha1), cudnnIdesc, x->ptr<Tx>(),
        cudnnFdesc, w->ptr<Tw>(),
        cudnnConvDesc,
        algo,
        workSpace, workSpaceSize,
        (void*)(&alpha2), cudnnOdesc, y->ptr<Ty>(),
        cudnnBdesc, b->ptr<Tb>(),
        activationDesc,
        cudnnOdesc, y->ptr<Ty>())
    );
    if (workSpace)
        exe.temp_allocator->free(workSpace, workSpaceSize, allocation);

    checkCudaErrors(cudnnDestroyTensorDescriptor( cudnnIdesc ));
    checkCudaErrors(cudnnDestroyFilterDescriptor( cudnnFdesc ));
    checkCudaErrors(cudnnDestroyTensorDescriptor( cudnnBdesc ));
    checkCudaErrors(cudnnDestroyTensorDescriptor( cudnnOdesc ));
    checkCudaErrors(cudnnDestroyConvolutionDescriptor( cudnnConvDesc ));
}
#endif
#endif // JIT

} // jittor
