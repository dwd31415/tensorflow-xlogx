#include "tensorflow/core/framework/op.h"

REGISTER_OP("XLogXGradOp")
.Input("in: float32")
.Output("series: float32");

#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>
#include "settings.h"

using namespace tensorflow;

class XLogXGradOp : public OpKernel {
public:
    explicit XLogXGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Get input tensor
      const Tensor& input_tensor = context->input(0);
      auto input = input_tensor.flat<float>();
      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
      auto output = output_tensor->flat<float>();
      const int N = input.size();
      for (int i = 0; i < N; i++) {
#if FIX_DERIVATIVE_FROM_0_TO_EPSILON
          if (input(i) < EPSILON){
#else
          if (input(i) == 0){
#endif
              // This is not smooth, but there is no limit (that is a number) of log(x)+1 as x -> 0, but the derivative would have to be negative.
              output(i) = -1;
          } else {
              output(i) = 1 + logf(input(i));
          }
#if THROW_ERROR_FOR_NEGATIVE_LOGS_ON_CPU
          if if(input(i) < 0){
              context->CtxFailureWithWarning(errors::InvalidArgument("x*log(x) is only defined for x>= 0"));
              return;
          }
#endif
      }
  }
};

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

bool cuda_grad_op_launcher(const float *in, const int N, float* out);

class XLogXGradOpGPU : public OpKernel {
public:
    explicit XLogXGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Get input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<float>();
        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output = output_tensor->flat<float>();
        const int N = input.size();
        if (!cuda_grad_op_launcher(input.data(), N, output.data())){
            context->CtxFailureWithWarning(errors::Internal("FATAL CUDA ERROR"));
            return;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("XLogXGradOp").Device(DEVICE_GPU), XLogXGradOpGPU);
#endif
REGISTER_KERNEL_BUILDER(Name("XLogXGradOp").Device(DEVICE_CPU), XLogXGradOp);
