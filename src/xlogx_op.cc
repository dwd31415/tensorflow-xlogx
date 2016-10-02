#include "tensorflow/core/framework/op.h"

REGISTER_OP("XLogXOp")
.Input("in: float32")
.Output("series: float32");

#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>
#define THROW_ERROR_FOR_NEGATIVE_LOGS_ON_CPU 0

using namespace tensorflow;

class XLogXOp : public OpKernel {
public:
    explicit XLogXOp(OpKernelConstruction* context) : OpKernel(context) {}

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
          if (input(i) == 0){
              output(i) = 0;
          } else {
              output(i) = input(i) * logf(input(i));
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

bool cuda_op_launcher(const float *in, const int N, float* out);

class XLogXOpGPU : public OpKernel {
public:
    explicit XLogXOpGPU(OpKernelConstruction* context) : OpKernel(context) {}

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
        if (!cuda_op_launcher(input.data(), N, output.data())){
            context->CtxFailureWithWarning(errors::Internal("FATAL CUDA ERROR"));
            return;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("XLogXOp").Device(DEVICE_GPU), XLogXOpGPU);
#endif
REGISTER_KERNEL_BUILDER(Name("XLogXOp").Device(DEVICE_CPU), XLogXOp);
