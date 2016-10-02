#include "tensorflow/core/framework/op.h"

REGISTER_OP("XLogXOp")
.Input("in: float32")
.Output("series: float32");

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ExampleOp : public OpKernel {
public:
    explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}

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
      for (int i = 1; i < N; i++) {
          if (input(i) == -1){
              output(i) = input(i);
          } else{
              output(i) = (float)i;
          }
      }
  }
};

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

void cuda_op_launcher(const float *in, const int N, float* out);

class ExampleOpGPU : public OpKernel {
public:
    explicit ExampleOpGPU(OpKernelConstruction* context) : OpKernel(context) {}

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
        cuda_op_launcher(input.data(), N, output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("XLogXOp").Device(DEVICE_GPU), ExampleOpGPU);
#endif
REGISTER_KERNEL_BUILDER(Name("XLogXOp").Device(DEVICE_CPU), ExampleOp);
