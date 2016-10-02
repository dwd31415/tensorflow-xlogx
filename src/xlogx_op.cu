#include "settings.h"

__global__ void cuda_op_function(const float *in, const int N, float* out){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        if(in[i] == 0.0f) {
            out[i] = 0.0f;
        } else {
            out[i] = logf(in[i]) * in[i];
        }
    }
}

bool cuda_op_launcher(const float *in, const int N, float* out){
    cudaGetLastError();
    cuda_op_function<<<32,256>>>(in, N, out);
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }else{
        return true;
    }
}

__global__ void cuda_op_gradient(const float *in, const int N, float* out){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
#if FIX_DERIVATIVE_FROM_0_TO_EPSILON
        if (in[i] < EPSILON){
#else
        if (in[i] == 0){
#endif
            out[i] = 0.0f;
        } else {
            out[i] = 1 + logf(in[i]);
        }
    }
}

bool cuda_grad_op_launcher(const float *in, const int N, float* out){
    cudaGetLastError();
    cuda_op_gradient<<<32,256>>>(in, N, out);
    if (cudaGetLastError() != cudaSuccess)
    {
        return false;
    }else{
        return true;
    }
}
