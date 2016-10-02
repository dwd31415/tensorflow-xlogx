__global__ void cuda_op_function(const float *in, const int N, float* out){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        out[i] = (float)(2*i) + 1.0f;
        if(in[i] == -1.0f){
            out[i] = in[i];
        }
    }
}

void cuda_op_launcher(const float *in, const int N, float* out){
    cuda_op_function<<<32,256>>>(in, N, out);
}
