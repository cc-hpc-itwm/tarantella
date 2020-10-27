#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

__global__ void AddOneKernel(const float* in1,const float* in2, const int N, float* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    out[i] = in1[i] + 2*in2[i];
  }
}

void AddOneKernelLauncher(const float* in1,const float* in2, const int N, float* out) {
  TF_CHECK_OK(::tensorflow::GpuLaunchKernel(AddOneKernel, 32, 256, 0, nullptr,
                                            in1,in2, N, out));
}

#endif
