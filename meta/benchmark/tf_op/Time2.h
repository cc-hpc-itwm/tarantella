#ifndef TIME2_H_
#define TIME2_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
template <typename Eigen::GpuDevice, typename T>
struct ExampleFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif TIME2_H_