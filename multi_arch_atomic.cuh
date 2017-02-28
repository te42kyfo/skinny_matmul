

template <typename T>
__device__ T multiArchAtomicAdd(T* address, int* tag, T value);

template <>
__device__ float multiArchAtomicAdd(float* address, int* tag, float value) {
  return atomicAdd(address, value);
}


// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
__device__ double multiArchAtomicAdd(double* address, int* tag, double val) {
#if __CUDA_ARCH__ > 600
  return atomicAdd(address, val);
#else

  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);

#endif
}

template <>
__device__ PseudoQuad multiArchAtomicAdd(PseudoQuad* address, int* tag,
                                         PseudoQuad value) {
  while (0 == atomicCAS(tag, 0, 1))
    ;

  PseudoQuad oldValue = *address;

  *address = accu(oldValue, value);

  __threadfence();

  *tag = 0;
  return oldValue;
}
