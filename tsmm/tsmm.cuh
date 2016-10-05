

#include "../cu_complex.h"
#include "../eq.cuh"
#include "cublas.cuh"
#include "fix1.cuh"
#include "fix2.cuh"
#include "fix_fb.cuh"


template <typename T>
__device__ inline T __shfl_xor_t(T var, unsigned int srcLane, int width = 32) {
  int *a = reinterpret_cast<int *>(&var);
  for (int i = 0; i < sizeof(T) / 4; i++) {
    a[i] = __shfl_xor(a[i], srcLane, width);
  }
  return *reinterpret_cast<T *>(a);
}

template <typename T>
__device__ inline T warpReduce(T lval, int width) {
  for (int offset = width / 2; offset > 0; offset /= 2) {
    lval = accu(lval, __shfl_xor_t(lval, offset, width));
  }
  return lval;
}

template <>
bool tsmm<double, 0, 0>(const size_t blockCount, const int K,
                        const double alpha, const double *A, const int lda,
                        const double *B, const int ldb, const double beta,
                        double *C, const int ldc) {
  std::cout << "not implemented\n";
  return false;
}
