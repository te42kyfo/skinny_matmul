#pragma once
#include "../eq.cuh"

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

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_fix2_kernel(const T *__restrict__ A,
                                        const T *__restrict__ B,
                                        T *__restrict__ out, const int K,
                                        const int lda, const int ldb,
                                        const int ldc, T alpha, T beta) {
  const int GANGSIZE = 32 / sizeof(T);

  int gId = threadIdx.x % GANGSIZE;
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;

  for (int row = tidx / GANGSIZE; row < K;
       row += gridDim.x * BLOCKSIZE / GANGSIZE) {
    for (int n = 0; n < N; n++) {
      T gval;
      zero(gval);
      for (int i = 0; i < (M - 1) / GANGSIZE + 1; i++) {
        int m = i * GANGSIZE + gId;
        if (m < M || M % GANGSIZE == 0)
          gval = axpy(gval, A[row * lda + m], B[m * ldb + n]);
      }
      if (BETAISZERO) {
        out[row * ldc + n] = scale(alpha, warpReduce(gval, GANGSIZE));
      } else {
        out[row * ldc + n] =
            axpby(warpReduce(gval, GANGSIZE), out[row * ldc + n], alpha, beta);
      }
    }
  }
}

template <typename T, int M, int N>
bool tsmm_fix2(const size_t blockCount, const int varM, const int varN,
               const int K, const T *A, const int lda, const T alpha,
               const T *B, const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;
  if (blockCount == 0) return true;
  const int BLOCKSIZE = 256;

  if (M >= 32 / sizeof(T)) {
    T Tzero;
    zero(Tzero);
    if (eq(beta, Tzero)) {
      tsmm_fix2_kernel<T, M, N, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
          A, B, C, K, lda, ldb, ldc, alpha, beta);
    } else {
      tsmm_fix2_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
          A, B, C, K, lda, ldb, ldc, alpha, beta);
    }
    return true;
  }

  return false;
}
