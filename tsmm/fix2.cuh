#pragma once
#include "../eq.cuh"
#include "../warpReduce.cuh"

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
          gval = axpy(gval, A[row * lda + m], B[n * ldb + m]);
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
  if (varM != M || varN != N || A == C) return false;

  const int BLOCKSIZE = 256;

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
