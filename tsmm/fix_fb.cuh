#pragma once
#include "../eq.cuh"

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_fix_fb_kernel(const T *__restrict__ A,
                                          const T *__restrict__ B, T *out,
                                          const int K, const int lda,
                                          const int ldb, const int ldc, T alpha,
                                          T beta) {
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int n = tidx % N;

  if (tidx / N == gridDim.x * BLOCKSIZE / N && !BETAISZERO) return;

  for (int row = tidx / N; row < K; row += gridDim.x * BLOCKSIZE / N) {
    T sum;
    zero(sum);
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, A[row * lda + m], B[n * ldb + m]);
    }
    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum);
    } else {
      out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
    }
  }
}

template <typename T, int M, int N>
bool tsmm_fix_fb(const int blockCount, const int varM, const int varN,
                 const int K, const T *A, const int lda, const T alpha,
                 const T *B, const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N || A == C) return false;

  const int BLOCKSIZE = 256;
  T Tzero;
  zero(Tzero);
  if (eq(beta, Tzero)) {
    tsmm_fix_fb_kernel<T, M, N, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc, alpha, beta);
  } else {
    tsmm_fix_fb_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc, alpha, beta);
  }
  return true;
}
