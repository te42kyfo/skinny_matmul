#pragma once
#include "../eq.cuh"

template <typename T, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_var1_kernel(const T *A, const T *B, T *out,
                                        const int M, const int N, const int K,
                                        const int lda, const int ldb,
                                        const int ldc, T alpha, T beta) {
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int n = tidx % N;

  if (tidx / N == gridDim.x * BLOCKSIZE / N && !BETAISZERO) return;

  for (int row = tidx / N; row < K; row += gridDim.x * BLOCKSIZE / N) {
    T sum;
    zero(sum);
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, A[row * lda + m], __ldg(B + n * ldb + m));
    }
    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum);
    } else {
      out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
    }
  }
}

template <typename T>
bool tsmm_var1(const size_t blockCount, const int varM, const int varN,
               const int K, const T *A, const int lda, const T alpha,
               const T *B, const int ldb, const T beta, T *C, const int ldc) {
  if (A == C) return false;
  if (blockCount == 0) return true;
  const int BLOCKSIZE = 256;

  T Tzero;
  zero(Tzero);
  if (eq(beta, Tzero)) {
    tsmm_var1_kernel<T, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
        A, B, C, varM, varN, K, lda, ldb, ldc, alpha, beta);
  } else {
    tsmm_var1_kernel<T, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
        A, B, C, varM, varN, K, lda, ldb, ldc, alpha, beta);
  }
  return true;
}
