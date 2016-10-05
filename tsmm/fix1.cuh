#pragma once
#include "../eq.cuh"

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_fix1_kernel(const T *A, const T *__restrict__ B,
                                        T *out, const int K, const int lda,
                                        const int ldb, const int ldc, T alpha,
                                        T beta) {
  int warpLane = threadIdx.x % 32;
  const int rowsPerWarp = 32 / N;
  const int n = warpLane % N;

  if (warpLane >= rowsPerWarp * N) {
    warpLane = rowsPerWarp * N - 1;
  }
  const int localRow = threadIdx.x / 32 * rowsPerWarp + warpLane / N;

  T __shared__ rowCache[(M / N <= 16) ? M * BLOCKSIZE / 32 * rowsPerWarp : 1];

  for (int row = blockIdx.x * BLOCKSIZE / 32 * rowsPerWarp + localRow; row < K;
       row += BLOCKSIZE * gridDim.x / 32 * rowsPerWarp) {
    for (int i = 0; i < M / N; i++) {
      rowCache[localRow * M + n + i * N] = A[row * lda + n + i * N];
    }

    T sum;
    zero(sum);
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, rowCache[localRow * M + m], B[m * ldb + n]);
    }
    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum);
    } else {
      out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
    }
  }
}

template <typename T, int M, int N>
bool tsmm_fix1(const size_t blockCount, const int varM, const int varN,
               const int K, const T *A, const int lda, const T alpha,
               const T *B, const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;
  if (blockCount == 0) return true;
  const int BLOCKSIZE = 256;

  if (M % N == 0 && M / N <= 16) {
    T Tzero;
    zero(Tzero);
    if (eq(beta, Tzero)) {
      tsmm_fix1_kernel<T, M, N, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
          A, B, C, K, lda, ldb, ldc, alpha, beta);
    } else {
      tsmm_fix1_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
          A, B, C, K, lda, ldb, ldc, alpha, beta);
    }
    return true;
  }

  return false;
}
