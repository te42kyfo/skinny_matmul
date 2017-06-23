#pragma once
#include "../eq.cuh"

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_fix4_kernel(const T *__restrict__ A,
                                        const T *__restrict__ B,
                                        T *__restrict__ out, const int K,
                                        const int lda, const int ldb,
                                        const int ldc, T alpha, T beta) {
  const int rowsPerBlock = BLOCKSIZE / N;
  int n = threadIdx.x % N;
  int localRow = threadIdx.x / N;

  __shared__ T rowCache[BLOCKSIZE];

  int idx = blockIdx.x * rowsPerBlock;
  // Block synchronous loop
  for (; idx < K - rowsPerBlock; idx += gridDim.x * rowsPerBlock) {
    int row = idx + localRow;

    __syncthreads();
    rowCache[threadIdx.x] = A[row * lda + n];
    __syncthreads();

    T sum = 0;

    for (int m = 0; m < M; m++) {
      sum = axpy(sum, rowCache[localRow * N + m], B[n * ldb + m]);
    }
    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum);
    } else {
      out[row * ldc + n] = axpby(sum, __ldg(out + row * ldc + n), alpha, beta);
    }
  }

  // Remainder Loop
  for (int row = idx + localRow; row < K; row += gridDim.x * rowsPerBlock) {
    T sum = 0;
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, A[row * lda + m], B[n * ldb + m]);
    }
    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum);
    } else {
      out[row * ldc + n] = axpby(sum, __ldg(out + row * ldc + n), alpha, beta);
    }
  }
}

template <typename T, int M, int N>
bool tsmm_fix4(size_t blockCount, const int varM, const int varN, const int K,
               const T *A, const int lda, const T alpha, const T *B,
               const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N || A == C) return false;

  const int BLOCKSIZE = (256 / N) * N;

  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaFuncSetCacheConfig(tsmm_fix4_kernel<T, M, N, BLOCKSIZE, true>,
                         cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(tsmm_fix4_kernel<T, M, N, BLOCKSIZE, false>,
                         cudaFuncCachePreferShared);

  T Tzero;
  zero(Tzero);
  if (eq(beta, Tzero)) {
    tsmm_fix4_kernel<T, M, N, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc, alpha, beta);
  } else {
    tsmm_fix4_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc, alpha, beta);
  }
  return true;
}
