#pragma once
#include "../eq.cuh"

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void __launch_bounds__(BLOCKSIZE)
    tsmm_fix_fb_kernel(const T *__restrict__ A, const T *__restrict__ B, T *out,
                       const int K, const int lda, const int ldb, const int ldc,
                       T alpha, T beta) {
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int n = tidx % N;

  __shared__ T bCache[M][N];
  for (int mn = threadIdx.x; mn < M * N; mn += BLOCKSIZE) {
    int tn = mn / M;
    int tm = mn % M;
    bCache[tm][tn] = B[tn * ldb + tm];
  }

  __syncthreads();

  if (tidx / N == gridDim.x * BLOCKSIZE / N && !BETAISZERO) return;

  int row = tidx / N;
  for (; row < K / 2; row += gridDim.x * BLOCKSIZE / N) {
    T sum1;
    zero(sum1);
    T sum2;
    zero(sum2);

#pragma unroll(M <= 8 ? M : 4)
    for (int m = 0; m < M; m++) {
      sum1 = axpy(sum1, A[row * lda + m], bCache[m][n]);
      sum2 = axpy(sum2, A[(row + K / 2) * lda + m], bCache[m][n]);
    }
    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum1);
      out[(row + K / 2) * ldc + n] = scale(alpha, sum2);
    } else {
      out[row * ldc + n] = axpby(sum1, out[row * ldc + n], alpha, beta);
      out[(row + K / 2) * ldc + n] =
          axpby(sum2, out[(row + K / 2) * ldc + n], alpha, beta);
    }
  }

  // remainder loop
  for (row += K / 2; row < K; row += gridDim.x * BLOCKSIZE / N) {
    T sum;
    zero(sum);

#pragma unroll(M <= 8 ? M : 4)
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, A[row * lda + m], bCache[m][n]);
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

  const int BLOCKSIZE = ((1 << 15) / (M * N * 8) < 3) ? 1024 : 512;
  T Tzero;
  zero(Tzero);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  if (eq(beta, Tzero)) {
    tsmm_fix_fb_kernel<T, M, N, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc, alpha, beta);
  } else {
    tsmm_fix_fb_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc, alpha, beta);
  }
  return true;
}
