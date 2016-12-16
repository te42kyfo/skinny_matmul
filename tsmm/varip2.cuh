#pragma once
#include "../eq.cuh"

template <typename T, bool BETAISZERO>
static __global__ void __launch_bounds__(256, 8)
    tsmm_varip2_kernel(const T *A, const T *B, T *out, const int M, const int N,
                       const int K, const int lda, const int ldb, const int ldc,
                       T alpha, T beta) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = tidx % N;

  int row = tidx / N;
  if (row >= K) return;
  //  for (int row = tidx / N; row < K; row += gridDim.x * blockDim.x / N) {
  T sum;
  zero(sum);
#pragma unroll 4
  for (int m = 0; m < M; m++) {
    sum = axpy(sum, __ldg(A + row * lda + m), __ldg(B + n * ldb + m));
  }

  __syncthreads();

  if (BETAISZERO) {
    out[row * ldc + n] = scale(alpha, sum);
  } else {
    out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
  }
  //}
}

template <typename T>
bool tsmm_varip2(const size_t blockCount, const int varM, const int varN,
                 const int K, const T *A, const int lda, const T alpha,
                 const T *B, const int ldb, const T beta, T *C, const int ldc) {
  const int threadsPerBlock = (256 / varN) * varN;
  int newBlockCount = K * varN / threadsPerBlock + 1;

  T Tzero;
  zero(Tzero);
  if (eq(beta, Tzero)) {
    tsmm_varip2_kernel<T, true><<<newBlockCount, threadsPerBlock>>>(
        A, B, C, varM, varN, K, lda, ldb, ldc, alpha, beta);
  } else {
    tsmm_varip2_kernel<T, false><<<newBlockCount, threadsPerBlock>>>(
        A, B, C, varM, varN, K, lda, ldb, ldc, alpha, beta);
  }
  return true;
}
