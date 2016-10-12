#pragma once
#include "../eq.cuh"

template <typename T, int REMAINDER>
static __global__ void __launch_bounds__(256, 4)
    tsmm_varip3_kernel(const T *A, const T *B, T *out, const int M, const int N,
                       const int K, const int lda, const int ldb, const int ldc,
                       T alpha, T beta) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = tidx % N;

  int row = tidx / N;
  if (row > K) return;
  //  for (int row = tidx / N; row < K; row += gridDim.x * blockDim.x / N) {
  T sum;
  zero(sum);

  int offset1 = (n + 0) % 4;
  int offset2 = (n + 1) % 4;
  int offset3 = (n + 2) % 4;
  int offset4 = (n + 3) % 4;

  const T *Abase = A + row * lda;
  const T *Bbase = B + n * ldb;
  for (int m = 0; m < M - 3; m += 4) {
    sum = axpy(sum, __ldg(Abase + m + offset1), __ldg(Bbase + m + offset1));
    sum = axpy(sum, __ldg(Abase + m + offset2), __ldg(Bbase + m + offset2));
    sum = axpy(sum, __ldg(Abase + m + offset3), __ldg(Bbase + m + offset3));
    sum = axpy(sum, __ldg(Abase + m + offset4), __ldg(Bbase + m + offset4));
  }
  for (int i = 0; i < REMAINDER; i++) {
    sum = axpy(sum, __ldg(Abase + M - REMAINDER + i),
               __ldg(Bbase + M - REMAINDER + i));
  }

  __syncthreads();

  out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
}

template <typename T>
bool tsmm_varip3(const size_t blockCount, const int varM, const int varN,
                 const int K, const T *A, const int lda, const T alpha,
                 const T *B, const int ldb, const T beta, T *C, const int ldc) {
  //  if (varM % 4 != 0) return false;
  const int threadsPerBlock = (256 / varN) * varN;
  int newBlockCount = (K * varN / threadsPerBlock / 13 + 1) * 13;

  if (varM % 4 == 0)
    tsmm_varip3_kernel<T, 0><<<newBlockCount, threadsPerBlock>>>(
        A, B, C, varM, varN, K, lda, ldb, ldc, alpha, beta);

  if (varM % 4 == 1)
    tsmm_varip3_kernel<T, 1><<<newBlockCount, threadsPerBlock>>>(
        A, B, C, varM, varN, K, lda, ldb, ldc, alpha, beta);

  if (varM % 4 == 2)
    tsmm_varip3_kernel<T, 2><<<newBlockCount, threadsPerBlock>>>(
        A, B, C, varM, varN, K, lda, ldb, ldc, alpha, beta);

  if (varM % 4 == 3)
    tsmm_varip3_kernel<T, 3><<<newBlockCount, threadsPerBlock>>>(
        A, B, C, varM, varN, K, lda, ldb, ldc, alpha, beta);

  return true;
}
