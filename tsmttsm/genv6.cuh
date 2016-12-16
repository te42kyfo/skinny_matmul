#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "../gpu_error.cuh"

namespace GENV6 {

template <typename T, int M, int N>
__global__ void deviceReduce(T *blockResults, T *result, T alpha, T beta,
                             int blockCount, int lda, int ldb, int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= M * N) return;

  int n = tidx / M;
  int m = tidx % M;

  T sum = 0.0;
  for (int i = 0; i < blockCount; i++) {
    sum += blockResults[i * N * M + n * M + m];
  }

  result[n * ldc + m] = result[n * ldc + m] * beta + sum * alpha;
}

template <typename T, int M, int N, int BLOCKSIZE, bool TRANSPOSE>
__global__ void blockProductKernel(const T *A, const T *B, T *out, int K,
                                   int lda, int ldb, int ldc) {
  const int rowsPerBlock = BLOCKSIZE / M;
  int m = threadIdx.x % M;
  int n = threadIdx.x / M;

  __shared__ T ACache[BLOCKSIZE];
  __shared__ T BCache[N * rowsPerBlock];

  ACache[threadIdx.x] = 0.0;
  if (threadIdx.x < N * rowsPerBlock) BCache[threadIdx.x] = 0.0;

  T threadSum = 0;

  int idx;
  int inc = gridDim.x * rowsPerBlock;
  for (idx = blockIdx.x * rowsPerBlock; idx < K - rowsPerBlock + 1;
       idx += inc) {
    __syncthreads();
    ACache[threadIdx.x] = A[(idx + n) * lda + m];
    //    for (int i = threadIdx.x; i < rowsPerBlock * N; i += BLOCKSIZE)
    BCache[threadIdx.x] = B[(idx + n) * ldb + m];

    __syncthreads();
    //    if (m < M && n < N)
    for (int i = 0; i < BLOCKSIZE / M; i++) {
      threadSum += ACache[i * M + m] * BCache[i * N + n];
    }
  }
  // Remainder loop
  for (; idx < K; idx++) {
    if (m < M && n < N) threadSum += A[idx * lda + m] * B[idx * ldb + n];
  }

  if (m < M && n < N) out[blockIdx.x * N * M + n * M + m] = threadSum;
}

void *d_temp_storage = NULL;

template <typename T, int M, int N>
bool tsmttsm(const int blockCount, const int varM, const int varN, int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;
  if (d_temp_storage == NULL)
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(dtype) * 100 * 100 * 1000));
  if (blockCount * M * N > 100 * 100 * 1000) return false;

  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  const int blockSize = (1024 / M) * M;

  GENV6::blockProductKernel<T, M, N, blockSize,
                            false><<<blockCount, blockSize>>>(
      A, B, (T *)d_temp_storage, K, lda, ldb, ldc);

  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault);

  GENV6::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      (T *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
