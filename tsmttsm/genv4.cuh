#pragma once

#include "cub.cuh"
#include <cuda_runtime.h>

namespace GENV4 {

template <typename T, int M, int N>
__global__ void deviceReduce(T *blockResults, T *result, T alpha, T beta,
                             int blockCount, size_t lda, size_t ldb,
                             size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= M * N) return;

  int n = tidx / M;
  int m = tidx % M;

  T sum = 0.0;
  for (int i = 0; i < blockCount; i++) {
    sum += blockResults[i * N * ldc + n * ldc + m];
  }

  result[n * ldc + m] = result[n * ldc + m] * beta + sum * alpha;
}

template <typename T, int M, int N, int BLOCKSIZE, bool TRANSPOSE>
__global__ void blockProductKernel(const T *A, const T *B, T *out, size_t K,
                                   size_t lda, size_t ldb, size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ T blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int m = tidx % M;
  int n = (tidx / M) % N;

  if (blockDim.x * gridDim.x / M / N == tidx / M / N) return;

  T threadSum = 0;

  for (size_t idx = tidx / M / N; idx < K;
       idx += blockDim.x * gridDim.x / M / N) {
    threadSum += A[idx * lda + m] * B[idx * ldb + n];
  }

  __syncthreads();
  blockStorage[threadIdx.x] = threadSum;
  __syncthreads();

  if (threadIdx.x < M * N) {
    T blockSum = 0.0;
    for (int i = threadIdx.x; i < BLOCKSIZE; i += M * N) {
      blockSum += blockStorage[i];
    }

    out[blockIdx.x * N * ldc + n * ldc + m] = blockSum;
  }
}

template <typename T, int M, int N>
void matmul(size_t &temp_storage_bytes, T *d_temp_storage, size_t blockCount,
            const int K, const T alpha, const T *A, const int lda, const T *B,
            const int ldb, const T beta, T *C, const int ldc) {
  if (M * N > blockCount * 256) {
    blockCount = M * N / 256 + 1;
  }

  if (temp_storage_bytes == 0) {
    temp_storage_bytes = blockCount * sizeof(T) * N * ldc;
    return;
  }
  cudaMemset(d_temp_storage, 0, temp_storage_bytes);

  GENV4::blockProductKernel<T, M, N, 256, false><<<blockCount, 256>>>(
      A, B, d_temp_storage, K, lda, ldb, ldc);

  GENV4::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
}
}
