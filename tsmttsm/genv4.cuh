#pragma once

#include <cuda_runtime.h>
#include "../gpu_error.cuh"

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

template <typename T, int M, int N, int BLOCKSIZE>
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

void *d_temp_storage = NULL;

template <typename T, int M, int N>
bool tsmttsm(int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;
  if (d_temp_storage == NULL)
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(dtype) * 100 * 100 * 1000));
  if (blockCount * M * N > 100 * 100 * 1000) return false;

  if (M * N > blockCount * 256) {
    blockCount = M * N / 256 + 1;
  }

  cudaMemset(d_temp_storage, 0, 100 * 100 * 1000 * sizeof(dtype));

  GENV4::blockProductKernel<T, M, N, 256><<<blockCount, 256>>>(
      A, B, (T *)d_temp_storage, K, lda, ldb, ldc);

  GENV4::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      (T *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
