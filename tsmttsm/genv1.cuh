#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "../../cub/cub.cuh"

namespace GENV1 {

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

  T threadSum[M][N];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      threadSum[m][n] = 0.0;
    }
  }

  for (size_t idx = tidx; idx < K; idx += blockDim.x * gridDim.x) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        threadSum[m][n] += A[idx * lda + m] * B[idx * ldb + n];
      }
    }
  }

  typedef cub::BlockReduce<T, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T blockSum[M][N];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      blockSum[m][n] = BlockReduce(temp_storage).Sum(threadSum[m][n]);
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        out[blockIdx.x * N * ldc + n * ldc + m] = blockSum[m][n];
      }
    }
  }
}

void *d_temp_storage = NULL;

template <typename T, int M, int N>
bool tsmttsm(const int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;

  if (d_temp_storage == NULL)
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(dtype) * 100 * 100 * 1000));
  if (blockCount * M * N > 100 * 100 * 1000) return false;

  GENV1::blockProductKernel<T, M, N, 256><<<blockCount, 256>>>(
      A, B, (T *)d_temp_storage, K, lda, ldb, ldc);

  GENV1::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      (T *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
