#pragma once

#include "cub.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace GENV3 {

template <int M, int N>
__global__ void deviceReduce(double *blockResults, double *result, double alpha,
                             double beta, int blockCount, size_t lda,
                             size_t ldb, size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= M * N) return;

  int n = tidx / M;
  int m = tidx % M;

  double sum = 0.0;
  for (int i = 0; i < blockCount; i++) {
    sum += blockResults[i * M * ldc + m * ldc + n];
  }

  result[m * ldc + n] = result[m * ldc + n] * beta + sum * alpha;
}

template <int M, int N, int BLOCKSIZE, bool TRANSPOSE>
__global__ void blockProductKernel(const double *A, const double *B,
                                   double *out, size_t K, size_t lda,
                                   size_t ldb, size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ double blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int m = tidx % M;

  if (blockDim.x * gridDim.x / M == tidx / M) return;

  double threadSum[N];
  for (int n = 0; n < N; n++) {
    threadSum[n] = 0;
  }

  for (size_t idx = tidx / M; idx < K; idx += blockDim.x * gridDim.x / M) {
    for (int n = 0; n < N; n++) {
      threadSum[n] += A[idx * lda + m] * B[idx * ldb + n];
    }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      double blockSum = 0.0;
      for (int i = threadIdx.x; i < BLOCKSIZE; i += M) {
        blockSum += blockStorage[i];
      }
      if (TRANSPOSE) {
        out[blockIdx.x * N * ldc + n * ldc + m] = blockSum;
      } else {
        out[blockIdx.x * M * ldc + m * ldc + n] = blockSum;
      }
    }
  }
}

template <int M, int N>
void matmul(size_t &temp_storage_bytes, double *d_temp_storage,
            const size_t blockCount, const int K, const double alpha,
            const double *A, const int lda, const double *B, const int ldb,
            const double beta, double *C, const int ldc) {
  if (temp_storage_bytes == 0) {
    temp_storage_bytes = blockCount * sizeof(double) * M * ldc;
    return;
  }
  if (N > M) {
    GENV3::blockProductKernel<N, M, 256, true><<<blockCount, 256>>>(
        B, A, d_temp_storage, K, ldb, lda, ldc);
  } else {
    GENV3::blockProductKernel<M, N, 256, false><<<blockCount, 256>>>(
        A, B, d_temp_storage, K, lda, ldb, ldc);
  }
  GENV3::deviceReduce<M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
}
}
