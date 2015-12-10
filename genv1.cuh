#pragma once

#include "cub.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace GENV1 {

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

template <int M, int N, int BLOCKSIZE>
__global__ void blockProductKernel(const double *A, const double *B,
                                   double *out, size_t K, size_t lda,
                                   size_t ldb, size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double threadSum[M][N];

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

  typedef cub::BlockReduce<double, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double blockSum[M][N];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      blockSum[m][n] = BlockReduce(temp_storage).Sum(threadSum[m][n]);
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        out[blockIdx.x * M * ldc + m * ldc + n] = blockSum[m][n];
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
    cudaFuncSetCacheConfig(blockProductKernel<M, N, 256>,
                           cudaFuncCachePreferL1);
    temp_storage_bytes = blockCount * sizeof(double) * M * ldc;
    return;
  }
  GENV1::blockProductKernel<M, N, 256><<<blockCount, 256>>>(
      A, B, d_temp_storage, K, lda, ldb, ldc);

  GENV1::deviceReduce<M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
}
}
