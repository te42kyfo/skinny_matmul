#pragma once

#include "cub.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace GENV4 {

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
  int n = (tidx / M) % N;

  if (blockDim.x * gridDim.x / M / N == tidx / M / N) return;

  double threadSum;
  threadSum = 0;

  for (size_t idx = tidx / M / N; idx < K;
       idx += blockDim.x * gridDim.x / M / N) {
    threadSum += A[idx * lda + m] * B[idx * ldb + n];
  }

  __syncthreads();
  blockStorage[threadIdx.x] = threadSum;
  __syncthreads();

  if (threadIdx.x < M * N) {
    double blockSum = 0.0;
    for (int i = threadIdx.x; i < BLOCKSIZE; i += M * N) {
      blockSum += blockStorage[i];
    }
    if (TRANSPOSE) {
      out[blockIdx.x * N * ldc + n * ldc + m] = blockSum;
    } else {
      out[blockIdx.x * M * ldc + m * ldc + n] = blockSum;
    }
  }
}

template <int M, int N>
void matmul(size_t &temp_storage_bytes, double *d_temp_storage,
            size_t blockCount, const int K, const double alpha, const double *A,
            const int lda, const double *B, const int ldb, const double beta,
            double *C, const int ldc) {
  if (M * N > blockCount * 256) {
    blockCount = M * N / 256 + 1;
  }

  if (temp_storage_bytes == 0) {
    temp_storage_bytes = blockCount * sizeof(double) * M * ldc;
    return;
  }
  cudaMemset(d_temp_storage, 0, temp_storage_bytes);

  if (N > M) {
    GENV4::blockProductKernel<N, M, 256, true><<<blockCount, 256>>>(
        B, A, d_temp_storage, K, ldb, lda, ldc);
  } else {
    GENV4::blockProductKernel<M, N, 256, false><<<blockCount, 256>>>(
        A, B, d_temp_storage, K, lda, ldb, ldc);
  }

  GENV4::deviceReduce<M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
}
}
