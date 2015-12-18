#pragma once

#include "cub.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace GENV5 {

template <int M, int N>
__global__ void deviceReduce(double *blockResults, double *result, double alpha,
                             double beta, int blockCount, int lda, int ldb,
                             int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

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
                                   double *out, int K, const int lda,
                                   const int ldb, const int ldc) {
  int const warpsPerColumn = (M - 1) / 32 + 1;
  int const columnsPerBlock = BLOCKSIZE / 32 / warpsPerColumn;

  int warpLane = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  __shared__ double rowCache[BLOCKSIZE / 32][N];
  __shared__ double blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int m = threadIdx.x % (warpsPerColumn * 32);

  if (m > M) m = 0;

  double threadSum[N];
  for (int n = 0; n < N; n++) {
    threadSum[n] = 0;
  }

  for (size_t k =
           blockIdx.x * columnsPerBlock + threadIdx.x / warpsPerColumn / 32;
       k < K; k += gridDim.x * columnsPerBlock) {
    // Split of N=1 as a special case, because warp divergence can lead to
    // hazards in the row cache
    if (N > 1) {
      for (int n = warpLane; n < N; n += 32) {
        rowCache[warpId][n] = B[k * ldb + n];
      }
      for (int n = 0; n < N; n++) {
        threadSum[n] += A[k * lda + m] * rowCache[warpId][n];
      }
    } else {
      for (int n = 0; n < N; n++) {
        threadSum[n] += A[k * lda + m] * B[k * ldb + n];
      }
    }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      double blockSum = 0.0;
      for (int i = threadIdx.x; i < columnsPerBlock * warpsPerColumn * 32;
           i += 32 * warpsPerColumn) {
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
            const int blockCount, const int K, const double alpha,
            const double *A, const int lda, const double *B, const int ldb,
            const double beta, double *C, const int ldc) {
  if (temp_storage_bytes == 0) {
    temp_storage_bytes = blockCount * sizeof(double) * M * ldc;
    return;
  }

  if (N > M) {
    GENV5::blockProductKernel<N, M, 256, true><<<blockCount, 256>>>(
        B, A, d_temp_storage, K, ldb, lda, ldc);
  } else {
    GENV5::blockProductKernel<M, N, 256, false><<<blockCount, 256>>>(
        A, B, d_temp_storage, K, lda, ldb, ldc);
  }
  GENV5::deviceReduce<M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
}
}
