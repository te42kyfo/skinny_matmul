#pragma once

#include "cub.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace GENV5 {

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
__global__ void blockProductKernel(const T *A, const T *B, T *out, int K,
                                   const int lda, const int ldb,
                                   const int ldc) {
  int const warpsPerColumn = (M - 1) / 32 + 1;
  int const columnsPerBlock = BLOCKSIZE / 32 / warpsPerColumn;

  int warpLane = threadIdx.x % 32;
  int warpId = threadIdx.x / 32;

  __shared__ T rowCache[BLOCKSIZE / 32][N];
  __shared__ T blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int m = threadIdx.x % (warpsPerColumn * 32);

  if (m > M) m = 0;

  T threadSum[N];
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
      T blockSum = 0.0;
      for (int i = threadIdx.x; i < columnsPerBlock * warpsPerColumn * 32;
           i += 32 * warpsPerColumn) {
        blockSum += blockStorage[i];
      }
      if (TRANSPOSE) {
        out[blockIdx.x * M * ldc + m * ldc + n] = blockSum;
      } else {
        out[blockIdx.x * N * ldc + n * ldc + m] = blockSum;
      }
    }
  }
}

template <typename T, int M, int N>
void matmul(size_t &temp_storage_bytes, T *d_temp_storage, const int blockCount,
            const int K, const T alpha, const T *A, const int lda, const T *B,
            const int ldb, const T beta, T *C, const int ldc) {
  if (temp_storage_bytes == 0) {
    temp_storage_bytes = blockCount * sizeof(T) * N * ldc;
    return;
  }

  if (N > M) {
    GENV5::blockProductKernel<T, N, M, 256, true><<<blockCount, 256>>>(
        B, A, d_temp_storage, K, ldb, lda, ldc);
  } else {
    GENV5::blockProductKernel<T, M, N, 256, false><<<blockCount, 256>>>(
        A, B, d_temp_storage, K, lda, ldb, ldc);
  }
  GENV5::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
}
}
