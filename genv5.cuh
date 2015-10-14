#pragma once

#include "cub.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace GENV5 {

__device__ inline double double_shfl_xor(double var, unsigned int srcLane,
                                         int width = 32) {
  int2 a = *reinterpret_cast<int2 *>(&var);
  a.x = __shfl_xor(a.x, srcLane, width);
  a.y = __shfl_xor(a.y, srcLane, width);
  return *reinterpret_cast<double *>(&a);
}

template <int M, int N>
__global__ void deviceReduce(double *blockResults, double *result,
                             int blockCount) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= M * N) return;

  int n = tidx / M;
  int m = tidx % M;

  double sum = 0;
  for (int i = 0; i < blockCount; i++) {
    sum += blockResults[i * M * N + n * M + m];
  }

  result[n * M + m] = sum;
}

template <int M, int N, int BLOCKSIZE, bool TRANSPOSE>
__global__ void blockProductKernel(double *A, double *B, double *out,
                                   size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ double blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int warpId = tidx / 32;
  int warpLane = tidx % 32;

  double threadSum[N];
  for (int n = 0; n < N; n++) {
    threadSum[n] = 0;
  }

  for (size_t k = warpId; k < K; k += blockDim.x * gridDim.x / 32) {
    for (int n = 0; n < N; n++) {
      threadSum[n] += A[k * M + warpLane] * B[k * N + n];
    }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      double blockSum = 0.0;
      for (int i = threadIdx.x; i < BLOCKSIZE; i += 32) {
        blockSum += blockStorage[i];
      }
      out[blockIdx.x * M * N + n * M + warpLane] = blockSum;
    }
  }
}

template <int M, int N>
void matmul(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
            double *B, double *result, const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    temp_storage_bytes = blockCount * sizeof(double) * M * N;
    return;
  }

  GENV5::blockProductKernel<M, N, 256, false><<<blockCount, 256>>>(
      A, B, d_temp_storage, K);

  GENV5::deviceReduce<M, N><<<M * N / 256 + 1, 256>>>(d_temp_storage, result,
                                                      blockCount);
}
}
