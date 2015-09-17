#pragma once

#include "cub.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace GENV2 {

__device__ inline double double_shfl_xor(double var, unsigned int srcLane,
                                         int width = 32) {
  int2 a = *reinterpret_cast<int2 *>(&var);
  a.x = __shfl_xor(a.x, srcLane, width);
  a.y = __shfl_xor(a.y, srcLane, width);
  return *reinterpret_cast<double *>(&a);
}

template <int M, int N, int BLOCKSIZE>
__global__ void blockProductKernel(double *A, double *B, double *out,
                                   size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ double blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int m = tidx % M;

  if (blockDim.x * gridDim.x / M == tidx / M) return;

  double threadSum[N];
  for (int n = 0; n < N; n++) {
    threadSum[n] = 0.0;
  }

  for (size_t idx = tidx / M; idx < K; idx += blockDim.x * gridDim.x / M) {
    for (int n = 0; n < N; n++) {
      threadSum[n] += A[idx * M + m] * B[idx + K * n];
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
      out[blockIdx.x + gridDim.x * (n * M + m)] = blockSum;
    }
  }
}

template <int M, int N>
void matmul(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
            double *B, double *result, const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_storage,
                           result, blockCount);
    temp_storage_bytes =
        (temp_storage_bytes + blockCount * sizeof(double)) * M * N;
    return;
  }
  GENV2::blockProductKernel<M, N, 256> << <blockCount, 256>>>
      (A, B, d_temp_storage, K);

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      cub::DeviceReduce::Sum(d_temp_storage + (M * N) * blockCount,
                             temp_storage_bytes,
                             d_temp_storage + blockCount * (n * M + m),
                             result + (n * M + m), blockCount);
    }
  }
}
}
