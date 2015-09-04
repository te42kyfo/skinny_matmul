#pragma once

#include "../cub/cub.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace oneXone {

template <int BLOCKSIZE>
__global__ void blockScalarProductKernel(double *A, double *B, double *out,
                                         size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double threadSum = 0.0;
  for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    threadSum += A[idx] * B[idx];
  }

  typedef cub::BlockReduce<double, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double blockSum = BlockReduce(temp_storage).Sum(threadSum);

  if (threadIdx.x == 0) out[blockIdx.x] = blockSum;
}

void oneXone(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
             double *B, double *result, const size_t M, const size_t N,
             const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_storage,
                           result, blockCount);
    temp_storage_bytes += blockCount * sizeof(double);
    return;
  }
  oneXone::blockScalarProductKernel<256> << <blockCount, 256>>>
      (A, B, d_temp_storage, K);
  cub::DeviceReduce::Sum(d_temp_storage + blockCount * sizeof(double),
                         temp_storage_bytes, d_temp_storage, result,
                         blockCount);
}
}

namespace twoXone {

template <int BLOCKSIZE>
__global__ void blockScalarProductKernel(double *A, double *B, double *out,
                                         size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double threadSum1 = 0.0;
  double threadSum2 = 0.0;

  for (size_t idx = tidx; idx < K; idx += blockDim.x * gridDim.x) {
    threadSum1 += A[idx * 2] * B[idx];
    threadSum2 += A[idx * 2 + 1] * B[idx];
  }

  typedef cub::BlockReduce<double, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;

  double blockSum1 = BlockReduce(temp_storage1).Sum(threadSum1);
  double blockSum2 = BlockReduce(temp_storage2).Sum(threadSum2);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = blockSum1;
    out[blockIdx.x + gridDim.x] = blockSum2;
  }
}

void twoXone(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
             double *B, double *result, const size_t M, const size_t N,
             const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_storage,
                           result, blockCount);
    temp_storage_bytes = (temp_storage_bytes + blockCount * sizeof(double)) * 2;
    return;
  }
  twoXone::blockScalarProductKernel<256> << <blockCount, 256>>>
      (A, B, d_temp_storage, K);
  cub::DeviceReduce::Sum(d_temp_storage + 2 * blockCount, temp_storage_bytes,
                         d_temp_storage, result, blockCount);
  cub::DeviceReduce::Sum(d_temp_storage + 2 * blockCount, temp_storage_bytes,
                         d_temp_storage + blockCount, result + 1, blockCount);
}
}

namespace twoXtwo {

template <int BLOCKSIZE>
__global__ void blockScalarProductKernel(double *A, double *B, double *out,
                                         size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double threadSum1 = 0.0;
  double threadSum2 = 0.0;
  double threadSum3 = 0.0;
  double threadSum4 = 0.0;

  for (size_t idx = tidx; idx < K; idx += blockDim.x * gridDim.x) {
    threadSum1 += A[idx * 2] * B[idx];
    threadSum2 += A[idx * 2 + 1] * B[idx];
    threadSum3 += A[idx * 2] * B[idx + K];
    threadSum4 += A[idx * 2 + 1] * B[idx + K];
  }

  typedef cub::BlockReduce<double, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage1;

  double blockSum1;
  double blockSum2;
  double blockSum3;
  double blockSum4;

  blockSum1 = BlockReduce(temp_storage1).Sum(threadSum1);
  __syncthreads();
  blockSum2 = BlockReduce(temp_storage1).Sum(threadSum2);
  __syncthreads();
  blockSum3 = BlockReduce(temp_storage1).Sum(threadSum3);
  __syncthreads();
  blockSum4 = BlockReduce(temp_storage1).Sum(threadSum4);
  __syncthreads();

  if (threadIdx.x == 0) {
    out[blockIdx.x + gridDim.x * 0] = blockSum1;
    out[blockIdx.x + gridDim.x * 1] = blockSum2;
    out[blockIdx.x + gridDim.x * 2] = blockSum3;
    out[blockIdx.x + gridDim.x * 3] = blockSum4;
  }
}

void twoXtwo(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
             double *B, double *result, const size_t M, const size_t N,
             const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_storage,
                           result, blockCount);
    temp_storage_bytes =
        (temp_storage_bytes + blockCount * sizeof(double)) * M * N;
    return;
  }
  twoXtwo::blockScalarProductKernel<256> << <blockCount, 256>>>
      (A, B, d_temp_storage, K);
  cub::DeviceReduce::Sum(d_temp_storage + 4 * blockCount, temp_storage_bytes,
                         d_temp_storage + blockCount * 0, result + 0,
                         blockCount);
  cub::DeviceReduce::Sum(d_temp_storage + 4 * blockCount, temp_storage_bytes,
                         d_temp_storage + blockCount * 1, result + 1,
                         blockCount);
  cub::DeviceReduce::Sum(d_temp_storage + 4 * blockCount, temp_storage_bytes,
                         d_temp_storage + blockCount * 2, result + 2,
                         blockCount);
  cub::DeviceReduce::Sum(d_temp_storage + 4 * blockCount, temp_storage_bytes,
                         d_temp_storage + blockCount * 3, result + 3,
                         blockCount);
}
}

namespace MXN {

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

  double threadSum[M][N];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      threadSum[m][n] = 0.0;
    }
  }

  for (size_t idx = tidx; idx < K; idx += blockDim.x * gridDim.x) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        threadSum[m][n] += A[idx * M + m] * B[idx + K * n];
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
        out[blockIdx.x + gridDim.x * (n * M + m)] = blockSum[m][n];
      }
    }
  }
}

template <int M, int N>
void MXN(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
         double *B, double *result, const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_storage,
                           result, blockCount);
    temp_storage_bytes =
        (temp_storage_bytes + blockCount * sizeof(double)) * M * N;
    return;
  }
  MXN::blockProductKernel<M, N, 256> << <blockCount, 256>>>
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
