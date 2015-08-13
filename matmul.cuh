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

namespace fourXfour {

__device__ inline double double_shfl_xor(double var, unsigned int srcLane,
                                         int width = 32) {
  int2 a = *reinterpret_cast<int2 *>(&var);
  a.x = __shfl_xor(a.x, srcLane, width);
  a.y = __shfl_xor(a.y, srcLane, width);
  return *reinterpret_cast<double *>(&a);
}

template <int BLOCKSIZE>
__global__ void blockProductKernel(double *A, double *B, double *out,
                                   size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double threadSum1 = 0;
  double threadSum2 = 0;
  double threadSum3 = 0;
  double threadSum4 = 0;

  __shared__ double transposeBuffer[BLOCKSIZE];

  for (size_t kidx = tidx; kidx < K * 4; kidx += blockDim.x * gridDim.x) {
    transposeBuffer[threadIdx.x] =
        B[kidx / 4 + (threadIdx.x / 64) * K + threadIdx.x % 64];

    double a = A[kidx];
    threadSum1 += a * transposeBuffer[threadIdx.x / 4];
    threadSum2 += a * transposeBuffer[threadIdx.x / 4 + 64];
    threadSum3 += a * transposeBuffer[threadIdx.x / 4 + 128];
    threadSum4 += a * transposeBuffer[threadIdx.x / 4 + 192];
  }

  if (tidx == 0) {
    out[blockIdx.x * 16 + threadIdx.x * 4 + 1] =
        threadSum1 + threadSum2 + threadSum3 + threadSum4;
    //    out[blockIdx.x * 16 + threadIdx.x * 4 + 2] = threadSum2;
    // out[blockIdx.x * 16 + threadIdx.x * 4 + 3] = threadSum3;
    // out[blockIdx.x * 16 + threadIdx.x * 4 + 4] = threadSum4;
  }
  /*
  for (int i = 16; i >= 4; i /= 2) {
    threadSum1 += double_shfl_xor(threadSum1, i, 32);
    threadSum2 += double_shfl_xor(threadSum2, i, 32);
    threadSum3 += double_shfl_xor(threadSum3, i, 32);
    threadSum4 += double_shfl_xor(threadSum4, i, 32);
  }
  if (warpLaneId < 4) {
    warpSums[(threadIdx.x / 32) * 16 + warpLaneId * 4 + 0] = threadSum1;
    warpSums[(threadIdx.x / 32) * 16 + warpLaneId * 4 + 1] = threadSum2;
    warpSums[(threadIdx.x / 32) * 16 + warpLaneId * 4 + 2] = threadSum3;
    warpSums[(threadIdx.x / 32) * 16 + warpLaneId * 4 + 3] = threadSum4;
  }

  __syncthreads();
  double blockSum = 0;
  if (threadIdx.x < 16) {
    for (int i = 0; i < 8; i++) {
      blockSum += warpSums[i * 16 + threadIdx.x];
    }

    out[blockIdx.x * 16 + threadIdx.x] = blockSum;
  }*/
}

void fourXfour(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
               double *B, double *result, const size_t M, const size_t N,
               const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    temp_storage_bytes = blockCount * sizeof(double);
    return;
  }

  fourXfour::blockProductKernel<256> << <blockCount, 256>>>
      (A, B, d_temp_storage, K);
}
}

void matmul(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
            double *B, double *result, const size_t M, const size_t N,
            const size_t K, const int blockCount) {
  if (M == 1 && N == 1) {
    oneXone::oneXone(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K,
                     blockCount);
    return;
  }
  if (M == 2 && N == 1) {
    twoXone::twoXone(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K,
                     blockCount);
    return;
  }
  if (M == 2 && N == 2) {
    twoXtwo::twoXtwo(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K,
                     blockCount);
    return;
  }

  if (M == 4 && N == 4) {
    fourXfour::fourXfour(temp_storage_bytes, d_temp_storage, A, B, result, M, N,
                         K, blockCount);
    return;
  }
  std::cout << "No matmul variant for " << M << "xKx" << N << "\n";
  exit(1);
}
