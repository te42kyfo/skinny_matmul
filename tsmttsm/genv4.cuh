#pragma once

#include <cuda_runtime.h>
#include "../gpu_error.cuh"

namespace GENV4 {

template <typename T, int M, int N>
__global__ void deviceReduce(T *blockResults, T *result, T alpha, T beta,
                             int blockCount, int lda, int ldb, int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= M * N) return;

  int n = tidx / M;
  int m = tidx % M;

  T sum = 0.0;
  for (int i = 0; i < blockCount; i++) {
    sum += blockResults[i * N * M + n * M + m];
  }

  result[n * ldc + m] = result[n * ldc + m] * beta + sum * alpha;
}

enum class MEMPATH { GLOBAL, TEX };

template <typename T, int M, int N, int BLOCKSIZE, MEMPATH mempath>
__global__ void blockProductKernel(const T *A, const T *B, T *out, int K,
                                   int lda, int ldb, int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ T blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int m = tidx % M;
  int n = (tidx / M) % N;

  if (blockDim.x * gridDim.x / M / N == tidx / M / N) return;

  T threadSum = 0;

  const int unrollFactor = 8;
  const int subblockSize = K / unrollFactor;
  int idx = tidx / M / N;

  for (; idx < K / unrollFactor; idx += blockDim.x * gridDim.x / M / N) {
    for (int u = 0; u < unrollFactor; u++) {
      if (mempath == MEMPATH::GLOBAL) {
        threadSum += A[(idx + u * subblockSize) * lda + m] *
                     B[(idx + u * subblockSize) * ldb + n];
      }
      if (mempath == MEMPATH::TEX) {
        threadSum += __ldg(A + (idx + u * subblockSize) * lda + m) *
                     __ldg(B + (idx + u * subblockSize) * ldb + n);
      }
    }
  }

  for (idx = subblockSize * unrollFactor + tidx / M / N; idx < K;
       idx += blockDim.x * gridDim.x / M / N) {
    threadSum += __ldg(A + idx * lda + m) * __ldg(B + idx * ldb + n);
  }

  __syncthreads();
  blockStorage[threadIdx.x] = threadSum;
  __syncthreads();

  if (threadIdx.x < M * N) {
    T blockSum = 0.0;
    for (int i = threadIdx.x; i < BLOCKSIZE; i += M * N) {
      blockSum += blockStorage[i];
    }

    out[blockIdx.x * N * M + n * M + m] = blockSum;
  }
}

void *d_temp_storage = NULL;
size_t temp_storage_size = 0;

template <typename T, int M, int N, MEMPATH mempath>
bool tsmttsm(int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;

  size_t required_temp_storage_size = M * N * blockCount;
  if (temp_storage_size < required_temp_storage_size) {
    //std::cout << "GENV4: Reallocate. Was " << temp_storage_size;
    GPU_ERROR(cudaFree(d_temp_storage));
    temp_storage_size = 3 * required_temp_storage_size;
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(T) * temp_storage_size));
    //std::cout << " is now " << temp_storage_size << "\n";
  }

  if (M * N > blockCount * 256) {
    blockCount = M * N / 256 + 1;
  }

  cudaMemset(d_temp_storage, 0, M * N * blockCount * sizeof(dtype));

  GENV4::blockProductKernel<T, M, N, 256, mempath><<<blockCount, 256>>>(
      A, B, (T *)d_temp_storage, K, lda, ldb, ldc);

  //  GENV4::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
  //    (T *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
