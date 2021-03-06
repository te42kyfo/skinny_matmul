#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "../../cub/cub.cuh"

namespace GENV1 {

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
    sum += blockResults[i * N * M + n * M + m];
  }

  result[n * ldc + m] = result[n * ldc + m] * beta + sum * alpha;
}

enum class MEMPATH { GLOBAL, TEX };

template <typename T, int M, int N, int BLOCKSIZE, MEMPATH mempath>
__global__ void blockProductKernel(const T *A, const T *B, T *out, size_t K,
                                   size_t lda, size_t ldb, size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  T threadSum[M][N];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      threadSum[m][n] = 0.0;
    }
  }

  for (size_t idx = tidx; idx < K; idx += blockDim.x * gridDim.x) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        if (mempath == MEMPATH::GLOBAL) {
          threadSum[m][n] += A[idx * lda + m] * B[idx * ldb + n];
        }
        if (mempath == MEMPATH::TEX) {
          threadSum[m][n] +=
              __ldg(A + idx * lda + m) * __ldg(B + idx * ldb + n);
        }
      }
    }
  }

  typedef cub::BlockReduce<T, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T blockSum[M][N];

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      blockSum[m][n] = BlockReduce(temp_storage).Sum(threadSum[m][n]);
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        out[blockIdx.x * N * M + n * M + m] = blockSum[m][n];
      }
    }
  }
}

void *d_temp_storage = NULL;
size_t temp_storage_size = 0;

template <typename T, int M, int N, MEMPATH mempath>
bool tsmttsm(const int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;

  size_t required_temp_storage_size = M * N * blockCount;
  if (temp_storage_size < required_temp_storage_size) {
    std::cout << "GENV1: Reallocate. Was " << temp_storage_size;
    GPU_ERROR(cudaFree(d_temp_storage));
    temp_storage_size = 3 * required_temp_storage_size;
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(T) * temp_storage_size));
    std::cout << " is now " << temp_storage_size << "\n";
  }

  GENV1::blockProductKernel<T, M, N, 256, mempath><<<blockCount, 256>>>(
      A, B, (T *)d_temp_storage, K, lda, ldb, ldc);

  //  GENV1::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
  //    (T *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
