#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "../gpu_error.cuh"

namespace GENV3 {

template <typename T, typename iT, int M, int N>
__global__ void deviceReduce(iT *blockResults, T *result, T alpha, T beta,
                             int blockCount, int lda, int ldb, int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= M * N) return;

  int n = tidx / M;
  int m = tidx % M;

  iT sum;
  zero(sum);
  for (int i = 0; i < blockCount; i++) {
    sum = accu(sum, blockResults[i * N * M + n * M + m]);
  }

  result[n * ldc + m] = accu(scale(result[n * ldc + m], beta),
                             convert<iT, T>(scale2(renormalize(sum), alpha)));
}

enum class MEMPATH { GLOBAL, TEX };

template <typename T, typename iT, int M, int N, int BLOCKSIZE, bool TRANSPOSE,
          MEMPATH BLOAD>
__global__ void blockProductKernel(const T *A, const T *B, iT *out, const int K,
                                   const int lda, const int ldb,
                                   const int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ iT blockStorage[BLOCKSIZE];

  zero(blockStorage[threadIdx.x]);

  int m = tidx % M;

  if (blockDim.x * gridDim.x / M == tidx / M) return;

  iT threadSum[N];
  for (int n = 0; n < N; n++) {
    zero(threadSum[n]);
  }

  for (int idx = tidx / M; idx < K; idx += blockDim.x * gridDim.x / M) {
    for (int n = 0; n < N; n++) {
      if (BLOAD == MEMPATH::GLOBAL) {
        threadSum[n] = axpy2(threadSum[n], A[idx * lda + m], B[idx * ldb + n]);
      } else if (BLOAD == MEMPATH::TEX) {
        threadSum[n] =
            axpy2(threadSum[n], A[idx * lda + m], __ldg(B + idx * ldb + n));
      }
    }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      iT blockSum;
      zero(blockSum);
      for (int i = threadIdx.x; i < BLOCKSIZE; i += M) {
        blockSum = accu(blockSum, blockStorage[i]);
      }
      if (TRANSPOSE) {
        out[blockIdx.x * M * N + m * N + n] = blockSum;
      } else {
        out[blockIdx.x * N * M + n * M + m] = blockSum;
      }
    }
  }
}

void *d_temp_storage = NULL;
size_t temp_storage_size = 0;

template <typename T, typename iT, int M, int N, MEMPATH BLOAD>
bool tsmttsm(const int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;

  size_t required_temp_storage_size = M * N * blockCount;
  if (temp_storage_size < required_temp_storage_size) {
    std::cout << "GENV3: Reallocate. Was " << temp_storage_size;
    GPU_ERROR(cudaFree(d_temp_storage));
    temp_storage_size = 3 * required_temp_storage_size;
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(iT) * temp_storage_size));
    std::cout << " is now " << temp_storage_size << "\n";
  }

  if (N > M) {
    GENV3::blockProductKernel<T, iT, N, M, 256, true,
                              BLOAD><<<blockCount, 256>>>(
        B, A, (iT *)d_temp_storage, K, ldb, lda, ldc);
  } else {
    GENV3::blockProductKernel<T, iT, M, N, 256, false,
                              BLOAD><<<blockCount, 256>>>(
        A, B, (iT *)d_temp_storage, K, lda, ldb, ldc);
  }
  //  GENV3::deviceReduce<T, iT, M, N><<<M * N / 256 + 1, 256>>>(
  //    (iT *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
