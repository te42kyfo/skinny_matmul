#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "../gpu_error.cuh"

namespace GENV3X {

template <typename T, int M, int N>
__global__ void deviceReduce(T *blockResults, T *result, T alpha, T beta,
                             int blockCount, int lda, int ldb, int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx >= M * N) return;

  int n = tidx / M;
  int m = tidx % M;

  T sum = 0.0;
  for (int i = 0; i < blockCount; i++) {
    sum += blockResults[i * N * ldc + n * ldc + m];
  }

  result[n * ldc + m] = result[n * ldc + m] * beta + sum * alpha;
}

template <typename T, int M, int N, int BLOCKSIZE, bool TRANSPOSE,
          int THREAD_PER_N>
__global__ void blockProductKernel(const T *A, const T *B, T *out, int K,
                                   int lda, int ldb, int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  const int threadsPerN = N % THREAD_PER_N == 0 ? THREAD_PER_N : 1;
  const int nPerThread = (N / threadsPerN);

  __shared__ T blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  const int m = tidx % M;
  const int nhalf = (tidx / M) % threadsPerN;

  if (blockDim.x * gridDim.x / M / threadsPerN == tidx / M / threadsPerN)
    return;

  T threadSum[nPerThread];
  for (int n = 0; n < nPerThread; n++) {
    threadSum[n] = 0;
  }

  const T *baseB = B + nhalf * nPerThread;
  for (int idx = tidx / M / threadsPerN; idx < K;
       idx += blockDim.x * gridDim.x / M / threadsPerN) {
    for (int n = 0; n < nPerThread; n++) {
      threadSum[n] += A[idx * lda + m] * __ldg(baseB + idx * ldb + n);
    }
  }

  for (int n = 0; n < N / threadsPerN; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M * threadsPerN) {
      T blockSum = 0.0;

      for (int i = threadIdx.x; i < BLOCKSIZE; i += M * threadsPerN) {
        blockSum += blockStorage[i];
      }

      int fullN = nhalf * nPerThread + n;
      if (TRANSPOSE) {
        out[blockIdx.x * M * ldc + m * ldc + fullN] = blockSum;
      } else {
        out[blockIdx.x * N * ldc + fullN * ldc + m] = blockSum;
      }
    }
  }
}

void *d_temp_storage = NULL;

template <typename T, int M, int N, int THREAD_PER_N>
bool tsmttsm(const int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;
  if (d_temp_storage == NULL)
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(dtype) * 100 * 100 * 1000));
  if (blockCount * M * N > 100 * 100 * 1000) return false;

  //  if (N % THREAD_PER_N != 0) return false;

  if (N > M) {
    GENV3X::blockProductKernel<T, N, M, 256, true,
                               THREAD_PER_N><<<blockCount, 256>>>(
        B, A, (T *)d_temp_storage, K, ldb, lda, ldc);
  } else {
    GENV3X::blockProductKernel<T, M, N, 256, false,
                               THREAD_PER_N><<<blockCount, 256>>>(
        A, B, (T *)d_temp_storage, K, lda, ldb, ldc);
  }
  GENV3X::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      (T *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
