#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "cu_complex.h"

namespace SPECSMALL {

template <typename T, int M, int N>
__global__ void deviceReduce(T *blockResults, T *result, T alpha, T beta,
                             int blockCount, size_t lda, size_t ldb,
                             size_t ldc) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= M * N) return;
  int n = tidx / M;
  int m = tidx % M;

  T sum;
  zero(sum);
  for (int i = 0; i < blockCount; i++) {
    sum = accu(sum, blockResults[i * N * ldc + n * ldc + m]);
  }

  result[n * ldc + m] = axpby(result[n * ldc + m], sum, beta, alpha);
}

template <typename T, int M, int N, int BLOCKSIZE, bool TRANSPOSE, bool SELF>
__global__ void blockProductKernel(const T *A, const T *B, T *out, const int K,
                                   const int lda, const int ldb,
                                   const int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int warpLane = threadIdx.x % 32;
  int rowsPerWarp = 32 / M;
  int m = warpLane % M;

  if (warpLane >= rowsPerWarp * M) {
    warpLane = rowsPerWarp * M - 1;
    m = warpLane % M;
  }

  __shared__ T blockStorage[BLOCKSIZE];

  zero(blockStorage[threadIdx.x]);

  T threadSum[N];
  for (int n = 0; n < N; n++) {
    zero(threadSum[n]);
  }

  for (int idx = (tidx / 32) * rowsPerWarp + warpLane / M; idx < K;
       idx += blockDim.x * gridDim.x / 32 * rowsPerWarp) {
    T av = A[idx * lda + m];
    if (!SELF) {
      blockStorage[threadIdx.x] = B[idx * ldb + m];
    } else {
      blockStorage[threadIdx.x] = av;
    }
    int localAddress = threadIdx.x - m;
    for (int n = 0; n < N; n++) {
      threadSum[n] = axpy(threadSum[n], av, blockStorage[localAddress + n]);
    }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      T blockSum;
      zero(blockSum);
      for (int w = 0; w < BLOCKSIZE / 32; w++) {
        for (int wp = threadIdx.x; wp < rowsPerWarp * M; wp += M) {
          blockSum = accu(blockSum, blockStorage[w * 32 + wp]);
        }
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
void matmul(size_t &temp_storage_bytes, T *d_temp_storage,
            const size_t blockCount, const int K, const T alpha, const T *A,
            const int lda, const T *B, const int ldb, const T beta, T *C,
            const int ldc) {
  if (temp_storage_bytes == 0) {
    temp_storage_bytes = blockCount * sizeof(T) * N * ldc;
    return;
  }

  if (M > 32 || N > 32) {
    std::cerr << "This Kernel cannot be instanciated for M,N > 32\n";
    return;
  }

  int const blocksize = 256;

  if (N > M) {
    SPECSMALL::blockProductKernel<T, N, M, blocksize, true,
                                  false><<<blockCount, blocksize>>>(
        B, A, d_temp_storage, K, ldb, lda, ldc);

  } else {
    if (M == N && A == B) {
      SPECSMALL::blockProductKernel<T, M, N, blocksize, false,
                                    true><<<blockCount, blocksize>>>(
          A, B, d_temp_storage, K, lda, ldb, ldc);
    } else {
      SPECSMALL::blockProductKernel<T, M, N, blocksize, false,
                                    false><<<blockCount, blocksize>>>(
          A, B, d_temp_storage, K, lda, ldb, ldc);
    }
  }
  SPECSMALL::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
}
}
