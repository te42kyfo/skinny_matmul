#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "../cu_complex.h"

namespace SPECSMALL {

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
    sum = accu(sum, blockResults[i * N * ldc + n * ldc + m]);
  }

  result[n * ldc + m] = accu(scale(result[n * ldc + m], beta),
                             convert<iT, T>(scale2(renormalize(sum), alpha)));
}

template <typename T, typename iT, int M, int N, int BLOCKSIZE, bool TRANSPOSE,
          bool SELF>
__global__ void blockProductKernel(const T *A, const T *B, iT *out, const int K,
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

  __shared__ iT blockStorage[BLOCKSIZE];
  T *rowCache = reinterpret_cast<T *>(blockStorage);

  zero(blockStorage[threadIdx.x]);
  __syncthreads();

  iT threadSum[N];
  for (int n = 0; n < N; n++) {
    zero(threadSum[n]);
  }

  for (int idx = (tidx / 32) * rowsPerWarp + warpLane / M; idx < K;
       idx += blockDim.x * gridDim.x / 32 * rowsPerWarp) {
    T av = A[idx * lda + m];
    if (!SELF) {
      rowCache[threadIdx.x] = B[idx * ldb + m];
    } else {
      rowCache[threadIdx.x] = av;
    }

    int localAddress = threadIdx.x - m;
    for (int n = 0; n < N; n++) {
      threadSum[n] = axpy2(threadSum[n], av, rowCache[localAddress + n]);
    }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      iT blockSum;
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

void *d_temp_storage = NULL;

template <typename T, typename iT, int M, int N>
bool tsmttsm(const int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;
  if (varM > 32 || varN > 32) return false;
  if (d_temp_storage == NULL)
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(iT) * 100 * 100 * 1000));
  if (blockCount * M * N > 100 * 100 * 1000) return false;

  int const blocksize = 256;

  if (N > M) {
    SPECSMALL::blockProductKernel<T, iT, N, M, blocksize, true,
                                  false><<<blockCount, blocksize>>>(
        B, A, (iT *)d_temp_storage, K, ldb, lda, ldc);

  } else {
    if (M == N && A == B) {
      SPECSMALL::blockProductKernel<T, iT, M, N, blocksize, false,
                                    true><<<blockCount, blocksize>>>(
          A, B, (iT *)d_temp_storage, K, lda, ldb, ldc);
    } else {
      SPECSMALL::blockProductKernel<T, iT, M, N, blocksize, false,
                                    false><<<blockCount, blocksize>>>(
          A, B, (iT *)d_temp_storage, K, lda, ldb, ldc);
    }
  }
  SPECSMALL::deviceReduce<T, iT, M, N><<<M * N / 256 + 1, 256>>>(
      (iT *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
