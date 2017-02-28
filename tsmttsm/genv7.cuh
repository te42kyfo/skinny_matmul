#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "../PseudoQuad.cuh"
#include "../cu_complex.h"

namespace GENV7 {

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
  const int rowsPerBlock = BLOCKSIZE / M;
  int m = threadIdx.x % M;
  int localRow = threadIdx.x / M;
  int bOffset = localRow * ldb + m;
  int aOffset = localRow * lda + m;
  if (m >= N) bOffset = localRow * ldb + 0;
  if (bOffset >= rowsPerBlock * ldb) bOffset = 0;
  if (aOffset >= rowsPerBlock * lda) aOffset = 0;

  __shared__ iT
      blockStorage[(rowsPerBlock + 1) * M * (sizeof(T) > sizeof(iT) ? 2 : 1)];
  T *rowCache = reinterpret_cast<T *>(blockStorage);

  zero(blockStorage[threadIdx.x]);
  __syncthreads();

  iT threadSum[N];
  for (int n = 0; n < N; n++) {
    zero(threadSum[n]);
  }

  // Block synchronous loop
  int idx = blockIdx.x * rowsPerBlock;
  for (; idx < K - rowsPerBlock; idx += gridDim.x * rowsPerBlock) {
    T av = A[idx * lda + aOffset];
    T bv;

    if (!SELF) {
      bv = B[idx * ldb + bOffset];
    } else {
      bv = av;
    }

    __syncthreads();
    rowCache[threadIdx.x] = bv;
    __syncthreads();

    int localAddress = threadIdx.x - m;
    for (int n = 0; n < N; n++) {
      threadSum[n] = axpy2(threadSum[n], av, rowCache[localAddress + n]);
    }
  }

  // Remainder loop
  for (idx = idx + localRow; idx < K; idx += gridDim.x * rowsPerBlock) {
    T av = A[idx * lda + m];
    for (int n = 0; n < N; n++) {
      threadSum[n] = axpy2(threadSum[n], av, B[idx * ldb + n]);
    }
  }

  // Calculate block results
  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      iT blockSum;
      zero(blockSum);
      for (int w = 0; w < rowsPerBlock; w++) {
        blockSum = accu(blockSum, blockStorage[w * M + m]);
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
bool tsmttsm(int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;
  if (varM > 32 || varN > 32) return false;
  if (d_temp_storage == NULL)
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(iT) * 100 * 100 * 1000));
  if (blockCount * M * N > 100 * 100 * 1000) return false;

  int const blocksize = 256;

  if (N > M) {
    GENV7::blockProductKernel<T, iT, N, M, blocksize, true,
                              false><<<blockCount, blocksize>>>(
        B, A, (iT *)d_temp_storage, K, ldb, lda, ldc);

  } else {
    if (M == N && A == B) {
      GENV7::blockProductKernel<T, iT, M, N, blocksize, false,
                                true><<<blockCount, blocksize>>>(
          A, B, (iT *)d_temp_storage, K, lda, ldb, ldc);
    } else {
      GENV7::blockProductKernel<T, iT, M, N, blocksize, false,
                                false><<<blockCount, blocksize>>>(
          A, B, (iT *)d_temp_storage, K, lda, ldb, ldc);
    }
  }
  // GENV7::deviceReduce<T, iT, M, N><<<M * N / 256 + 1, 256>>>(
  //    (iT *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}