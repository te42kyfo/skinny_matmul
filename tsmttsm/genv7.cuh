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
__global__ void __launch_bounds__(BLOCKSIZE)
    blockProductKernel(const T *A, const T *B, iT *out, const int K,
                       const int lda, const int ldb, const int ldc) {
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
  T avNow = __ldg(A + idx * lda + aOffset);
  T bvNow = __ldg(B + idx * ldb + bOffset);
  T avNext = 0;
  T bvNext = 0;

  for (; idx < K - rowsPerBlock; idx += gridDim.x * rowsPerBlock) {
    int idxNext = idx + gridDim.x * rowsPerBlock;
    avNext = __ldg(A + idxNext * lda + aOffset);

    if (!SELF) {
      bvNext = __ldg(B + idxNext * ldb + bOffset);
    } else {
      bvNext = avNext;
    }
    __syncthreads();
    rowCache[threadIdx.x] = bvNow;
    __syncthreads();

    int localAddress = threadIdx.x - m;
    for (int n = 0; n < N; n++) {
      threadSum[n] = axpy2(threadSum[n], avNow, rowCache[localAddress + n]);
    }
    avNow = avNext;
    bvNow = bvNext;
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

    for (unsigned int s = rowsPerBlock / 2; s > 0; s >>= 1) {
      if (localRow < s) {
        blockStorage[localRow * M + m] += blockStorage[(localRow + s) * M + m];
      }
      __syncthreads();
    }

    if (threadIdx.x < M) {
      if (TRANSPOSE) {
        out[blockIdx.x * M * ldc + m * ldc + n] = blockStorage[m];
      } else {
        out[blockIdx.x * N * ldc + n * ldc + m] = blockStorage[m];
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

  if (d_temp_storage == NULL)
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(iT) * 100 * 100 * 1000));
  if (blockCount * M * N > 100 * 100 * 1000) return false;

  if (N > M) {
    int const blocksize = (256 / N) * N;
    GENV7::blockProductKernel<T, iT, N, M, blocksize, true,
                              false><<<blockCount, blocksize>>>(
        B, A, (iT *)d_temp_storage, K, ldb, lda, ldc);

  } else {
    int const blocksize = (256 / M) * M;
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

  //  GENV7::deviceReduce<T, iT, M, N><<<M * N / 256 + 1, 256>>>(
  //    (iT *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);

  return true;
}
}
