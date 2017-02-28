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

  __shared__ iT blockStorage[2 * (rowsPerBlock + 1) * M *
                             (sizeof(T) > sizeof(iT) ? 2 : 1)];
  T *rowCache = reinterpret_cast<T *>(blockStorage);

  zero(blockStorage[threadIdx.x]);
  zero(blockStorage[threadIdx.x + BLOCKSIZE]);

  __syncthreads();

  iT threadSum[N];
  for (int n = 0; n < N; n++) {
    zero(threadSum[n]);
  }

  // Block synchronous loop
  int idx = 2 * blockIdx.x * rowsPerBlock;
  for (; idx < K - 2 * rowsPerBlock; idx += 2 * gridDim.x * rowsPerBlock) {
    // if (threadIdx.x == 0)
    //  printf("M  %d %d-%d\n", blockIdx.x, idx, idx + 2 * rowsPerBlock);
    T av1 = A[idx * lda + aOffset];
    T av2 = A[(idx + rowsPerBlock) * lda + aOffset];
    T bv1 = av1;
    T bv2 = av2;

    if (!SELF) {
      bv1 = B[idx * ldb + bOffset];
      bv2 = B[(idx + rowsPerBlock) * ldb + bOffset];
    }

    __syncthreads();
    rowCache[threadIdx.x] = bv1;
    rowCache[BLOCKSIZE + threadIdx.x] = bv2;
    __syncthreads();

    int localAddress = threadIdx.x - m;
    for (int n = 0; n < N; n++) {
      threadSum[n] = axpy2(threadSum[n], av1, rowCache[localAddress + n]);
      threadSum[n] =
          axpy2(threadSum[n], av2, rowCache[BLOCKSIZE + localAddress + n]);
    }
  }

  // if(threadIdx.x == 0) printf("   %d  @%d\n", blockIdx.x, idx);
  // Remainder loop
  for (idx = idx + localRow; idx < K; idx += gridDim.x * rowsPerBlock) {
    // if (threadIdx.x == 0)
    //  printf("R1 %d %d-%d\n", blockIdx.x, idx, idx + rowsPerBlock);
    T av = A[idx * lda + m];
    for (int n = 0; n < N; n++) {
      threadSum[n] = axpy2(threadSum[n], av, B[idx * ldb + n]);
    }
    idx += rowsPerBlock;
    if (idx >= K) break;
    // if (threadIdx.x == 0)
    //  printf("R2 %d %d-%d\n", blockIdx.x, idx, idx + rowsPerBlock);
    av = A[idx * lda + m];
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
  //GENV7::deviceReduce<T, iT, M, N><<<M * N / 256 + 1, 256>>>(
  //    (iT *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
  return true;
}
}
