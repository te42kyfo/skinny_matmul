#pragma once

#include <cuda_runtime.h>
#include <iostream>

namespace SPEC8X8 {

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
    sum += blockResults[i * N * ldc + n * ldc + m];
  }

  result[n * ldc + m] = result[n * ldc + m] * beta + sum * alpha;
}

template <typename T, int M, int N, int BLOCKSIZE, bool TRANSPOSE>
__launch_bounds__(BLOCKSIZE, 8)
__global__ void blockProductKernel(const T *A, const T *B, T *out, const int K,
                                   const int lda, const int ldb, const int ldc) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ T blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int m = tidx % M;

  T threadSum[N];
  for (int n = 0; n < N; n++) {
    threadSum[n] = 0;
  }

  //try warp broadcast
  for (int idx = tidx / M; idx < K; idx += blockDim.x * gridDim.x / M) {
    T av = A[idx * lda + m];
    blockStorage[threadIdx.x] = B[idx*ldb+m];
    int localAddress = threadIdx.x-m;
    for (int n = 0; n < N; n++) {
      threadSum[n] +=  av * blockStorage[localAddress+n];
    }
  }

  for (int n = 0; n < N; n++) {
    __syncthreads();
    blockStorage[threadIdx.x] = threadSum[n];
    __syncthreads();

    if (threadIdx.x < M) {
      T blockSum = 0.0;
      for (int i = threadIdx.x; i < BLOCKSIZE; i += M) {
        blockSum += blockStorage[i];
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
  if (M != 8 && N != 8) {
    std::cout << "This kernel ist supossed to be specialized with M,N=8\n";
    return;
  }

  if (N > M) {
    SPEC8X8::blockProductKernel<T, N, M, 256, true><<<blockCount, 256>>>(
        B, A, d_temp_storage, K, ldb, lda, ldc);
  } else {
    SPEC8X8::blockProductKernel<T, M, N, 256, false><<<blockCount, 256>>>(
        A, B, d_temp_storage, K, lda, ldb, ldc);
  }
  SPEC8X8::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);
}
}
