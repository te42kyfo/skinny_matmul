#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace GENV3_INST {

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

void __device__ storeAddress(int tidx, int *addressBuffer, int address,
                             int flag, int &addressCounter, int addnum) {
  if( addressCounter < addnum) {
    addressBuffer[(tidx * addnum + addressCounter) * 2 + 0] = address;
    addressBuffer[(tidx * addnum + addressCounter) * 2 + 1] = flag;
    addressCounter++;
  }
  //  if (addressCounter >= addnum) addressCounter = 0;
}

template <typename T, int M, int N, int BLOCKSIZE, bool TRANSPOSE>
__global__ void blockProductKernel(const T *A, const T *B, T *out, size_t K,
                                   size_t lda, size_t ldb, size_t ldc,
                                   int *addressBuffer, int addnum) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int addressCounter = 0;

  __shared__ T blockStorage[BLOCKSIZE];

  blockStorage[threadIdx.x] = 0.0;

  int m = tidx % M;

  if (blockDim.x * gridDim.x / M == tidx / M) return;

  T threadSum[N];
  for (int n = 0; n < N; n++) {
    threadSum[n] = 0;
  }

  for (size_t idx = tidx / M; idx < K; idx += blockDim.x * gridDim.x / M) {
    storeAddress(tidx, addressBuffer, (int)(idx * lda + m), 0, addressCounter,
                 addnum);
    for (int n = 0; n < N; n++) {
      threadSum[n] += A[idx * lda + m] * B[idx * ldb + n];
      storeAddress(tidx, addressBuffer, (int)(idx * ldb + n), 1, addressCounter,
                   addnum);
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

  int addnum = 100;
  int *d_addressBuffer = NULL;
  std::vector<int> h_addressBuffer(blockCount * 256 * addnum * 2);
  cudaMalloc(&d_addressBuffer, sizeof(int) * blockCount * 256 * addnum * 2);

  if (N > M) {
    GENV3_INST::blockProductKernel<T, N, M, 256, true><<<blockCount, 256>>>(
        B, A, d_temp_storage, K, ldb, lda, ldc, d_addressBuffer, addnum);
  } else {
    GENV3_INST::blockProductKernel<T, M, N, 256, false><<<blockCount, 256>>>(
        A, B, d_temp_storage, K, lda, ldb, ldc, d_addressBuffer, addnum);
  }
  GENV3_INST::deviceReduce<T, M, N><<<M * N / 256 + 1, 256>>>(
      d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc);

  cudaMemcpy(h_addressBuffer.data(), d_addressBuffer,
             sizeof(int) * 2 * blockCount * 256 * addnum, cudaMemcpyDefault);
  for (int tid = 0; tid < blockCount * 256; tid++) {
    std::cout << tid << " ";
    for (int n = 0; n < addnum * 2; n++) {
      std::cout << h_addressBuffer[tid * addnum * 2 + n] << " ";
    }
    std::cout << "\n";
  }
  cudaFree(d_addressBuffer);
}
}
