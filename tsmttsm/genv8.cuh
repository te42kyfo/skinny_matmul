#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "../PseudoQuad.cuh"
#include "../cu_complex.h"

namespace GENV8 {

template <typename T, typename iT, int M, int N>
__global__ void deviceReduce(iT *blockResults, T *result, T alpha, T beta,
                             int blockCount, int lda, int ldb, int ldc,
                             int ldi) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= M * N) return;
  int n = tidx / M;
  int m = tidx % M;

  iT sum;
  zero(sum);
  for (int i = 0; i < blockCount; i++) {
    sum = accu(sum, blockResults[i * N * ldi + n * ldi + m]);
  }

  result[n * ldc + m] = accu(scale(result[n * ldc + m], beta),
                             convert<iT, T>(scale2(renormalize(sum), alpha)));
}

// round to next smaller power of two
__device__ int roundPoT(int v) {
  int r = v;
  r |= r >> 1;
  r |= r >> 2;
  r |= r >> 4;
  r |= r >> 8;
  r |= r >> 16;
  r -= (r >> 1);
  return r;
}

template <typename T, typename iT, int M, int N, int TILESIZE, int BLOCKSIZE,
          bool TRANSPOSE, bool SELF>
__global__ void __launch_bounds__(BLOCKSIZE)
    blockProductKernel(const T *A, const T *B, iT *out, const int K,
                       const int lda, const int ldb, const int ldc) {
  __shared__ iT blockStorage[BLOCKSIZE];

  // ceil division
  const int mthreads = ((M - 1) / TILESIZE) + 1;
  const int nthreads = ((N - 1) / TILESIZE) + 1;
  const int rowsPerBlock = BLOCKSIZE / (mthreads * nthreads);
  const int localRow = threadIdx.x / (mthreads * nthreads);

  const int midx = threadIdx.x % mthreads;
  const int nidx = (threadIdx.x / mthreads) % nthreads;

  // leading dimension of internal buffer
  const int ldi = mthreads * TILESIZE;

  const int redSteps = roundPoT(rowsPerBlock);

  // Zero intermediate result buffer
  iT threadSum[TILESIZE][TILESIZE];
  for (int m = 0; m < TILESIZE; m++) {
    for (int n = 0; n < TILESIZE; n++) {
      threadSum[m][n] = 0;
    }
  }

  // Block synchronous loop
  int idx = blockIdx.x * rowsPerBlock;
  for (; idx < K - rowsPerBlock; idx += gridDim.x * rowsPerBlock) {
    for (int m = 0; m < TILESIZE; m++) {
      T av = __ldg(A + (idx + localRow) * lda + midx * TILESIZE + m);
      for (int n = 0; n < TILESIZE; n++) {
        T bv = __ldg(B + (idx + localRow) * ldb + nidx * TILESIZE + n);
        threadSum[m][n] = axpy2(threadSum[m][n], av, bv);
      }
    }
  }

  // Remainder loop
  for (idx = idx + localRow; idx < K; idx += gridDim.x * rowsPerBlock) {
    //    printf("%d %d %d\n", idx, blockIdx.x, localRow);
    for (int m = 0; m < TILESIZE; m++) {
      T av = __ldg(A + idx * lda + midx * TILESIZE + m);
      for (int n = 0; n < TILESIZE; n++) {
        T bv = __ldg(B + idx * ldb + nidx * TILESIZE + n);
        threadSum[m][n] = axpy2(threadSum[m][n], av, bv);
      }
    }
  }

  // Calculate block results
  for (int m = 0; m < TILESIZE; m++) {
    for (int n = 0; n < TILESIZE; n++) {
      __syncthreads();
      blockStorage[threadIdx.x] = threadSum[m][n];
      __syncthreads();

      for (unsigned int s = redSteps; s > 0; s /= 2) {
        if (localRow + s < rowsPerBlock && localRow < s) {
          int localOffset = nidx * mthreads + midx;
          int stride = mthreads * nthreads;
          blockStorage[localRow * stride + localOffset] +=
              blockStorage[(localRow + s) * stride + localOffset];
        }
        __syncthreads();
      }
      if (localRow == 0 && nidx * TILESIZE + n < N) {
        //        if (TRANSPOSE) {
        //  out[blockIdx.x * M * ldc + m * ldc + n] = blockStorage[m];
        //} else {
        out[blockIdx.x * N * ldi + ((nidx * TILESIZE) + n) * ldi +
            midx * TILESIZE + m] = blockStorage[nidx * mthreads + midx];
        //}
      }
    }
  }
}

void *d_temp_storage = NULL;
size_t temp_storage_size = 0;

template <typename T, typename iT, int M, int N, int tileSize>
bool tsmttsm(int blockCount, const int varM, const int varN, const int K,
             const T *A, const int lda, const T alpha, const T *B,
             const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;

  const int threadsPerRow = ((M - 1) / tileSize + 1) * ((N - 1) / tileSize + 1);
  const int mthreads = ((M - 1) / tileSize) + 1;
  const int ldi = mthreads * tileSize;

  int const blocksize =
      threadsPerRow * ((256 / threadsPerRow < 1) ? 1 : 256 / threadsPerRow);

  //  std::cout << threadsPerRow << " " << blocksize << "\n";
  if (threadsPerRow > blocksize || blocksize > 1024) {
    std::cout << "GENV8 Error: not enough threads in block or over maximum "
                 "thread block size\n";
    return false;
  }

  size_t required_temp_storage_size = ldi * N * blockCount;
  if (temp_storage_size < required_temp_storage_size) {
    // std::cout << "GENV8: Reallocate. Was " << temp_storage_size;
    GPU_ERROR(cudaFree(d_temp_storage));
    temp_storage_size = 3 * required_temp_storage_size;
    GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(iT) * temp_storage_size));
    //    std::cout << " is now " << temp_storage_size << "\n";
  }

  // if (N > M) {
  //   GENV8::blockProductKernel<T, iT, N, M, tileSize, blocksize, true,
  //                             false><<<blockCount, blocksize>>>(
  //       B, A, (iT *)d_temp_storage, K, ldb, lda, ldc);

  // } else {
  // if (M == N && A == B) {
  //   GENV8::blockProductKernel<T, iT, M, N, tileSize, blocksize, false,
  //                             true><<<blockCount, blocksize>>>(
  //       A, B, (iT *)d_temp_storage, K, lda, ldb, ldc);
  // } else {
  GENV8::blockProductKernel<T, iT, M, N, tileSize, blocksize, false,
                            false><<<blockCount, blocksize>>>(
      A, B, (iT *)d_temp_storage, K, lda, ldb, ldc);
  //   }
  // }

  // GENV8::deviceReduce<T, iT, M, N><<<M * N / 256 + 1, 256>>>(
  //    (iT *)d_temp_storage, C, alpha, beta, blockCount, lda, ldb, ldc, ldi);

  return true;
}
}
