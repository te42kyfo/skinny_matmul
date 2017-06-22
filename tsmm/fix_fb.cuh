#pragma once
#include "../cminmax.h"
#include "../eq.cuh"

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_fix_fb_kernel(const T *__restrict__ A,
                                          const T *__restrict__ B, T *out,
                                          const int K, const int lda,
                                          const int ldb, const int ldc, T alpha,
                                          T beta) {
  int n = threadIdx.x % N;
  int localRow = threadIdx.x / N;
  int rowsPerBlock = BLOCKSIZE / N;
  if (localRow == rowsPerBlock) {
    localRow = 0;
    n = 0;
    printf("%d\n", threadIdx.x);
  }

  __shared__ __volatile__ T bCache[M][N];
#pragma unroll(1)
  for (int mn = threadIdx.x; mn < M * N; mn += BLOCKSIZE) {
    int tn = mn / M;
    int tm = mn % M;
    bCache[tm][tn] = B[tn * ldb + tm];
  }

  __syncthreads();

  int idx = blockIdx.x * rowsPerBlock;

  int halfSection = (K / 2 / rowsPerBlock) * rowsPerBlock;

  for (; idx < halfSection; idx += gridDim.x * rowsPerBlock) {
    int row = idx + localRow;
    T sum1, sum2;
    zero(sum1);
    zero(sum2);

    const int o1 = row * lda;
    const int o2 = (row + halfSection) * lda;

    for (int m = 0; m < M; m++) {
      T bV = bCache[m][n];
      sum1 = axpy(sum1, A[o1 + m], bV);
      sum2 = axpy(sum2, A[o2 + m], bV);
    }

    __syncthreads();

    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum1);
      out[(row + halfSection) * ldc + n] = scale(alpha, sum2);
    } else {
      out[row * ldc + n] = axpby(sum1, out[row * ldc + n], alpha, beta);
      out[(row + halfSection) * ldc + n] =
          axpby(sum2, out[(row + halfSection) * ldc + n], alpha, beta);
    }
  }

  // remainder loop
  for (idx += halfSection + localRow; idx < K;
       idx += gridDim.x * rowsPerBlock) {

    T sum;
    zero(sum);

    // #pragma unroll(M <= 8 ? M : 1)
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, A[idx * lda + m], bCache[m][n]);
    }
    __syncthreads();
    if (BETAISZERO) {
      out[idx * ldc + n] = scale(alpha, sum);
    } else {
      out[idx * ldc + n] = axpby(sum, out[idx * ldc + n], alpha, beta);
    }
  }
}

template <typename T, int M, int N>
bool tsmm_fix_fb(const int blockCount, const int varM, const int varN,
                 const int K, const T *A, const int lda, const T alpha,
                 const T *B, const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N) return false;

  const int BLOCKSIZE = (((M * N > 1024) ? (M * N > 55 ? 1024 : 384) : 256) / N)*N;

  //  std::cout << BLOCKSIZE << "\n";

  T Tzero;
  zero(Tzero);

  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaFuncSetCacheConfig(tsmm_fix_fb_kernel<T, M, N, BLOCKSIZE, true>,
                         cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(tsmm_fix_fb_kernel<T, M, N, BLOCKSIZE, false>,
                         cudaFuncCachePreferShared);

  if (eq(beta, Tzero)) {
    tsmm_fix_fb_kernel<T, M, N, BLOCKSIZE, true>
        <<<blockCount, BLOCKSIZE>>>(A, B, C, K, lda, ldb, ldc, alpha, beta);
  } else {
    tsmm_fix_fb_kernel<T, M, N, BLOCKSIZE, false>
        <<<blockCount, BLOCKSIZE>>>(A, B, C, K, lda, ldb, ldc, alpha, beta);
  }
  return true;
}
