#pragma once
#include "../eq.cuh"

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ __launch_bounds__(BLOCKSIZE) void tsmm_fix3_kernel(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ out,
    const int K, const int lda, const int ldb, const int ldc, T alpha, T beta) {
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;

  __shared__  T bCache[M][N + (N%2==0 ? 1 : 0)];
#pragma unroll(1)
  for (int mn = threadIdx.x; mn < M * N; mn += BLOCKSIZE) {
    int tn = mn / M;
    int tm = mn % M;
    bCache[tm][tn] = B[tn * ldb + tm];
  }

  __syncthreads();

  for (int row = tidx; row < K; row += gridDim.x * blockDim.x) {

    T avals[M];
    for (int m = 0; m < M; m++) {
      avals[m] = A[row * lda + m];
    }
    for (int n = 0; n < N; n++) {
      T sums;
      for (int m = 0; m < M; m++) {
        sums = axpy(sums, avals[m], bCache[m][n]);
      }

      if (BETAISZERO) {
        out[row * ldc + n] = scale(alpha, sums);
      } else {
        out[row * ldc + n] =
            axpby(sums, __ldg(out + row * ldc + n), alpha, beta);
      }
    }
  }
}

template <typename T, int M, int N>
bool tsmm_fix3(const size_t blockCount, const int varM, const int varN,
               const int K, const T *A, const int lda, const T alpha,
               const T *B, const int ldb, const T beta, T *C, const int ldc) {
  if (varM != M || varN != N || A == C) return false;

  const int BLOCKSIZE = 128;

  struct cudaFuncAttributes funcAttrib;
  cudaFuncGetAttributes(&funcAttrib,
                        tsmm_fix3_kernel<T, M, N, BLOCKSIZE, false>);

  //  std::cout << M << ": " << funcAttrib.numRegs << "\n";

  T Tzero;
  zero(Tzero);
  if (eq(beta, Tzero)) {
    tsmm_fix3_kernel<T, M, N, BLOCKSIZE, true>
        <<<blockCount, BLOCKSIZE>>>(A, B, C, K, lda, ldb, ldc, alpha, beta);
  } else {
    tsmm_fix3_kernel<T, M, N, BLOCKSIZE, false>
        <<<blockCount, BLOCKSIZE>>>(A, B, C, K, lda, ldb, ldc, alpha, beta);
  }
  return true;
}
