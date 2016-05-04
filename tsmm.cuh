#include "cu_complex.h"

template <typename T, int M, int N, int BLOCKSIZE, bool betaiszero>
static __global__ void ghost_tsmm_cu_rm_cm(const T *A, const T *__restrict__ B,
                                           T *out, const int K, const int lda,
                                           const int ldb, const int ldc) {
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int n = threadIdx.x % N;

  int localRow = threadIdx.x / N;

  T __shared__ rowCache[M * BLOCKSIZE / N];

  for (int row = tidx / N; row < K; row += gridDim.x * BLOCKSIZE / N) {
    for (int i = 0; i < M / N; i++) {
      rowCache[localRow * M + n + i * N] = A[row * lda + n + i * N];
    }

    T sum = 0;
    for (int m = 0; m < M; m ++) {
      sum += rowCache[localRow * M + m] * B[m * ldb + n];
    }

    out[row * ldc + n] = sum;
  }
}

template <typename T, int M, int N>
void tsmm(const size_t blockCount, const int K, const T alpha, const T *A,
          const int lda, const T *B, const int ldb, const T beta, T *C,
          const int ldc) {
  const int BLOCKSIZE = 256;
  ghost_tsmm_cu_rm_cm<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
      A, B, C, K, lda, ldb, ldc);
}
