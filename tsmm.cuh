#include "cu_complex.h"

template <typename T, int M, int N, bool betaiszero>
static __global__ void ghost_tsmm_cu_rm_cm(const T *A, const T *B, T *out,
                                           const int K, const int lda,
                                           const int ldb, const int ldc) {}

template <typename T, int M, int N>
void tsmm(const size_t blockCount, const int K, const T alpha, const T *A,
          const int lda, const T *B, const int ldb, const T beta, T *C,
          const int ldc) {
  ghost_tsmm_cu_rm_cm<T, M, N, false><<<256, 8 * 13>>>(A, B, C, K, lda, ldb, ldc);
}
