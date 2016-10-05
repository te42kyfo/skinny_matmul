#pragma once
#include "cublas.cuh"
#include "fix1.cuh"
#include "fix2.cuh"
#include "fix_fb.cuh"

template <typename T, int M, int N>
bool tsmm_fix_blend(const int blockCount, const int varM, const int varN,
                    const int K, const T *A, const int lda, const T alpha,
                    const T *B, const int ldb, const T beta, T *C,
                    const int ldc) {
  if (M >= 7 && N >= 4 && tsmm_fix1<T, M, N>(blockCount, varM, varN, K, A, lda,
                                             alpha, B, ldb, beta, C, ldc))
    return true;
  if (M >= 14 && N >= 14 && tsmm_cublas<T>(blockCount, varM, varN, K, A, lda,
                                           alpha, B, ldb, beta, C, ldc))
    return true;
  if (M >= 7 && N <= 5 && tsmm_fix2<T, M, N>(blockCount, varM, varN, K, A, lda,
                                             alpha, B, ldb, beta, C, ldc))
    return true;
  if (tsmm_fix_fb<T, M, N>(blockCount, varM, varN, K, A, lda, alpha, B, ldb,
                           beta, C, ldc))
    return true;

  return tsmm_cublas<T>(blockCount, varM, varN, K, A, lda, alpha, B, ldb, beta,
                        C, ldc);
}
