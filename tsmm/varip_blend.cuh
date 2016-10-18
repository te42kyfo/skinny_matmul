#pragma once
#include "../eq.cuh"

#include "cublas.cuh"
#include "varip2.cuh"
#include "varip3.cuh"

template <typename T>
bool tsmm_varip_blend(const size_t blockCount, const int varM, const int varN,
                      const int K, const T *A, const int lda, const T alpha,
                      const T *B, const int ldb, const T beta, T *C,
                      const int ldc) {
  if (varN < 8 && (varM + 2) > 5 * varN &&
      tsmm_varip3(blockCount, varM, varN, K, A, lda, alpha, B, ldb, beta, C,
                  ldc)) {
    return true;
  } else if (sqrt((47 - varN) * (47 - varN) +
                  0.35 * (75 - varM) * (75 - varM)) > 35 &&
             !(varN > 12 && varM > 70) && !(varM > 16 && varN > 50) &&
             tsmm_varip2(blockCount, varM, varN, K, A, lda, alpha, B, ldb, beta,
                         C, ldc)) {
    return true;
  } else {
    return tsmm_cublas(blockCount, varM, varN, K, A, lda, alpha, B, ldb, beta,
                       C, ldc);
  }

  return false;
}
