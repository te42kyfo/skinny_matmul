#pragma once

#define ADD_
#include <iostream>
#include <typeinfo>
#include "magma.h"
#include "magma_lapack.h"

static bool is_magma_initialized = false;

template <typename T>
bool tsmttsm_magma(const int blockCount, const int M, const int N, const int K,
                   const T *A, const int lda, const T alpha, const T *B,
                   const int ldb, const T beta, T *C, const int ldc) {
  if (!is_magma_initialized) {
    magma_init();
    is_magma_initialized = true;
  }

  if (typeid(T) == typeid(double)) {
    magma_dgemm(MagmaNoTrans, MagmaTrans, M, N, K, alpha, (double *)A, lda,
                (double *)B, ldb, beta, (double *)C, ldc);
  } else if (typeid(T) == typeid(float)) {
    magma_sgemm(MagmaNoTrans, MagmaTrans, M, N, K, alpha, (float *)A, lda,
                (float *)B, ldb, beta, (float *)C, ldc);
  } else {
    return false;
  }
  return true;
}
