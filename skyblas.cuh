#include <iostream>

#include "genv1.cuh"
#include "genv2.cuh"
#include "genv3.cuh"
#include "genv4.cuh"
#include "genv5.cuh"
#include "gen_cublas.cuh"

#ifndef SKYBLAS_GENVER
#define SKYBLAS_GENVER GENV3
#endif

namespace Skyblas {

enum MEMORY_ORDER { ROW, COLUMN };

template <size_t TM, size_t TN>
void dgemm(size_t &temp_storage_bytes, double *d_temp_storage,
           const size_t blockCount, const MEMORY_ORDER AOrder,
           const MEMORY_ORDER BOrder, const int M, const int N, const int K,
           const double alpha, const double *A, const int lda, const double *B,
           const int ldb, const double beta, double *C, const int ldc) {
  if (TM == M && TN == N) {
    if (AOrder == Skyblas::COLUMN && BOrder == Skyblas::ROW) {
      GENV3::matmul<TM, TN>(temp_storage_bytes, d_temp_storage, blockCount, K,
                            alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
      std::cout << "Wrong memory Ordering\n";
      return;
    }
  } else if (TM == N && TN == K) {
    if (AOrder == Skyblas::ROW && BOrder == Skyblas::ROW) {
      std::cout << "Kernel 2\n";
    } else {
      std::cout << "Wrong memory Ordering\n";
      return;
    }
  } else {
    std::cout << "Skydgemm is specialized for " << TM << "xNx" << TN
              << ", input is " << M << "xNx" << N << "\n";
  }
}
}
