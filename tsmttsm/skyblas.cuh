#include <iostream>

#ifndef SKYBLAS_GENVER
#define SKYBLAS_GENVER GENV3
#endif

#if SKYBLAS_GENVER == GENV1
#include "genv1.cuh"
#endif
#if SKYBLAS_GENVER == GENV3
#include "genv3.cuh"
#endif
#if SKYBLAS_GENVER == GENV4
#include "genv4.cuh"
#endif
#if SKYBLAS_GENVER == GENV5
#include "genv5.cuh"
#endif
#if SKYBLAS_GENVER == GEN_CUBLAS
#include "gen_cublas.cuh"
#endif
#if SKYBLAS_GENVER == GENV3_INST
#include "genv3_inst.cuh"
#endif
#if SKYBLAS_GENVER == SPEC8X8
#include "spec8x8.cuh"
#endif
#if SKYBLAS_GENVER == SPECSMALL
#include "specsmall.cuh"
#endif

namespace Skyblas {

enum MEMORY_ORDER { ROW, COLUMN };

template <typename T, size_t TM, size_t TN>
void dgemm(size_t &temp_storage_bytes, T *d_temp_storage,
           const size_t blockCount, const MEMORY_ORDER AOrder,
           const MEMORY_ORDER BOrder, const int M, const int N, const int K,
           const T alpha, const T *A, const int lda, const T *B, const int ldb,
           const T beta, T *C, const int ldc) {
  if (TM == M && TN == N) {
    if (AOrder == Skyblas::COLUMN && BOrder == Skyblas::ROW) {
      SKYBLAS_GENVER::matmul<T, TM, TN>(temp_storage_bytes, d_temp_storage,
                                        blockCount, K, alpha, A, lda, B, ldb,
                                        beta, C, ldc);
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

template <>
void dgemm<float, 0, 0>(size_t &temp_storage_bytes, float *d_temp_storage,
                        const size_t blockCount, const MEMORY_ORDER AOrder,
                        const MEMORY_ORDER BOrder, const int M, const int N,
                        const int K, const float alpha, const float *A,
                        const int lda, const float *B, const int ldb,
                        const float beta, float *C, const int ldc) {
  std::cout << "Can't instance with zero matrix dimensions\n";
}
template <>
void dgemm<cuFloatComplex, 0, 0>(
    size_t &temp_storage_bytes, cuFloatComplex *d_temp_storage,
    const size_t blockCount, const MEMORY_ORDER AOrder,
    const MEMORY_ORDER BOrder, const int M, const int N, const int K,
    const cuFloatComplex alpha, const cuFloatComplex *A, const int lda,
    const cuFloatComplex *B, const int ldb, const cuFloatComplex beta,
    cuFloatComplex *C, const int ldc) {
  std::cout << "Can't instance with zero matrix dimensions\n";
}
template <>
void dgemm<cuDoubleComplex, 0, 0>(
    size_t &temp_storage_bytes, cuDoubleComplex *d_temp_storage,
    const size_t blockCount, const MEMORY_ORDER AOrder,
    const MEMORY_ORDER BOrder, const int M, const int N, const int K,
    const cuDoubleComplex alpha, const cuDoubleComplex *A, const int lda,
    const cuDoubleComplex *B, const int ldb, const cuDoubleComplex beta,
    cuDoubleComplex *C, const int ldc) {
  std::cout << "Can't instance with zero matrix dimensions\n";
}
template <>
void dgemm<double, 0, 0>(size_t &temp_storage_bytes, double *d_temp_storage,
                         const size_t blockCount, const MEMORY_ORDER AOrder,
                         const MEMORY_ORDER BOrder, const int M, const int N,
                         const int K, const double alpha, const double *A,
                         const int lda, const double *B, const int ldb,
                         const double beta, double *C, const int ldc) {
  std::cout << "Can't instance with zero matrix dimensions\n";
}
}
