#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <typeinfo>

cublasHandle_t tsmttsm_cublas_handle;
bool tsmttsm_cublas_initialized = false;

template <typename T>
bool tsmttsm_cublas(const int blockCount, const int M, const int N, const int K,
                    const T *A, const int lda, const T alpha, const T *B,
                    const int ldb, const T beta, T *C, const int ldc) {
  if (!tsmttsm_cublas_handle) {
    cublasCreate(&tsmttsm_cublas_handle);
    tsmttsm_cublas_initialized = true;
  }

  cublasStatus_t status;
  if (typeid(T) == typeid(double)) {
    status = cublasDgemm(tsmttsm_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N,
                         K, (double *)&alpha, (double *)A, lda, (double *)B,
                         ldb, (double *)&beta, (double *)C, ldc);
  } else if (typeid(T) == typeid(float)) {
    status = cublasSgemm(tsmttsm_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N,
                         K, (float *)&alpha, (float *)A, lda, (float *)B, ldb,
                         (float *)&beta, (float *)C, ldc);
  } else if (typeid(T) == typeid(cuDoubleComplex)) {
    status = cublasZgemm(tsmttsm_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N,
                         K, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)A,
                         lda, (cuDoubleComplex *)B, ldb,
                         (cuDoubleComplex *)&beta, (cuDoubleComplex *)C, ldc);
  } else if (typeid(T) == typeid(cuComplex)) {
    status =
        cublasCgemm(tsmttsm_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
                    (cuComplex *)&alpha, (cuComplex *)A, lda, (cuComplex *)B,
                    ldb, (cuComplex *)&beta, (cuComplex *)C, ldc);

  } else {
    return false;
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    return false;
  }
  return true;
}
