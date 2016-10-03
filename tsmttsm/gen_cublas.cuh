#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <typeinfo>
namespace GEN_CUBLAS {

cublasHandle_t handle;

template <typename T, int M, int N>
void matmul(size_t &temp_storage_bytes, T *d_temp_storage,
            const size_t blockCount, const int K, const T alpha, const T *A,
            const int lda, const T *B, const int ldb, const T beta, T *C,
            const int ldc) {
  if (d_temp_storage == NULL && temp_storage_bytes == 0) {
    temp_storage_bytes = 8;
    cublasCreate(&handle);
    return;
  }

  cublasStatus_t status;
  if (typeid(T) == typeid(double)) {
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
                         (double *)&alpha, (double *)A, lda, (double *)B, ldb,
                         (double *)&beta, (double *)C, ldc);
  } else if (typeid(T) == typeid(float)) {
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
                         (float *)&alpha, (float *)A, lda, (float *)B, ldb,
                         (float *)&beta, (float *)C, ldc);
  } else {
    std::cout << "cublasXgemm is implemented only for double and float\n";
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "not success\n";
  }
}
}
