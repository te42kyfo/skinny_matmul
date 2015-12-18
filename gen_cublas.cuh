#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

namespace GEN_CUBLAS {

cublasHandle_t handle;

template <int M, int N>
void matmul(size_t &temp_storage_bytes, double *d_temp_storage,
            const size_t blockCount, const int K, const double alpha,
            const double *A, const int lda, const double *B, const int ldb,
            const double beta, double *C, const int ldc) {
  if (d_temp_storage == NULL && temp_storage_bytes == 0) {
    temp_storage_bytes = 8;
    cublasCreate(&handle);
    return;
  }

  cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
                                      &alpha, A, lda, B, ldb, &beta, C, ldc);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "not success\n";
  }
}
}
