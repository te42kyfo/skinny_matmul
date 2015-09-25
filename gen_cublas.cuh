#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

namespace GEN_CUBLAS {

cublasHandle_t handle;

template <int M, int N>
void matmul(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
            double *B, double *result, const size_t K, const int blockCount) {
  if (d_temp_storage == NULL && temp_storage_bytes == 0) {
    temp_storage_bytes = 8;
    cublasCreate(&handle);
    return;
  }

  double alpha = 1.0;
  double beta = 0.0;
  cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
                                      &alpha, A, M, B, N, &beta, result, M);

  if( status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "not success\n";
  }
}
}
