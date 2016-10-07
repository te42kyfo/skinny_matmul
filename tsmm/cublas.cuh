#pragma once
#include <cublas_v2.h>
#include <typeinfo>
#include "../eq.cuh"
#include "../gpu_error.cuh"

struct scopedCuMemDelete {
  void *_ptr;
  scopedCuMemDelete(void *ptr) : _ptr(ptr) {}
  ~scopedCuMemDelete() { GPU_ERROR(cudaFree(_ptr)); }
};

cublasHandle_t cublas_handle;
bool cublas_handle_initialized = false;
template <typename T>
bool tsmm_cublas(const int blockCount, const int M, const int N, const int K,
                 const T *A, const int lda, const T alpha, const T *B,
                 const int ldb, const T beta, T *C, const int ldc) {
  if (blockCount == 0) return true;
  if (!cublas_handle_initialized) {
    cublasCreate(&cublas_handle);
    cublas_handle_initialized = true;
  }

  scopedCuMemDelete A_scopedDeleter(0);
  if (A == C) {
    GPU_ERROR(cudaMalloc(&A_scopedDeleter._ptr, sizeof(T) * lda * K));
    GPU_ERROR(cudaMemcpy(A_scopedDeleter._ptr, C, sizeof(T) * lda * K,
                         cudaMemcpyDefault));
    A = (T *)A_scopedDeleter._ptr;
  }

  cublasStatus_t status;
  if (typeid(T) == typeid(double)) {
    status = cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M,
                         (double *)&alpha, (double *)B, ldb, (double *)A, lda,
                         (double *)&beta, (double *)C, ldc);
  } else if (typeid(T) == typeid(float)) {
    status = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M,
                         (float *)&alpha, (float *)B, ldb, (float *)A, lda,
                         (float *)&beta, (float *)C, ldc);
  } else if (typeid(T) == typeid(cuDoubleComplex)) {
    status = cublasZgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M,
                         (cuDoubleComplex *)&alpha, (cuDoubleComplex *)B, ldb,
                         (cuDoubleComplex *)A, lda, (cuDoubleComplex *)&beta,
                         (cuDoubleComplex *)C, ldc);
  } else if (typeid(T) == typeid(cuComplex)) {
    status =
        cublasCgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M,
                    (cuComplex *)&alpha, (cuComplex *)B, ldb, (cuComplex *)A,
                    lda, (cuComplex *)&beta, (cuComplex *)C, ldc);

  } else {
    return false;
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    return false;
  }
  return true;
}
