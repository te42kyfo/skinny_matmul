#include <cublas_v2.h>
#include <typeinfo>
#include "cu_complex.h"

namespace {

template <typename T>
bool eq(const T lhs, const T rhs) {
  return lhs == rhs;
};

template <>
bool eq<cuDoubleComplex>(const cuDoubleComplex lhs, const cuDoubleComplex rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

template <>
bool eq<cuFloatComplex>(const cuFloatComplex lhs, const cuFloatComplex rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_fallback_kernel(const T *__restrict__ A,
                                            const T *__restrict__ B, T *out,
                                            const int K, const int lda,
                                            const int ldb, const int ldc,
                                            T alpha, T beta) {
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int n = tidx % N;

  if (tidx / N == gridDim.x * BLOCKSIZE / N && !BETAISZERO) return;

  for (int row = tidx / N; row < K; row += gridDim.x * BLOCKSIZE / N) {
    T sum;
    zero(sum);
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, A[row * lda + m], B[m * ldb + n]);
    }
    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum);
    } else {
      out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
    }
  }
}

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_v1_kernel(const T *A, const T *__restrict__ B,
                                      T *out, const int K, const int lda,
                                      const int ldb, const int ldc, T alpha,
                                      T beta) {
  int warpLane = threadIdx.x % 32;
  const int rowsPerWarp = 32 / N;
  const int n = warpLane % N;

  if (warpLane >= rowsPerWarp * N) {
    warpLane = rowsPerWarp * N - 1;
  }
  const int localRow = threadIdx.x / 32 * rowsPerWarp + warpLane / N;

  T __shared__ rowCache[(M / N <= 16) ? M * BLOCKSIZE / 32 * rowsPerWarp : 1];

  for (int row = blockIdx.x * BLOCKSIZE / 32 * rowsPerWarp + localRow; row < K;
       row += BLOCKSIZE * gridDim.x / 32 * rowsPerWarp) {
    for (int i = 0; i < M / N; i++) {
      rowCache[localRow * M + n + i * N] = A[row * lda + n + i * N];
    }

    T sum;
    zero(sum);
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, rowCache[localRow * M + m], B[m * ldb + n]);
    }
    if (BETAISZERO) {
      out[row * ldc + n] = scale(alpha, sum);
    } else {
      out[row * ldc + n] = axpby(sum, out[row * ldc + n], alpha, beta);
    }
  }
}

template <typename T>
__device__ inline T __shfl_xor_t(T var, unsigned int srcLane, int width = 32) {
  int *a = reinterpret_cast<int *>(&var);
  for (int i = 0; i < sizeof(T) / 4; i++) {
    a[i] = __shfl_xor(a[i], srcLane, width);
  }
  return *reinterpret_cast<T *>(a);
}

template <typename T>
__device__ inline T warpReduce(T lval, int width) {
  for (int offset = width / 2; offset > 0; offset /= 2) {
    lval = accu(lval, __shfl_xor_t(lval, offset, width));
  }
  return lval;
}

template <typename T, int M, int N, int BLOCKSIZE, bool BETAISZERO>
static __global__ void tsmm_v2_kernel(const T *__restrict__ A,
                                      const T *__restrict__ B,
                                      T *__restrict__ out, const int K,
                                      const int lda, const int ldb,
                                      const int ldc, T alpha, T beta) {
  const int GANGSIZE = 32 / sizeof(T);

  int gId = threadIdx.x % GANGSIZE;
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;

  for (int row = tidx / GANGSIZE; row < K;
       row += gridDim.x * BLOCKSIZE / GANGSIZE) {
    for (int n = 0; n < N; n++) {
      T gval;
      zero(gval);
      for (int i = 0; i < (M - 1) / GANGSIZE + 1; i++) {
        int m = i * GANGSIZE + gId;
        if (m < M || M % GANGSIZE == 0)
          gval = axpy(gval, A[row * lda + m], B[m * ldb + n]);
      }
      if (BETAISZERO) {
        out[row * ldc + n] = scale(alpha, warpReduce(gval, GANGSIZE));
      } else {
        out[row * ldc + n] =
            axpby(warpReduce(gval, GANGSIZE), out[row * ldc + n], alpha, beta);
      }
    }
  }
}
template <typename T, int M, int N>
bool tsmm_fallback(const size_t blockCount, const int K, const T alpha,
                   const T *A, const int lda, const T *B, const int ldb,
                   const T beta, T *C, const int ldc) {
  const int BLOCKSIZE = 256;
  T Tzero;
  zero(Tzero);
  if (eq(beta, Tzero)) {
    tsmm_fallback_kernel<T, M, N, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc, alpha, beta);
  } else {
    tsmm_fallback_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc, alpha, beta);
  }
  return true;
}

template <typename T, int M, int N>
bool tsmm_v1(const size_t blockCount, const int K, const T alpha, const T *A,
             const int lda, const T *B, const int ldb, const T beta, T *C,
             const int ldc) {
  const int BLOCKSIZE = 256;

  if (M % N == 0 && M / N <= 16) {
    T Tzero;
    zero(Tzero);
    if (eq(beta, Tzero)) {
      tsmm_v1_kernel<T, M, N, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
          A, B, C, K, lda, ldb, ldc, alpha, beta);
    } else {
      tsmm_v1_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
          A, B, C, K, lda, ldb, ldc, alpha, beta);
    }
    return true;
  }

  return false;
}

template <typename T, int M, int N>
bool tsmm_v2(const size_t blockCount, const int K, const T alpha, const T *A,
             const int lda, const T *B, const int ldb, const T beta, T *C,
             const int ldc) {
  const int BLOCKSIZE = 256;

  if (M >= 32 / sizeof(T)) {
    T Tzero;
    zero(Tzero);
    if (eq(beta, Tzero)) {
      tsmm_v2_kernel<T, M, N, BLOCKSIZE, true><<<blockCount, BLOCKSIZE>>>(
          A, B, C, K, lda, ldb, ldc, alpha, beta);
    } else {
      tsmm_v2_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
          A, B, C, K, lda, ldb, ldc, alpha, beta);
    }
    return true;
  }

  return false;
}

cublasHandle_t cublas_handle;
bool cublas_handle_initialized = false;
template <typename T, int M, int N>
bool tsmm_cublas(const size_t blockCount, const int K, const T alpha,
                 const T *A, const int lda, const T *B, const int ldb,
                 const T beta, T *C, const int ldc) {
  if (!cublas_handle_initialized) {
    cublasCreate(&cublas_handle);
    cublas_handle_initialized = true;
  }

  cublasStatus_t status;
  if (typeid(T) == typeid(double)) {
    status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M,
                         (double *)&alpha, (double *)B, ldb, (double *)A, lda,
                         (double *)&beta, (double *)C, ldc);
  } else if (typeid(T) == typeid(float)) {
    status = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M,
                         (float *)&alpha, (float *)B, ldb, (float *)A, lda,
                         (float *)&beta, (float *)C, ldc);
  } else if (typeid(T) == typeid(cuDoubleComplex)) {
    status = cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M,
                         (cuDoubleComplex *)&alpha, (cuDoubleComplex *)B, ldb,
                         (cuDoubleComplex *)A, lda, (cuDoubleComplex *)&beta,
                         (cuDoubleComplex *)C, ldc);
  } else if (typeid(T) == typeid(cuComplex)) {
    status =
        cublasCgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, K, M,
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
}

template <typename T, int M, int N>
bool tsmm(const size_t blockCount, const int K, const T alpha, const T *A,
          const int lda, const T *B, const int ldb, const T beta, T *C,
          const int ldc) {
  if (M >= 7 && N >= 4 &&
      tsmm_v1<T, M, N>(blockCount, K, alpha, A, lda, B, ldb, beta, C, ldc))
    return true;
  if (M >= 14 && N >= 14 &&
      tsmm_cublas<T, M, N>(blockCount, K, alpha, A, lda, B, ldb, beta, C, ldc))
    return true;
  if (M >= 7 && N <= 5 &&
      tsmm_v2<T, M, N>(blockCount, K, alpha, A, lda, B, ldb, beta, C, ldc))
    return true;
  if (tsmm_fallback<T, M, N>(blockCount, K, alpha, A, lda, B, ldb, beta, C,
                             ldc))
    return true;

  return tsmm_cublas<T, M, N>(blockCount, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
bool tsmm<double, 0, 0>(const size_t blockCount, const int K,
                        const double alpha, const double *A, const int lda,
                        const double *B, const int ldb, const double beta,
                        double *C, const int ldc) {
  std::cout << "not implemented\n";
  return false;
}
