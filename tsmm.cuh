#include "cu_complex.h"

namespace {

template <typename T, int M, int N, int BLOCKSIZE, bool betaiszero>
static __global__ void tsmm_fallback_kernel(const T *A, const T *__restrict__ B,
                                            T *out, const int K, const int lda,
                                            const int ldb, const int ldc) {
  int tidx = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int n = tidx % N;

  for (int row = tidx / N; row < K; row += gridDim.x * BLOCKSIZE / N) {
    T sum;
    zero(sum);
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, A[row * lda + m], B[m * ldb + n]);
    }
    out[row * ldc + n] = sum;
  }
}

template <typename T, int M, int N, int BLOCKSIZE, bool betaiszero>
static __global__ void tsmm_v1_kernel(const T *A, const T *__restrict__ B,
                                      T *out, const int K, const int lda,
                                      const int ldb, const int ldc) {
  int warpLane = threadIdx.x % 32;
  const int rowsPerWarp = 32 / N;
  const int n = warpLane % N;

  if (warpLane >= rowsPerWarp * N) {
    warpLane = rowsPerWarp * N - 1;
  }
  const int localRow = threadIdx.x / 32 * rowsPerWarp + warpLane / N;

  T __shared__ rowCache[(M / N <= 8) ? M * BLOCKSIZE / 32 * rowsPerWarp : 1];

  for (int row = blockIdx.x * BLOCKSIZE / 32 * rowsPerWarp + localRow; row < K;
       row += BLOCKSIZE * gridDim.x / 32 * rowsPerWarp) {
    for (int i = 0; i < M / N; i++) {
      rowCache[localRow * M + n + i * N] = A[row * lda + n + i * N];
    }

    T sum = 0;
    for (int m = 0; m < M; m++) {
      sum = axpy(sum, rowCache[localRow * M + m], B[m * ldb + n]);
    }

    out[row * ldc + n] = sum;
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

template <typename T, int M, int N, int BLOCKSIZE, bool betaiszero>
static __global__ void tsmm_v2_kernel(const T *__restrict__ A,
                                      const T *__restrict__ B,
                                      T *__restrict__ out, const int K,
                                      const int lda, const int ldb,
                                      const int ldc) {
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
      out[row * ldc + n] = warpReduce(gval, GANGSIZE);
    }
  }
}
template <typename T, int M, int N>
bool tsmm_fallback(const size_t blockCount, const int K, const T alpha,
                   const T *A, const int lda, const T *B, const int ldb,
                   const T beta, T *C, const int ldc) {
  const int BLOCKSIZE = 256;
  tsmm_fallback_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
      A, B, C, K, lda, ldb, ldc);
  return true;
}

template <typename T, int M, int N>
bool tsmm_v1(const size_t blockCount, const int K, const T alpha, const T *A,
             const int lda, const T *B, const int ldb, const T beta, T *C,
             const int ldc) {
  const int BLOCKSIZE = 256;

  if (N >= 4 && M >= 4 && M % N == 0 && M / N <= 6) {
    tsmm_v1_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc);
    return true;
  }

  return false;
}

template <typename T, int M, int N>
bool tsmm_v2(const size_t blockCount, const int K, const T alpha, const T *A,
             const int lda, const T *B, const int ldb, const T beta, T *C,
             const int ldc) {
  const int BLOCKSIZE = 256;

  if (M > N && M >= 32 / sizeof(T) && (N < 8 || M % (32 / sizeof(T)) == 0)) {
    tsmm_v2_kernel<T, M, N, BLOCKSIZE, false><<<blockCount, BLOCKSIZE>>>(
        A, B, C, K, lda, ldb, ldc);
    return true;
  }

  return false;
}
}

template <typename T, int M, int N>
bool tsmm(const size_t blockCount, const int K, const T alpha, const T *A,
          const int lda, const T *B, const int ldb, const T beta, T *C,
          const int ldc) {
  if (tsmm_v1<T, M, N>(blockCount, K, alpha, A, lda, B, ldb, beta, C, ldc))
    return true;
  if (tsmm_v2<T, M, N>(blockCount, K, alpha, A, lda, B, ldb, beta, C, ldc))
    return true;
  if (tsmm_fallback<T, M, N>(blockCount, K, alpha, A, lda, B, ldb, beta, C,
                             ldc))
    return true;
  return false;
}

template <>
bool tsmm<double, 0, 0>(const size_t blockCount, const int K,
                        const double alpha, const double *A, const int lda,
                        const double *B, const int ldb, const double beta,
                        double *C, const int ldc) {
  std::cout << "not implemented\n";
  return false;
}
