#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <complex>

#include "skyblas.cuh"
#include "cu_complex.h"

#if !defined PARM || !defined PARN
#error "PARM or PARN is not specified! Specify M and N to measure"
#endif

using namespace std;


#define XSTR(s) STR(s)
#define STR(s) #s

#ifdef FC
typedef complex<float> htype;
typedef cuFloatComplex dtype;
dtype makeDtype(htype v) { return make_cuFloatComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuFloatComplex(v1, v2)

#elif DC
typedef complex<double> htype;
typedef cuDoubleComplex dtype;
dtype makeDtype(htype v) { return make_cuDoubleComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuDoubleComplex(v1, v2)

#elif FR
typedef float htype;
typedef float dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) float(v1)

#elif DR
typedef double htype;
typedef double dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) double(v1)

#endif

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

#define GPU_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    cerr << "GPUassert: \"" << cudaGetErrorString(code) << "\"  in " << file
         << ": " << line << "\n";
    if (abort) exit(code);
  }
}

__global__ void initKernel(dtype* A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = MAKE_DTYPE(idx % 3 - 1, 0);
  }
}

dtype* A;
dtype* B;
dtype* C;
dtype* d_temp_storage;

size_t temp_storage_bytes;

void initMatmul(Skyblas::MEMORY_ORDER AOrder, Skyblas::MEMORY_ORDER BOrder,
                int M, int N, int K, int lda, int ldb, int ldc,
                size_t blockCount) {
  GPU_ERROR(cudaMalloc(&A, sizeof(dtype) * lda * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(dtype) * ldb * K));
  GPU_ERROR(cudaMalloc(&C, sizeof(dtype) * ldc * N));
  initKernel<<<52, 256>>>(A, lda * K);
  initKernel<<<52, 256>>>(B, ldb * K);
  initKernel<<<52, 256>>>(C, ldc * N);

  temp_storage_bytes = 0;
  d_temp_storage = NULL;

  Skyblas::dgemm<dtype, PARM, PARN>(
      temp_storage_bytes, d_temp_storage, blockCount, AOrder, BOrder, M, N, K,
      makeDtype(1.0), A, lda, B, ldb, makeDtype(1.0), C, ldc);

  GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(dtype) * temp_storage_bytes));
}

void deInitMatmul() {
  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(C));
  GPU_ERROR(cudaFree(d_temp_storage));
}

double measureMatmul(Skyblas::MEMORY_ORDER AOrder, Skyblas::MEMORY_ORDER BOrder,
                     size_t M, size_t N, size_t K, int lda, int ldb, int ldc,
                     size_t blockCount, int iters, bool self) {
  GPU_ERROR(cudaDeviceSynchronize());

  htype halpha = 1.0;
  htype hbeta = 2.0;

  dtype dalpha = makeDtype(halpha);
  dtype dbeta = makeDtype(hbeta);

  double t1 = dtime();
  for (int iter = 0; iter < iters; iter++) {
    if (self)
      Skyblas::dgemm<dtype, PARM, PARN>(temp_storage_bytes, d_temp_storage,
                                        blockCount, AOrder, BOrder, M, N, K,
                                        dalpha, A, lda, A, lda, dbeta, C, ldc);
    else
      Skyblas::dgemm<dtype, PARM, PARN>(temp_storage_bytes, d_temp_storage,
                                        blockCount, AOrder, BOrder, M, N, K,
                                        dalpha, A, lda, B, ldb, dbeta, C, ldc);
  }
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  return (t2 - t1) / iters;
}

int main(int argc, char** argv) {
  size_t N = PARN;
  size_t M = PARM;
  bool self = false;

  if (M == 0 || N == 0) {
    std::cout << "  M   N         K  self  blockcount     time  perf\n";
    return 0;
  }

  size_t maxK = 2 * ((size_t)1 << 30) / ((M + N) * 8);
  size_t K = 0.2 * ((size_t)1 << 30) / ((M + N) * 8);

  initMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, maxK, M, N, N, 8 * 13);

  double resultTime = 0;
  while (resultTime < 0.1 && K * 2 <= maxK) {
    K *= 2;
    resultTime = measureMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, K, M, N, M,
                               26, 1, false);
  }

  int iters = max(1, (int)(0.05 / resultTime));

  size_t lda = M;
  double bestTime = 0;
  int bestBlockCount = 0;
  for (int blockCount = 1 * 13; blockCount <= 8 * 13; blockCount += 13) {
    int sampleSize = 3;
    vector<double> times(sampleSize);
    for (int t = 0; t < sampleSize; t++) {
      times[t] = measureMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, K, lda, N,
                               M, blockCount, iters, self);
    }
    sort(times.begin(), times.end());

    if (times[sampleSize / 2] < bestTime || bestBlockCount == 0) {
      bestTime = times[sampleSize / 2];
      bestBlockCount = blockCount;
    }
  }

  cout << XSTR(PREC) << " " << XSTR(MODE) << " " << setw(3) << M << " "
       << setw(3) << N << " " << setw(9) << K << "  "
       << ((self) ? "true " : "false") << " " << setw(10) << bestBlockCount
       << " " << setprecision(3) << setw(8) << bestTime << " " << setw(5)
       << M * N * K * 2 / bestTime * 1e-9 << "\n";
  cout.flush();

  deInitMatmul();
}
