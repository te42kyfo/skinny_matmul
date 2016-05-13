#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <complex>

#include "tsmm.cuh"
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
string mode = "float complex";
int flopsPerCell = 8;

#elif DC
typedef complex<double> htype;
typedef cuDoubleComplex dtype;
dtype makeDtype(htype v) { return make_cuDoubleComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuDoubleComplex(v1, v2)
string mode = "double complex";
int flopsPerCell = 8;

#elif FR
typedef float htype;
typedef float dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) float(v1)
string mode = "float real";
int flopsPerCell = 2;

#elif DR
typedef double htype;
typedef double dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) double(v1)
string mode = "double real";
int flopsPerCell = 2;

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

void initMatmul(int M, int N, int K, int lda, int ldb, int ldc,
                size_t blockCount) {
  GPU_ERROR(cudaMalloc(&A, sizeof(dtype) * lda * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(dtype) * ldb * M));
  GPU_ERROR(cudaMalloc(&C, sizeof(dtype) * ldc * K));
  initKernel<<<52, 256>>>(A, lda * K);
  initKernel<<<52, 256>>>(B, ldb * M);
  initKernel<<<52, 256>>>(C, ldc * K);
}

void deInitMatmul() {
  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(C));
}

double measureMatmul(size_t M, size_t N, size_t K, int lda, int ldb, int ldc,
                     size_t blockCount, int iters) {
  GPU_ERROR(cudaDeviceSynchronize());

  bool passed = true;
  double t1 = dtime();
  for (int iter = 0; iter < iters; iter++) {
    passed = tsmm<dtype, PARM, PARN>(blockCount, K, makeDtype(1.0), A, lda, B,
                                     ldb, makeDtype(1.0), C, ldc);
  }
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();
  double time = (t2 - t1) / iters;

  if (!passed)
    return -time;
  else
    return time;
}

int main(int argc, char** argv) {
  size_t N = PARN;
  size_t M = PARM;

  if (M == 0 || N == 0) {
    std::cout << "  M   N         K  blockcount     time  GFlop  GByte\n";
    return 0;
  }

  size_t maxK = 2 * ((size_t)1 << 30) / ((M + N) * 8);
  size_t K = 0.2 * ((size_t)1 << 30) / ((M + N) * 8);

  initMatmul(M, N, maxK, M, N, N, 8 * 13);

  double resultTime = 0;
  while (resultTime < 0.1 && K * 2 <= maxK) {
    K *= 2;
    resultTime = measureMatmul(M, N, K, M, N, N, 26, 1);
  }

  int iters = max(1, (int)(0.05 / resultTime));

  size_t lda = M;
  double bestTime = -1;
  int bestBlockCount = 0;
  for (int blockCount = 1 * 13; blockCount <= 8 * 13; blockCount += 13) {
    int sampleSize = 3;
    vector<double> times(sampleSize);
    for (int t = 0; t < sampleSize; t++) {
      times[t] = measureMatmul(M, N, K, lda, N, N, blockCount, iters);
    }
    times.erase(remove_if(begin(times), end(times), [](double time) {
                  return time < 0;
                }), end(times));
    sort(times.begin(), times.end());

    if (times.size() != 0 &&
        (times[sampleSize / 2] < bestTime || bestBlockCount == 0)) {
      bestTime = times[sampleSize / 2];
      bestBlockCount = blockCount;
    }
  }
  double flops = 0;
  double bw = 0;

  if (bestTime > 0) {
    flops = M * N * K * flopsPerCell / bestTime * 1e-9;
    bw = (K * N + K * M) * sizeof(dtype) / bestTime * 1e-9;
  }
  cout << setw(3) << M << " " << setw(3) << N << " " << setw(9) << K << "  "
       << setw(10) << bestBlockCount << " " << setprecision(3) << setw(8)
       << bestTime << " " << setw(5) << flops << " " << setw(5) << bw << "\n";
  cout.flush();

  deInitMatmul();
}
