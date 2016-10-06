#include <cuda_runtime.h>
#include <sys/time.h>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../cu_complex.h"
#include "../gpu_error.cuh"
#include "cublas.cuh"
#include "fix1.cuh"
#include "fix2.cuh"
#include "fix_blend.cuh"
#include "fix_fb.cuh"
#include "var1.cuh"

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

using MatmulFunctionType = function<bool(
    const size_t, const int, const int, const int, const dtype*, const int,
    const dtype, const dtype*, const int, const dtype, dtype*, const int)>;

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
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

double measureMatmul(MatmulFunctionType matmulFunction, size_t M, size_t N,
                     size_t K, int lda, int ldb, int ldc, size_t blockCount,
                     int iters, bool self, dtype beta) {
  GPU_ERROR(cudaDeviceSynchronize());

  bool passed = true;
  double t1 = dtime();
  for (int iter = 0; iter < iters; iter++) {
    if (self)
      passed = matmulFunction(blockCount, M, N, K, C, ldc, makeDtype(2.0), B,
                              ldb, makeDtype(beta), C, ldc);
    else
      passed = matmulFunction(blockCount, M, N, K, A, lda, makeDtype(2.0), B,
                              ldb, makeDtype(beta), C, ldc);
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
  if (PARM == 0 || PARN == 0) {
    std::cout << "  M   N  name       K  blockcount     time  GFlop  GByte\n";
    return 0;
  }
  int m1 = 0;
  int m2 = 0;
  int n1 = 0;
  int n2 = 0;
  if (argc == 2) {
    m1 = 1;
    m2 = stoi(argv[1]);
  }
  if (argc >= 3) {
    m1 = stoi(argv[1]);
    m2 = stoi(argv[2]);
  }
  if (argc == 4) {
    cout << "Incomplete set of arguments\n";
    exit(1);
  }
  if (argc == 5) {
    n1 = stoi(argv[3]);
    n2 = stoi(argv[4]);
  }
  if (argc == 1) {
    m1 = m2 = PARM;
    n1 = n2 = PARN;
  }

  vector<pair<MatmulFunctionType, string>> versions;

#if PARM != 0 && PARN != 0
#ifdef FIX_BLEND
  versions.push_back({tsmm_fix_blend<dtype, PARM, PARN>, "FBLEND"});
#endif
#ifdef FIX_FB
  versions.push_back({tsmm_fix_fb<dtype, PARM, PARN>, "FIX_FB"});
#endif
#ifdef FIX1
  versions.push_back({tsmm_fix1<dtype, PARM, PARN>, "FIX_V1"});
#endif
#ifdef FIX2
  versions.push_back({tsmm_fix2<dtype, PARM, PARN>, "FIX_V2"});
#endif
#ifdef CUBLAS
  versions.push_back({tsmm_cublas<dtype>, "CUBLAS"});
#endif
#ifdef VAR1
  versions.push_back({tsmm_var1<dtype>, "VAR_V1"});
#endif
#endif

  initMatmul(1, 1, 2 * ((size_t)1 << 30) / ((1 + 1) * 8), 1, 1, 1, 8 * 13);

  for (int M = m1; M <= m2; M++) {
    for (int N = n1; N <= n2; N++) {
      if (n1 == 0 && n2 == 0) N = M;

      size_t maxK = 2 * ((size_t)1 << 30) / ((M + N) * 8);
      size_t K = 200;

      // One warmup call
      measureMatmul(versions[0].first, M, N, K, M, N, N, 13, 1, true, -1.0);
      double resultTime =
          measureMatmul(versions[0].first, M, N, K, M, N, N, 13, 1, true, -1.0);

      while (resultTime < 0.005 && K < maxK) {
        K = min(maxK, 2 * K);
        resultTime = measureMatmul(versions[0].first, M, N, K, M, N, N, 13, 1,
                                   true, -1.0);
      }

      for (const auto& matmulVersion : versions) {
        for (int self = 0; self <= (M == N) ? 1 : 0; self++) {
          for (htype beta = 0.0; beta <= 1.0; beta += 1.0) {
            int iters = 1;

            size_t lda = M;
            double bestTime = -1;
            int bestBlockCount = 0;
            for (int blockCount = 1 * 13; blockCount <= 8 * 13;
                 blockCount += 13) {
              int sampleSize = 3;
              vector<double> times(sampleSize);
              for (int t = 0; t < sampleSize; t++) {
                times[t] =
                    measureMatmul(matmulVersion.first, M, N, K, lda, N, N,
                                  blockCount, iters, (self == 1), beta);
              }
              times.erase(remove_if(begin(times), end(times),
                                    [](double time) { return time < 0; }),
                          end(times));
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
              flops = (M + (beta == 0 ? 0 : 1)) * K * N * flopsPerCell /
                      bestTime * 1.0e-9;
              bw = ((beta == 0 || self == 1 ? 1.0 : 2.0) * N + M) * K *
                   sizeof(double) / bestTime * 1.0e-9;
            }
            cout << setw(3) << M << " " << setw(3) << N << " " << beta << " "
                 << (self == 1 ? "A*A" : "A*B") << " " << matmulVersion.second
                 << " " << setw(9) << K << "  " << setw(10) << bestBlockCount
                 << " " << setprecision(3) << setw(8) << bestTime << " "
                 << setw(5) << setprecision(3) << flops << " " << setw(5) << bw
                 << "\n";
            cout.flush();
          }
        }
      }
      if (versions.size() > 1) cout << "\n";
    }
  }
  deInitMatmul();
}
