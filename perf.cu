#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#include "skyblas.cuh"

#if !defined PARM || !defined PARN
#error "PARM or PARN is not specified! Specify M and N to measure"
#endif

using namespace std;

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

__global__ void initKernel(double* A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = idx % 3 - 1;
  }
}

double* A;
double* B;
double* C;
double* d_temp_storage;

size_t temp_storage_bytes;

void initMatmul(Skyblas::MEMORY_ORDER AOrder, Skyblas::MEMORY_ORDER BOrder,
                int M, int N, int K, int lda, int ldb, int ldc,
                size_t blockCount) {
  GPU_ERROR(cudaMalloc(&A, sizeof(double) * lda * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(double) * ldb * K));
  GPU_ERROR(cudaMalloc(&C, sizeof(double) * ldc * N));
  initKernel<<<52, 256>>>(A, lda * K);
  initKernel<<<52, 256>>>(B, ldb * K);
  initKernel<<<52, 256>>>(C, ldc * N);

  temp_storage_bytes = 0;
  d_temp_storage = NULL;

  Skyblas::dgemm<PARM, PARN>(temp_storage_bytes, d_temp_storage, blockCount,
                             AOrder, BOrder, M, N, K, 1.0, A, lda, B, ldb, 1.0,
                             C, ldc);

  GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(double) * temp_storage_bytes));
}

void deInitMatmul() {
  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(C));
  GPU_ERROR(cudaFree(d_temp_storage));
}

double measureMatmul(Skyblas::MEMORY_ORDER AOrder, Skyblas::MEMORY_ORDER BOrder,
                     size_t M, size_t N, size_t K, int lda, int ldb, int ldc,
                     size_t blockCount) {
  GPU_ERROR(cudaDeviceSynchronize());

  double alpha = 2.0;
  double beta = 1.0;
  int iters = 1;
  double t1 = dtime();
  for (int iter = 0; iter < iters; iter++) {
    Skyblas::dgemm<PARM, PARN>(temp_storage_bytes, d_temp_storage, blockCount,
                               AOrder, BOrder, M, N, K, alpha, A, lda, B, ldb,
                               beta, C, ldc);
  }
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  return (t2 - t1) / iters;
}

int main(int argc, char** argv) {
  size_t N = PARN;
  size_t M = PARM;

  if (M == 0 || N == 0) {
    std::cout << "M\tN\tK  \t lda \t blockcount \t time   \t  perf\n";
    return 0;
  }

  size_t maxK = 2 * ((size_t)1 << 30) / ((M + 32 + N) * 8);
  size_t K = 0.2 * ((size_t)1 << 30) / ((M + 32 + N) * 8);

  initMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, maxK, M + 32, N, N, 8 * 13);

  double resultTime = 0;
  while (resultTime < 0.3 && K * 2 < maxK) {
    K *= 2;
    resultTime =
        measureMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, K, M, N, M, 26);
  }

  for (size_t lda = M; lda <= M + 32; lda++) {
    double bestTime = 0;
    int bestBlockCount = 0;
    for (int blockCount = 1 * 13; blockCount <= 8 * 13; blockCount += 13) {
      int sampleSize = 1;
      vector<double> times(sampleSize);
      for (int t = 0; t < sampleSize; t++) {
        times[t] = measureMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, K, lda, N,
                                 M, blockCount);
      }

      sort(times.begin(), times.end());

      if (times[sampleSize / 2] < bestTime || bestBlockCount == 0) {
        bestTime = times[sampleSize / 2];
        bestBlockCount = blockCount;
      }
    }

    cout << M << "\t" << N << "\t" << K << "\t" << lda << "\t" << bestBlockCount
         << "\t" << setprecision(3) << "\t" << bestTime << "\t"
         << M * N * K * 2 / bestTime * 1e-9 << "\n";
    cout.flush();
  }
  deInitMatmul();
}
