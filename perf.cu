#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#include "matmul.cuh"

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
double* d_temp_storage;
double* result;
size_t temp_storage_bytes;

void initMatmul(const size_t M, const size_t N, const size_t K,
                const size_t blockCount) {
  GPU_ERROR(cudaMalloc(&A, sizeof(double) * M * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(double) * N * K));
  initKernel << <52, 256>>> (A, M * K);
  initKernel << <52, 256>>> (B, N * K);

  temp_storage_bytes = 0;
  d_temp_storage = NULL;
  result = NULL;
  matmul(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K, blockCount);

  GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(double) * temp_storage_bytes));
  GPU_ERROR(cudaMalloc(&result, sizeof(double) * M * N));
}

void deInitMatmul() {
  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(d_temp_storage));
  GPU_ERROR(cudaFree(result));
}

double measureMatmul(const size_t M, const size_t N, const size_t K,
                     const int blockCount) {
  GPU_ERROR(cudaDeviceSynchronize());

  int iters = 1;
  double t1 = dtime();
  for (int iter = 0; iter < iters; iter++) {
    matmul(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K,
           blockCount);
  }
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  return (t2 - t1) / iters;
}

int main(int argc, char** argv) {
  size_t N = PARN;
  size_t M = PARM;

  size_t maxK = 1 * ((size_t)1 << 30) / ((M + N) * 8);
  initMatmul(M, N, maxK, 8 * 13);

  size_t K = 0.2 * ((size_t)1 << 30) / ((M + N) * 8);

  double resultTime = 0;
  while (resultTime < 0.3 && K * 2 < maxK) {
    K *= 2;
    resultTime = measureMatmul(M, N, K, 26);
  }

  double bestTime = 0;
  int bestBlockCount = 0;

  for (size_t blockCount = 1 * 13; blockCount <= 8 * 13; blockCount += 13) {
    int sampleSize = 1;
    vector<double> times(sampleSize);
    for (int t = 0; t < sampleSize; t++) {
      times[t] = measureMatmul(M, N, K, blockCount);
    }

    sort(times.begin(), times.end());

    if (times[sampleSize / 2] < bestTime || bestBlockCount == 0) {
      bestTime = times[sampleSize / 2];
      bestBlockCount = blockCount;
    }
  }
  cout << M << " " << N << " " << K << "\t" << bestBlockCount << "\t"
       << setprecision(3) << "\t" << bestTime << "\t"
       << M * N * K * 2 / bestTime * 1e-9 << "\n";
  cout.flush();
  deInitMatmul();
}
