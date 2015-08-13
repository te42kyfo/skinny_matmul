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
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    cerr << "GPUassert: \"" << cudaGetErrorString(code) << "\"  in " << file
         << ": " << line << "\n";
    if (abort) exit(code);
  }
}

__global__ void initKernel(double *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 2.0;
  }
}

double measureMatmul(const size_t M, const size_t N, const size_t K,
                     const int blockCount) {
  double *A, *B, *d_temp_storage, *result;

  int iters = 3;
  GPU_ERROR(cudaMalloc(&A, sizeof(double) * M * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(double) * N * K));
  initKernel << <52, 256>>> (A, M * K);
  initKernel << <52, 256>>> (B, N * K);

  size_t temp_storage_bytes = 0;
  matmul(temp_storage_bytes, NULL, A, B, NULL, M, N, K, blockCount);

  GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(double) * temp_storage_bytes));
  GPU_ERROR(cudaMalloc(&result, sizeof(double) * M * N));

  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  for (int iter = 0; iter < iters; iter++) {
    matmul(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K,
           blockCount);
  }
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(d_temp_storage));
  GPU_ERROR(cudaFree(result));
  return (t2 - t1) / iters;
}

int main(int argc, char **argv) {
  int sampleSize = 5;

  size_t M = 2;
  size_t N = 2;
  size_t K = (size_t)5 * 1024 * 1024 * 1024 / (M + N) / 8 * 0.5;

  srand(time(NULL));

  for (size_t blockCount = 13; blockCount < 8 * 13; blockCount += 13) {
    vector<double> times(sampleSize);
    for (int t = 0; t < sampleSize; t++) {
      times[t] =
          measureMatmul(M, N, K + 2 * 1024 * (rand() % 1024), blockCount);
    }
    sort(times.begin(), times.end());

    cout << M << "xKx" << N << "\t" << setprecision(3) << blockCount << "\t"
         << (2 * M * N * K) * 1e-9 / times[sampleSize / 2] << std::endl
         << std::flush;
  }

  cout.flush();
}
