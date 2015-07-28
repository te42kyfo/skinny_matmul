
/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "../cub/cub.cuh"

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

size_t nextMultipleOf(size_t a, size_t b) { return (a / b + 1) * b; }

namespace oneXone {

template <int BLOCKSIZE>
__global__ void blockScalarProductKernel(double *A, double *B, double *out,
                                         size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double threadSum = 0.0;
  for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    threadSum += A[idx] * B[idx];
  }

  typedef cub::BlockReduce<double, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double blockSum = BlockReduce(temp_storage).Sum(threadSum);

  if (threadIdx.x == 0) out[blockIdx.x] = blockSum;
}

void oneXone(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
             double *B, double *result, const size_t M, const size_t N,
             const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_storage,
                           result, blockCount);
    temp_storage_bytes += blockCount * sizeof(double);
    return;
  }
  oneXone::blockScalarProductKernel<256> << <blockCount, 256>>>
      (A, B, d_temp_storage, K);
  cub::DeviceReduce::Sum(d_temp_storage + blockCount * sizeof(double),
                         temp_storage_bytes, d_temp_storage, result,
                         blockCount);
}
}

namespace twoXone {

template <int BLOCKSIZE>
__global__ void blockScalarProductKernel(double *A, double *B, double *out,
                                         size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  int warpLane = threadIdx.x % 32;

  double threadSum1 = 0.0;
  double threadSum2 = 0.0;

  __shared__ double transposeBuffer[64];

  for (size_t idx = tidx; idx < K; idx += blockDim.x * gridDim.x) {
    transposeBuffer[warpLane] = A[2 * idx - warpLane];
    transposeBuffer[warpLane + 32] = A[2 * idx - warpLane + 32];

    threadSum1 += transposeBuffer[warpLane] * B[idx];
    threadSum2 += transposeBuffer[warpLane + 32] * B[idx];
  }

  typedef cub::BlockReduce<double, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  double blockSum1 = BlockReduce(temp_storage).Sum(threadSum1);
  double blockSum2 = BlockReduce(temp_storage).Sum(threadSum2);
  if (threadIdx.x == 0) out[blockIdx.x] = blockSum1;
  if (threadIdx.x == 0) out[blockIdx.x + blockDim.x] = blockSum2;
}

void twoXone(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
             double *B, double *result, const size_t M, const size_t N,
             const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_storage,
                           result, blockCount);
    temp_storage_bytes = (temp_storage_bytes + blockCount * sizeof(double)) * 2;
    return;
  }
  twoXone::blockScalarProductKernel<256> << <blockCount, 256>>>
      (A, B, d_temp_storage, K);
  cub::DeviceReduce::Sum(d_temp_storage + 2 * blockCount * sizeof(double),
                         temp_storage_bytes, d_temp_storage, result,
                         blockCount);
  cub::DeviceReduce::Sum(
      d_temp_storage + 2 * blockCount * sizeof(double), temp_storage_bytes,
      d_temp_storage + blockCount * sizeof(double), result + 1, blockCount);
}
}

namespace twoXtwo {

template <int BLOCKSIZE>
__global__ void blockScalarProductKernel(double *A, double *B, double *out,
                                         size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  int warpLane = threadIdx.x % 32;

  double threadSum1 = 0.0;
  double threadSum2 = 0.0;
  double threadSum3 = 0.0;
  double threadSum4 = 0.0;

  __shared__ double transposeBuffer[64];

  for (size_t idx = tidx; idx < K; idx += blockDim.x * gridDim.x) {
    transposeBuffer[warpLane] = A[2 * idx - warpLane];
    transposeBuffer[warpLane + 32] = A[2 * idx - warpLane + 32];

    threadSum1 += transposeBuffer[warpLane] * B[idx];
    threadSum2 += transposeBuffer[warpLane + 32] * B[idx];
    threadSum3 += transposeBuffer[warpLane] * B[idx + K];
    threadSum4 += transposeBuffer[warpLane + 32] * B[idx + K];
  }

  typedef cub::BlockReduce<double, BLOCKSIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  double blockSum1 = BlockReduce(temp_storage).Sum(threadSum1);
  double blockSum2 = BlockReduce(temp_storage).Sum(threadSum2);
  double blockSum3 = BlockReduce(temp_storage).Sum(threadSum3);
  double blockSum4 = BlockReduce(temp_storage).Sum(threadSum4);

  if (threadIdx.x == 0) {
    out[blockIdx.x] = blockSum1;
    out[blockIdx.x + blockDim.x] = blockSum2;
    out[blockIdx.x + 2 * blockDim.x] = blockSum3;
    out[blockIdx.x + 3 * blockDim.x] = blockSum4;
  }
}

void twoXtwo(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
             double *B, double *result, const size_t M, const size_t N,
             const size_t K, const int blockCount) {
  if (temp_storage_bytes == 0) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_storage,
                           result, blockCount);
    temp_storage_bytes = (temp_storage_bytes + blockCount * sizeof(double)) * 4;
    return;
  }
  twoXtwo::blockScalarProductKernel<256> << <blockCount, 256>>>
      (A, B, d_temp_storage, K);
  cub::DeviceReduce::Sum(d_temp_storage + 4 * blockCount * sizeof(double),
                         temp_storage_bytes, d_temp_storage, result,
                         blockCount);
  cub::DeviceReduce::Sum(
      d_temp_storage + 4 * blockCount * sizeof(double), temp_storage_bytes,
      d_temp_storage + blockCount * sizeof(double), result + 1, blockCount);
  cub::DeviceReduce::Sum(
      d_temp_storage + 4 * blockCount * sizeof(double), temp_storage_bytes,
      d_temp_storage + 2 * blockCount * sizeof(double), result + 2, blockCount);
  cub::DeviceReduce::Sum(
      d_temp_storage + 4 * blockCount * sizeof(double), temp_storage_bytes,
      d_temp_storage + 3 * blockCount * sizeof(double), result + 3, blockCount);
}
}

void matmul(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
            double *B, double *result, const size_t M, const size_t N,
            const size_t K, const int blockCount) {
  if (M == 1 && N == 1) {
    oneXone::oneXone(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K,
                     blockCount);
    return;
  }
  if (M == 2 && N == 1) {
    twoXone::twoXone(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K,
                     blockCount);
    return;
  }
  if (M == 2 && N == 2) {
    twoXtwo::twoXtwo(temp_storage_bytes, d_temp_storage, A, B, result, M, N, K,
                     blockCount);
    return;
  }
  cout << "No matmul variant for " << M << "xKx" << N << "\n";
  exit(1);
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

  size_t M = 1;
  size_t N = 1;
  size_t K = (size_t)5 * 1024 * 1024 * 1024 / (M + N) / 8 * 0.4;

  for (size_t blockCount = 13; blockCount < 8 * 13; blockCount += 13) {
    vector<double> times(sampleSize);
    for (int t = 0; t < sampleSize; t++) {
      times[t] = measureMatmul(M, N, K + 8 * 1024 * (rand() % 21), blockCount);
    }
    sort(times.begin(), times.end());

    cout << M << "xKx" << N << "\t" << setprecision(3) << blockCount << "\t"
         << (2 * M * N * K) * 1e-9 / times[sampleSize / 2] << std::endl
         << std::flush;
  }

  cout.flush();
}
