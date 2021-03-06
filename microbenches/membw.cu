#include <iomanip>
#include <iostream>
#include "../benchdb.hpp"
#include "../dtime.hpp"
#include "../gpu_error.cuh"
#include "../metrics.cuh"

using namespace std;

BenchDB* dbptr;

__global__ void initKernel(double* A, int N) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 3.0;
  }
}

template <int N>
__global__ void copyKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < K / 2 / 4; idx += gridDim.x * blockDim.x) {
    A[idx] = __ldg(C + idx);
    A[idx + K / 4] = __ldg(C + K / 4 + idx);
    A[idx + K / 4 * 2] = __ldg(C + K / 4 * 2 + idx);
    A[idx + K / 4 * 3] = __ldg(C + K / 4 * 3 + idx);
  }
}

template <int N>
__global__ void scaleKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < K / 2; idx += gridDim.x * blockDim.x) {
    A[idx] = C[idx] * 1.2;
  }
}

template <int N>
__global__ void updateKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < K / 2; idx += gridDim.x * blockDim.x) {
    A[idx] = A[idx] * 1.2;
  }
}


template <int N>
__global__ void sumKernel(double* A, double* C, int K) {
  double* B = C + K;
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < K / 3 / 2; idx += gridDim.x * blockDim.x) {
    A[idx] = B[idx] + C[idx];
    A[idx + K / 2] = B[idx + K / 2] + C[idx + K / 2];
  }
}

template <int N>
__global__ void triadKernel(double* A, double* C, int K) {
  double* B = C + K;
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < K / 3 / 2; idx += gridDim.x * blockDim.x) {
    A[idx] = B[idx] * 1.2 + C[idx];
    A[idx + K / 2] = B[idx + K / 2] * 1.2 + C[idx + K / 2];
  }
}

template <int N>
__global__ void reduceKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  double sum = 0.0;
  for (int idx = tidx; idx < K; idx += gridDim.x * blockDim.x) {
    sum += A[idx];
  }
  if (tidx == 123123) C[tidx] = sum;
}

template <int N>
__global__ void reduceKernelUnroll(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  double sum = 0.0;
  for (int idx = tidx; idx < K / 2; idx += gridDim.x * blockDim.x) {
    sum += A[idx];
    sum += A[idx + K / 2];
  }
  if (tidx == 123123) C[tidx] = sum;
}

template <int N>
__global__ void rakeStoreKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < K; idx += gridDim.x * blockDim.x) {
    for (int n = 0; n < N; n++) {
      A[idx * N + n] = 1.2;
    }
  }
}

template <int N>
__global__ void rakeScaleKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  for (int idx = tidx; idx < K; idx += gridDim.x * blockDim.x) {
#pragma unroll(1)
    for (int n = 0; n < N; n++) {
      C[idx * N + n] = 1.2 * __ldg(A + idx * N + n);
    }
  }
}

template <int N>
__global__ void rakeKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  for (int idx = tidx; idx < K; idx += gridDim.x * blockDim.x) {
    for (int n = 0; n < N; n++) {
      sum += A[idx * N + n] * 1.2;
    }
  }

  if (tidx == 0) C[tidx] = sum;
}

template <int N>
__global__ void rakeLDGKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  for (int idx = tidx; idx < K; idx += gridDim.x * blockDim.x) {
    for (int n = 0; n < N; n++) {
      sum += __ldg(A + idx * N + n) * 1.2;
    }
  }

  if (tidx == 0) C[tidx] = sum;
}

template <int N>
__global__ void fatRakeKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  for (int idx = tidx / N; idx < K; idx += gridDim.x * blockDim.x / N) {
    for (int n = 0; n < N; n++) {
      sum += A[idx * N + n] * 1.2;
    }
  }

  if (tidx == 0) C[tidx] = sum;
}

template <int N>
__global__ void fatRakeLDGKernel(double* A, double* C, int K) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  for (int idx = tidx / N; idx < K; idx += gridDim.x * blockDim.x / N) {
    for (int n = 0; n < N; n++) {
      sum += __ldg(A + idx * N + n) * 1.2;
    }
  }

  if (tidx == 0) C[tidx] = sum;
}

double callKernel(void* func, double* dA, double* dC, int sizeA, int bC,
                  int bS) {
  cudaConfigureCall(bC, bS);
  cudaSetupArgument(dA, 0);
  cudaSetupArgument(dC, 8);
  cudaSetupArgument(sizeA, 16);
  cudaLaunch(func);
  return 0.0;
}

void measureMore(void* func, int N, string kernelName, double* dA, double* dC,
                 int sizeA) {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;

  int blockSize = 256;
  int K = sizeA / N;

  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
                                                          func, blockSize, 0));

  std::function<double()> measureLoadFunction = std::bind(
      callKernel, func, dA, dC, K, maxActiveBlocks * smCount, blockSize);

  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  measureLoadFunction();
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();
  double dt = t2 - t1;

  double readBW = measureMetric(measureLoadFunction, "dram_read_throughput") +
                  measureMetric(measureLoadFunction, "dram_write_throughput");
  double eccBW = measureMetric(measureLoadFunction, "ecc_throughput");
  double dramBW =
      ((measureMetric(measureLoadFunction, "dram_read_throughput") +
        measureMetric(measureLoadFunction, "dram_write_throughput")) -
       eccBW / 2) *
      1.e-9;
  double L2BW = measureMetric(measureLoadFunction, "l2_read_throughput") +
                measureMetric(measureLoadFunction, "l2_write_throughput");
  double L2hitrate = measureMetric(measureLoadFunction, "l2_tex_read_hit_rate");
  double texBW = measureMetric(measureLoadFunction, "tex_cache_throughput");
  double texHitrate = measureMetric(measureLoadFunction, "tex_cache_hit_rate");

  double appBW = (size_t)N * K * sizeof(double) / dt * 1e-9;

  cout << setprecision(3) << setw(3) << N << " " << setw(12) << kernelName
       << " "
       << " " << setw(3) << maxActiveBlocks * smCount << " " << setw(5)
       << dt * 1000 << "ms  "                                          //
       << setw(7) << appBW << " "                                      //
       << setw(7) << dramBW << " "                                     //
       << setw(7) << L2BW / 1.0e9 << " "                               //
       << setw(7) << L2hitrate << "% "                                 //
       << setw(7) << L2BW / appBW / 1.0e9 << "x "                      //
       << setw(7) << L2BW / 758.0e6 / 13 << "B/c "                     //
       << setprecision(4) << setw(7) << texBW / 1.0e9 << " "           //
       << setprecision(3) << setw(7) << texBW / appBW / 1.0e9 << "x "  //
       << setprecision(4) << setw(7) << texHitrate << "% "             //
       << setprecision(3) << setw(7) << texBW / 758.06e6 / 13 << "B/c\n";

  dbptr->insert({{"multype", "\"stream\""},
                 {"device", "\"" + deviceName + "\""},
                 {"M", to_string(N)},
                 {"N", to_string(N)},
                 {"name", "\"" + kernelName + "\""}},
                {{"K", to_string(K)},
                 {"time", to_string(dt)},
                 {"bw", to_string(appBW)},
                 {"l2bw", to_string(L2BW)},
                 {"drambw", to_string(dramBW)}});
}

void measureLess(void* func, int N, string kernelName, double* dA, double* dC,
                 int sizeA) {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;

  int blockSize = 256;
  int K = sizeA / N;

  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
                                                          func, blockSize, 0));

  std::function<double()> measureLoadFunction = std::bind(
      callKernel, func, dA, dC, K, maxActiveBlocks * smCount, blockSize);

  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  measureLoadFunction();
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();
  double dt = t2 - t1;

  double appBW = (size_t)N * K * sizeof(double) / dt * 1e-9;

  double L2BW, texHitrate;
  L2BW = (measureMetric(measureLoadFunction, "l2_read_throughput") +
          measureMetric(measureLoadFunction, "l2_write_throughput")) /
         1.e9;
  texHitrate = measureMetric(measureLoadFunction, "tex_cache_hit_rate");

  cout << setprecision(3) << setw(5) << dt * 1000 << "ms  " << setw(5) << appBW
       << " " << setw(5) << L2BW << " " << setw(6) << texHitrate << " | ";
  dbptr->insert({{"multype", "\"stream\""},
                 {"device", "\"" + deviceName + "\""},
                 {"M", to_string(N)},
                 {"N", to_string(N)},
                 {"name", "\"" + kernelName + "\""}},
                {{"K", to_string(K)},
                 {"time", to_string(dt)},
                 {"bw", to_string(appBW)},
                 {"l2bw", to_string(L2BW)}});
}

template <int N>
void measureAll(double* dA, double* dC, size_t sizeA) {
  cout << setw(3) << N << " ";
  /* measureLess((void*)(rakeKernel<N>), N, "rake", dA, dC, sizeA);
  measureLess((void*)(rakeLDGKernel<N>), N, "rakeLDG", dA, dC, sizeA);
  measureLess((void*)(fatRakeKernel<N>), N, "fatRake", dA, dC, sizeA);
  measureLess((void*)(fatRakeLDGKernel<N>), N, "fatRakeLDG", dA, dC, sizeA);
  */
  measureMore((void*)(copyKernel<N>), N, "copy", dA, dC, sizeA);
  measureMore((void*)(scaleKernel<N>), N, "scale", dA, dC, sizeA);
  measureMore((void*)(updateKernel<N>), N, "update", dA, dC, sizeA);
  measureMore((void*)(sumKernel<N>), N, "sum", dA, dC, sizeA);
  measureMore((void*)(triadKernel<N>), N, "triad", dA, dC, sizeA);
  measureMore((void*)(reduceKernel<N>), N, "reduce", dA, dC, sizeA);
  measureMore((void*)(reduceKernelUnroll<N>), N, "reduceUnroll", dA, dC, sizeA);
  cout << "\n";
}

template <int MAXN>
void measureSeries(double* dA, double* dC, size_t sizeA) {
  measureSeries<MAXN - 1>(dA, dC, sizeA);
  measureAll<MAXN>(dA, dC, sizeA);
}
template <>
void measureSeries<0>(double* dA, double* dC, size_t sizeA) {}

int main(int argc, char** argv) {
  measureMetricInit();
  BenchDB db("../benchmarks.db");
  dbptr = &db;

  size_t sizeA = 1 * ((size_t)1 << 30) / sizeof(double);
  double* dA;
  GPU_ERROR(cudaMalloc(&dA, 3 * sizeof(double) * sizeA));
  initKernel<<<52, 256>>>(dA, 3 * sizeA);

  measureSeries<1>(dA, dA + sizeA, sizeA);
}
