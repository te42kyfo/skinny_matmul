#include <iomanip>
#include <iostream>
#include "../dtime.hpp"
#include "../gpu_error.cuh"
#include "../metrics.cuh"

using namespace std;

double* dA;
size_t sizeA = 100000000;

__global__ void initKernel(double* A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 3.3;
  }
}

__global__ void test_kernel(double* A) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double localSum = 0.0;
#pragma unroll(4)
  for (int i = 0; i < 256; i++) {
    localSum += A[i];  //__ldg(A + i);
  }
  A[tidx] = localSum;
}

template <int iters>
__global__ void LDS_kernel(double* A, int K) {
  asm(".shared .f64 rowCache[64];\n\t");

  double sum = 2.0;

  for (int iter = 0; iter < iters; iter++) {
    for (int i = 0; i < 8; i++) {
      asm("{\n\t"
          ".reg .f64 t1;\n\t"
          ".reg .f64 t2;\n\t"
          "ld.volatile.shared.f64 t1, [rowCache+8];\n\t"
          "ld.volatile.shared.f64 t2, [rowCache];\n\t"
          "fma.rn.f64 %0, t2, t1, %0;\n\t"
          "}"
          : "+d"(sum)
          :);
    }
  }

  if (threadIdx.x == 0) A[0] = sum;
}

template <int iters>
double callLDSKernel() {
  const int bS = 1024;

  LDS_kernel<iters><<<2 * 13, bS>>>(dA, sizeA);
  return 0.0;
}

void LDS() {
  const int iters = 128;
  cudaDeviceSynchronize();
  double t1 = dtime();
  callLDSKernel<iters>();
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  double shmemThroughput =
      measureMetric(callLDSKernel<iters>, "shared_load_throughput");

  cout
      << "LDS: "  //                                                            //
      << setprecision(3) << setw(7) << (t2 - t1) * 1000 << "ms "  //
      << setprecision(3) << setw(7)
      << (double)2 * 2 * 8 * iters * 1024 * 13 / (t2 - t1) / 758.0e6 / 13 / 32
      << " "  //
      << setprecision(4) << setw(7) << shmemThroughput / 1.e9 << "\n";
}

__global__ void L2LatencyKernel(double* A, size_t innerIterations) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  size_t bufferSize = 1024 * 1024;
  double sum = 0;
#pragma unroll(1)
  for (int i = 0; i < innerIterations / bufferSize; i++) {
#pragma unroll(1)
    for (int n = 0; n < bufferSize; n++) {
      size_t idx = (tidx + n * 32) % bufferSize;
      sum += A[idx];
    }
  }

  A[tidx] = sum;
}

double callKernel(void* func, double* dA, size_t K, int bC, int bS) {
  cudaConfigureCall(bC, bS);
  cudaSetupArgument(dA, 0);
  cudaSetupArgument(K, 8);
  cudaLaunch(func);
  return 0.0;
}

void L2Latency() {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;

  size_t innerIterations = 1024 * 1024;
  std::function<double()> callKernelFunc =
      std::bind(callKernel, (void*)L2LatencyKernel, dA, innerIterations, 1*13, 32);

  callKernelFunc();
  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  callKernelFunc();
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  double L2BW = (measureMetric(callKernelFunc, "l2_read_throughput") +
                 measureMetric(callKernelFunc, "l2_write_throughput")) /
                1.e9;
  double L2hitrate = measureMetric(callKernelFunc, "l2_l1_read_hit_rate");

  double dt = t2 - t1;
  double clock = prop.clockRate * 1.0e3;
  cout << deviceName << "  " << dt * 1000 << "ms  " << clock / 1.e9 << "GHz  "
       << L2BW << "GB/s  " << L2hitrate << "%  " << dt * clock / innerIterations
       << "cyc\n";
}

int main(int argc, char** argv) {
  GPU_ERROR(cudaMalloc(&dA, sizeof(double) * sizeA));
  initKernel<<<52, 256>>>(dA, sizeA);
  LDS();
  L2Latency();
}
