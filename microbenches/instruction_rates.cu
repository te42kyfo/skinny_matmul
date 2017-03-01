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
  int bC;
  const int bS = 1024;

  /*  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &bC, LDS_kernel<iters>, bS, 0));

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  */
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

int main(int argc, char** argv) {
  GPU_ERROR(cudaMalloc(&dA, sizeof(double) * sizeA));
  initKernel<<<52, 256>>>(dA, sizeA);
  LDS();
}