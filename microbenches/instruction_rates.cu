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

enum class MEMPATH { TEX, L1 };

template <size_t bufferSize, MEMPATH MEM>
__global__ void L2LatencyKernel(double* A, size_t innerIterations) {


  double sum = 0;
#pragma unroll(1)
  for (int i = 0; i < innerIterations * 64 / bufferSize; i++) {
    #pragma unroll(1)
    for (int n = threadIdx.x; n < bufferSize; n+=64) {
      if (MEM == MEMPATH::TEX) {
        sum += __ldg(A + n);
      } else {
        sum += A[n];
      }
    }
  }

  A[threadIdx.x] = sum;
}

double callKernel(void* func, double* dA, size_t K, int bC, int bS) {
  cudaConfigureCall(bC, bS);
  cudaSetupArgument(dA, 0);
  cudaSetupArgument(K, 8);
  cudaLaunch(func);
  return 0.0;
}

template <size_t bufferSize, MEMPATH MEM>
void L2Latency() {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;

  size_t innerIterations =  256 * 1024;
  std::function<double()> callKernelFunc =
      std::bind(callKernel, (void*)L2LatencyKernel<bufferSize, MEM>, dA,
                innerIterations, 1, 64);

  callKernelFunc();
  callKernelFunc();
  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  callKernelFunc();
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  double L2BW = (measureMetric(callKernelFunc, "l2_read_throughput") +
                 measureMetric(callKernelFunc, "l2_write_throughput")) /
                1.e9;
  double texHitrate = measureMetric(callKernelFunc, "tex_cache_hit_rate");

  double L2hitrate = measureMetric(callKernelFunc, "l2_l1_read_hit_rate") +
                     measureMetric(callKernelFunc, "l2_tex_read_hit_rate");

  double dt = t2 - t1;
  double clock = prop.clockRate * 1.0e3;
  cout << setprecision(1) << setw(4) << fixed;
  cout << "Latency: " << deviceName << "  " << setw(8)
       << bufferSize * sizeof(double) / 1024 << "kB  " << dt * 1000 << "ms  "
       << clock / 1.e6 << "MHz  " << L2BW << "GB/s  " << setw(5) << L2hitrate
       << "%  " << setw(5) << texHitrate << "%  "
       << dt * clock / innerIterations << "cyc\n";
}

int main(int argc, char** argv) {
  measureMetricInit();

  GPU_ERROR(cudaMalloc(&dA, sizeof(double) * sizeA));
  initKernel<<<52, 256>>>(dA, sizeA);
  //  LDS();


  L2Latency<1 * 1024, MEMPATH::TEX>();  // 8
  L2Latency<1 * 1024, MEMPATH::TEX>();  // 8
  cout << "\n";
  L2Latency<2 * 1024, MEMPATH::TEX>();  // 16
  L2Latency<2 * 1024, MEMPATH::TEX>();  // 16
  cout << "\n";
  L2Latency<4 * 1024, MEMPATH::TEX>();  // 32
  L2Latency<4 * 1024, MEMPATH::TEX>();  // 32
  cout << "\n";
  L2Latency<64 * 1024, MEMPATH::TEX>();  // 512
  L2Latency<64 * 1024, MEMPATH::TEX>();  // 512
  cout << "\n";
  L2Latency<512 * 1024, MEMPATH::TEX>();  // 4MB
  L2Latency<512 * 1024, MEMPATH::TEX>();  // 4MB
  cout << "\n";
  L2Latency<1024 * 1024, MEMPATH::TEX>();  // 8MB
  L2Latency<1024 * 1024, MEMPATH::TEX>();  // 8MB
  cout << "\n";
  cout << "\n";
  L2Latency<2 * 1024, MEMPATH::L1>();  // 16
  L2Latency<2 * 1024, MEMPATH::L1>();  // 16
  cout << "\n";
  L2Latency<4 * 1024, MEMPATH::L1>();  // 32
  L2Latency<4 * 1024, MEMPATH::L1>();  // 32
  cout << "\n";
  L2Latency<64 * 1024, MEMPATH::L1>();  // 512
  L2Latency<64 * 1024, MEMPATH::L1>();  // 512
  cout << "\n";
  L2Latency<512 * 1024, MEMPATH::L1>();  // 4MB
  L2Latency<512 * 1024, MEMPATH::L1>();  // 4MB
  cout << "\n";
  L2Latency<1024 * 1024, MEMPATH::L1>();  // 8MB
  L2Latency<1024 * 1024, MEMPATH::L1>();  // 8MB
}
