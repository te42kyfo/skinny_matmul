#include <iomanip>
#include <iostream>
#include "../dtime.hpp"
#include "../gpu_error.cuh"
#include "../metrics.cuh"

using namespace std;

__global__ void initKernel(double* A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 3;
  }
}

template <int N>
__global__ void rakeKernel(double* A, double* C, size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  for (int idx = tidx; idx < K; idx += gridDim.x * blockDim.x) {
    for (int n = 0; n < N; n++) {
      sum += A[idx * N + n] * 1.2;
    }
  }

  if (tidx == 0) C[tidx] = sum;
}

template <int N>
__global__ void rakeLDGKernel(double* A, double* C, size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  for (int idx = tidx; idx < K; idx += gridDim.x * blockDim.x) {
    for (int n = 0; n < N; n++) {
      sum += __ldg(A + idx * N + n) * 1.2;
    }
  }

  if (tidx == 0) C[tidx] = sum;
}

template <int N>
__global__ void fatRakeKernel(double* A, double* C, size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  for (int idx = tidx / N; idx < K; idx += gridDim.x * blockDim.x / N) {
    for (int n = 0; n < N; n++) {
      sum += A[idx * N + n] * 1.2;
    }
  }

  if (tidx == 0) C[tidx] = sum;
}

template <int N>
__global__ void fatRakeLDGKernel(double* A, double* C, size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  double sum = 0;
  for (int idx = tidx / N; idx < K; idx += gridDim.x * blockDim.x / N) {
    for (int n = 0; n < N; n++) {
      sum += __ldg(A + idx * N + n) * 1.2;
    }
  }

  if (tidx == 0) C[tidx] = sum;
}

enum class KERNEL { rake, rakeLDG, fatRake, fatRakeLDG };

template <int N, KERNEL KT>
double callKernel(double* dA, double* dC, int sizeA, int bC, int bS) {
  if (KT == KERNEL::rake) rakeKernel<N><<<bC, bS>>>(dA, dC, sizeA);
  if (KT == KERNEL::rakeLDG) rakeLDGKernel<N><<<bC, bS>>>(dA, dC, sizeA);
  if (KT == KERNEL::fatRake) fatRakeKernel<N><<<bC, bS>>>(dA, dC, sizeA);
  if (KT == KERNEL::fatRakeLDG) fatRakeLDGKernel<N><<<bC, bS>>>(dA, dC, sizeA);
  return 0.0;
}

template <int N, KERNEL KT>
void measureMore(double* dA, double* dC, int sizeA) {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;

  int blockSize = 256;
  int K = sizeA / N;

  int maxActiveBlocks;
  if (KT == KERNEL::rake)
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, rakeKernel<N>, blockSize, 0));
  if (KT == KERNEL::rakeLDG)
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, rakeLDGKernel<N>, blockSize, 0));
  if (KT == KERNEL::fatRake)
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, fatRakeKernel<N>, blockSize, 0));
  if (KT == KERNEL::fatRakeLDG)
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, fatRakeLDGKernel<N>, blockSize, 0));

  std::function<double()> measureLoadFunction = std::bind(
      callKernel<N, KT>, dA, dC, K, maxActiveBlocks * smCount, blockSize);

  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  measureLoadFunction();
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();
  double dt = t2 - t1;

  double readBW = measureMetric(measureLoadFunction, "dram_read_throughput");
  double eccBW = measureMetric(measureLoadFunction, "ecc_throughput");
  double L2BW = measureMetric(measureLoadFunction, "l2_read_throughput");
  double L2hitrate = measureMetric(measureLoadFunction, "l2_tex_read_hit_rate");
  double texBW = measureMetric(measureLoadFunction, "tex_cache_throughput");
  double texHitrate = measureMetric(measureLoadFunction, "tex_cache_hit_rate");

  double appBW = (size_t)N * K * sizeof(double) / dt * 1e-9;

  cout << setprecision(3) << setw(3) << N << " " << setw(3)
       << maxActiveBlocks * smCount << " " << setw(5) << dt * 1000 << "ms  "  //
       << setw(7) << appBW << " "                                             //
       << setw(7) << (readBW - eccBW / 2) / 1.e9 << " "                       //
       << setw(7) << L2BW / 1.0e9 << " "                                      //
       << setw(7) << L2hitrate << "% "                                        //
       << setw(7) << L2BW / appBW / 1.0e9 << "x "                             //
       << setw(7) << L2BW / 758.0e6 / 13 << " "                               //
       << setprecision(4) << setw(7) << texBW / 1.0e9 << " "                  //
       << setprecision(3) << setw(7) << texBW / appBW / 1.0e9 << "x "         //
       << setw(7) << texHitrate << "%\n";
}

template <int N, KERNEL KT>
void measureLess(double* dA, double* dC, int sizeA) {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;

  int blockSize = 256;
  int K = sizeA / N;

  int maxActiveBlocks;
  if (KT == KERNEL::rake)
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, rakeKernel<N>, blockSize, 0));
  if (KT == KERNEL::rakeLDG)
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, rakeLDGKernel<N>, blockSize, 0));
  if (KT == KERNEL::fatRake)
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, fatRakeKernel<N>, blockSize, 0));
  if (KT == KERNEL::fatRakeLDG)
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, fatRakeLDGKernel<N>, blockSize, 0));

  std::function<double()> measureLoadFunction = std::bind(
      callKernel<N, KT>, dA, dC, K, maxActiveBlocks * smCount, blockSize);

  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  measureLoadFunction();
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();
  double dt = t2 - t1;

  double appBW = (size_t)N * K * sizeof(double) / dt * 1e-9;
  double L2BW = measureMetric(measureLoadFunction, "l2_read_throughput");
  double texHitrate = measureMetric(measureLoadFunction, "tex_cache_hit_rate");

  cout << setprecision(3) << setw(5) << appBW << " " << setw(5) << L2BW / 1.0e9
       << " " << setw(5) << texHitrate << " ";
}

template <int N>
void measureAll(double* dA, double* dC, size_t sizeA) {
  cout << setw(3) << N << " ";
  measureLess<N, KERNEL::rake>(dA, dC, sizeA);
  measureLess<N, KERNEL::rakeLDG>(dA, dC, sizeA);
  measureLess<N, KERNEL::fatRake>(dA, dC, sizeA);
  measureLess<N, KERNEL::fatRakeLDG>(dA, dC, sizeA);
  cout << "\n";
}

int main(int argc, char** argv) {
  size_t sizeA = 1 * ((size_t)1 << 30) / sizeof(double);
  size_t sizeC = 256 * 2000;
  double* dA;
  double* dC;
  GPU_ERROR(cudaMalloc(&dA, sizeof(double) * sizeA));
  GPU_ERROR(cudaMalloc(&dC, sizeof(double) * sizeC));
  initKernel<<<52, 256>>>(dA, sizeA);
  initKernel<<<52, 256>>>(dC, sizeC);

  measureAll<1>(dA, dC, sizeA);
  measureAll<2>(dA, dC, sizeA);
  measureAll<3>(dA, dC, sizeA);
  measureAll<4>(dA, dC, sizeA);
  measureAll<5>(dA, dC, sizeA);
  measureAll<6>(dA, dC, sizeA);
  measureAll<7>(dA, dC, sizeA);
  measureAll<8>(dA, dC, sizeA);
  measureAll<9>(dA, dC, sizeA);
  measureAll<10>(dA, dC, sizeA);
  measureAll<11>(dA, dC, sizeA);
  measureAll<12>(dA, dC, sizeA);
  measureAll<13>(dA, dC, sizeA);
  measureAll<14>(dA, dC, sizeA);
  measureAll<15>(dA, dC, sizeA);
  measureAll<16>(dA, dC, sizeA);
  measureAll<17>(dA, dC, sizeA);
  measureAll<18>(dA, dC, sizeA);
  measureAll<19>(dA, dC, sizeA);
  measureAll<20>(dA, dC, sizeA);
  measureAll<21>(dA, dC, sizeA);
  measureAll<22>(dA, dC, sizeA);
  measureAll<23>(dA, dC, sizeA);
  measureAll<24>(dA, dC, sizeA);
  measureAll<25>(dA, dC, sizeA);
  measureAll<26>(dA, dC, sizeA);
  measureAll<27>(dA, dC, sizeA);
  measureAll<28>(dA, dC, sizeA);
  measureAll<29>(dA, dC, sizeA);
  measureAll<30>(dA, dC, sizeA);
  measureAll<31>(dA, dC, sizeA);
  measureAll<32>(dA, dC, sizeA);
}