#include <iomanip>
#include <iostream>
#include "../dtime.hpp"
#include "../gpu_error.cuh"
#include "../metrics.cuh"

using namespace std;

__global__ void initKernel(double* A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 3.0;
  }
}

__global__ void heat_kernel(double* A, double* B, unsigned int* volatile tags,
                            int width, int height) {
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int iters = 1000;

  double* dst = A;
  double* src = B;

#pragma unroll(1)
  for (int i = 0; i < iters; i++) {
    //    for (int idx = tidx + width; idx < width * (height - 2);
    //      idx += gridDim.x * blockDim.x) {
    int x = tidx % width;
    int y = tidx / width;
    if (y > 0 && y < height - 1 && x > 0 && y < width)
      dst[tidx] = 0.25 * (src[tidx + width] + src[tidx - width] +  //
                          src[tidx + height] + src[tidx - height]);
    //}
    double* temp = dst;
    dst = src;
    src = temp;

    unsigned int old = atomicInc(tags, gridDim.x * blockDim.x - 1);
    __threadfence();
    while (true) {
      unsigned int val = tags[0];
      if (val == 0) break;
    }

    // if (tidx == 0) printf("\n");
    //    if (threadIdx.x == 0) printf("Iter: %d %d\n", i, blockIdx.x);

    atomicInc(tags + 1, gridDim.x * blockDim.x - 1);
    __threadfence();
    while (true) {
      unsigned int val = tags[1];  // atomicCAS(tags + 1, 0, 0);
      if (val == 0) break;
    }
  }
}

int main(int argc, char** argv) {
  int width = 128;
  int height = 128;

  size_t sizeA = width * height;
  double* dA;
  double* dB;
  unsigned int* dTag;
  unsigned int val[] = {0, 0};
  GPU_ERROR(cudaMalloc(&dTag, 2 * sizeof(unsigned int)));
  GPU_ERROR(cudaMalloc(&dA, sizeof(double) * sizeA));
  GPU_ERROR(cudaMalloc(&dB, sizeof(double) * sizeA));
  GPU_ERROR(
      cudaMemcpy(dTag, &val, 2 * sizeof(unsigned int), cudaMemcpyDefault));
  initKernel<<<52, 256>>>(dA, sizeA);
  initKernel<<<52, 256>>>(dB, sizeA);

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int blockSize = 256;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, heat_kernel, blockSize, 0));

  cout << smCount * maxActiveBlocks << " " << blockSize << "\n";
  double t1 = dtime();
  for (int i = 0; i < 100; i++) {
    heat_kernel<<<smCount * maxActiveBlocks, blockSize>>>(dA, dB, dTag, width,
                                                          height);
    GPU_ERROR(cudaDeviceSynchronize());
    cout << i << "\n";
  }
  double t2 = dtime();
  cout << t2 - t1 << "\n";
}
