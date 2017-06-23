#include <iomanip>
#include <iostream>
#include "../benchdb.hpp"
#include "../dtime.hpp"
#include "../gpu_error.cuh"
#include "../metrics.cuh"

using namespace std;

BenchDB* dbptr;

__global__ void initKernel(double* A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 1.2;
  }
}

template <int N>
__global__ void kernel(double* A, double* C, size_t K) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < K; idx += blockDim.x * gridDim.x) {
    double vA = __ldg(A + idx);
    double vC = __ldg(C + idx);
    double a = 0.5;
    double c = 0.5;
    for (int i = 0; i < N; i++) {
      a = a * vA;
      c = c * vC;
    }
    A[idx] = a * c;
  }
}

template <int N>
void measure(double* A, double* C, size_t K) {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;

  cudaDeviceSynchronize();
  double start = dtime();
  kernel<N><<<8 * smCount, 256>>>(A, C, K);
  cudaDeviceSynchronize();
  double end = dtime();
  double dt = end - start;
  double ic = (1.0 + 2.0 * N) / (3.0 * 8.0);
  double bw = (3.0 * 8.0 * K) / dt * 1e-9;
  double flops = bw * ic;
  cout << setprecision(3);
  cout << N << " " << setw(5) << ic << " " << setw(5) << dt * 1000.0 << "ms "
       << setw(5) << bw << " " << setw(5) << flops << "\n";

  dbptr->insert({{"multype", "\"roofline\""},
                 {"device", "\"" + deviceName + "\""},
                 {"M", to_string(N)},
                 {"N", to_string(N)},
                 {"name", "\"roofline\""}},
                {{"K", to_string(K)},
                 {"time", to_string(dt)},
                 {"bw", to_string(bw)},
                 {"flops", to_string(flops)}});
}

int main(int argc, char** argv) {
  BenchDB db("../benchmarks.db");
  dbptr = &db;

  size_t sizeA = 1 * ((size_t)1 << 30) / sizeof(double);
  size_t sizeC = 1 * ((size_t)1 << 30) / sizeof(double);
  double* dA;
  double* dC;
  GPU_ERROR(cudaMalloc(&dA, sizeof(double) * sizeA));
  GPU_ERROR(cudaMalloc(&dC, sizeof(double) * sizeC));
  initKernel<<<52, 256>>>(dA, sizeA);
  initKernel<<<52, 256>>>(dC, sizeC);

  measure<1>(dA, dC, sizeA);
  measure<2>(dA, dC, sizeA);
  measure<3>(dA, dC, sizeA);
  measure<4>(dA, dC, sizeA);
  measure<5>(dA, dC, sizeA);
  measure<6>(dA, dC, sizeA);
  measure<7>(dA, dC, sizeA);
  measure<8>(dA, dC, sizeA);
  measure<9>(dA, dC, sizeA);
  measure<10>(dA, dC, sizeA);
  measure<11>(dA, dC, sizeA);
  measure<12>(dA, dC, sizeA);
  measure<13>(dA, dC, sizeA);
  measure<14>(dA, dC, sizeA);
  measure<15>(dA, dC, sizeA);
  measure<16>(dA, dC, sizeA);
  measure<17>(dA, dC, sizeA);
  measure<18>(dA, dC, sizeA);
  measure<19>(dA, dC, sizeA);
  measure<20>(dA, dC, sizeA);
  measure<21>(dA, dC, sizeA);
  measure<22>(dA, dC, sizeA);
  measure<23>(dA, dC, sizeA);
  measure<24>(dA, dC, sizeA);
  measure<25>(dA, dC, sizeA);
  measure<26>(dA, dC, sizeA);
  measure<27>(dA, dC, sizeA);
  measure<28>(dA, dC, sizeA);
  measure<29>(dA, dC, sizeA);
  measure<30>(dA, dC, sizeA);
  measure<31>(dA, dC, sizeA);
  measure<32>(dA, dC, sizeA);
  measure<33>(dA, dC, sizeA);
  measure<34>(dA, dC, sizeA);
  measure<35>(dA, dC, sizeA);
  measure<36>(dA, dC, sizeA);
  measure<37>(dA, dC, sizeA);
  measure<38>(dA, dC, sizeA);
  measure<39>(dA, dC, sizeA);
  measure<40>(dA, dC, sizeA);
  measure<41>(dA, dC, sizeA);
  measure<42>(dA, dC, sizeA);
  measure<43>(dA, dC, sizeA);
  measure<44>(dA, dC, sizeA);
  measure<45>(dA, dC, sizeA);
  measure<46>(dA, dC, sizeA);
  measure<47>(dA, dC, sizeA);
  measure<48>(dA, dC, sizeA);
  measure<49>(dA, dC, sizeA);
  measure<50>(dA, dC, sizeA);
  measure<51>(dA, dC, sizeA);
  measure<52>(dA, dC, sizeA);
  measure<53>(dA, dC, sizeA);
  measure<54>(dA, dC, sizeA);
  measure<55>(dA, dC, sizeA);
  measure<56>(dA, dC, sizeA);
  measure<57>(dA, dC, sizeA);
  measure<58>(dA, dC, sizeA);
  measure<59>(dA, dC, sizeA);
  measure<60>(dA, dC, sizeA);
  measure<61>(dA, dC, sizeA);
  measure<62>(dA, dC, sizeA);
  measure<63>(dA, dC, sizeA);
  measure<64>(dA, dC, sizeA);
  measure<65>(dA, dC, sizeA);
  measure<66>(dA, dC, sizeA);
  measure<67>(dA, dC, sizeA);
  measure<68>(dA, dC, sizeA);
  measure<69>(dA, dC, sizeA);
  measure<70>(dA, dC, sizeA);
  measure<71>(dA, dC, sizeA);
  measure<72>(dA, dC, sizeA);
  measure<73>(dA, dC, sizeA);
  measure<74>(dA, dC, sizeA);
  measure<75>(dA, dC, sizeA);
  measure<76>(dA, dC, sizeA);
  measure<77>(dA, dC, sizeA);
  measure<78>(dA, dC, sizeA);
  measure<79>(dA, dC, sizeA);
  measure<80>(dA, dC, sizeA);
  measure<81>(dA, dC, sizeA);
  measure<82>(dA, dC, sizeA);
  measure<83>(dA, dC, sizeA);
  measure<84>(dA, dC, sizeA);
  measure<85>(dA, dC, sizeA);
  measure<86>(dA, dC, sizeA);
  measure<87>(dA, dC, sizeA);
  measure<88>(dA, dC, sizeA);
  measure<89>(dA, dC, sizeA);
  measure<90>(dA, dC, sizeA);
  measure<91>(dA, dC, sizeA);
  measure<92>(dA, dC, sizeA);
  measure<93>(dA, dC, sizeA);
  measure<94>(dA, dC, sizeA);
  measure<95>(dA, dC, sizeA);
  measure<96>(dA, dC, sizeA);
  measure<97>(dA, dC, sizeA);
  measure<98>(dA, dC, sizeA);
  measure<99>(dA, dC, sizeA);
  measure<100>(dA, dC, sizeA);
  measure<101>(dA, dC, sizeA);
  measure<102>(dA, dC, sizeA);
  measure<103>(dA, dC, sizeA);
  measure<104>(dA, dC, sizeA);
  measure<105>(dA, dC, sizeA);
  measure<106>(dA, dC, sizeA);
  measure<107>(dA, dC, sizeA);
  measure<108>(dA, dC, sizeA);
  measure<109>(dA, dC, sizeA);
  measure<110>(dA, dC, sizeA);
  measure<111>(dA, dC, sizeA);
  measure<112>(dA, dC, sizeA);
  measure<113>(dA, dC, sizeA);
  measure<114>(dA, dC, sizeA);
  measure<115>(dA, dC, sizeA);
  measure<116>(dA, dC, sizeA);
  measure<117>(dA, dC, sizeA);
  measure<118>(dA, dC, sizeA);
  measure<119>(dA, dC, sizeA);
  measure<120>(dA, dC, sizeA);
  measure<121>(dA, dC, sizeA);
  measure<122>(dA, dC, sizeA);
  measure<123>(dA, dC, sizeA);
  measure<124>(dA, dC, sizeA);
  measure<125>(dA, dC, sizeA);
  measure<126>(dA, dC, sizeA);
  measure<127>(dA, dC, sizeA);
  measure<128>(dA, dC, sizeA);
}
