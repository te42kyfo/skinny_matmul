#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
#include "benchdb.hpp"
#include "cu_complex.h"
#include "dtime.hpp"
#include "gpu_error.cuh"
#include "metrics.cuh"
#include "types.hpp"
#include "versions.hpp"

#if !defined PARM || !defined PARN
#error "PARM or PARN is not specified! Specify M and N to measure"
#endif

using namespace std;

#ifdef TSMM
bool tsmttsm_mode = false;
bool tsmm_mode = true;
std::string multype = "TSMM";
#endif
#ifdef TSMTTSM
bool tsmttsm_mode = true;
bool tsmm_mode = false;
std::string multype = "TSMTTSM";
#endif

__global__ void initKernel(dtype* A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = MAKE_DTYPE(idx % 3 - 1, 0);
  }
}

dtype* A;
dtype* B;
dtype* C;
size_t totalA, totalB, totalC;

void initMatmul() {
  GPU_ERROR(cudaMalloc(&A, sizeof(dtype) * totalA));
  GPU_ERROR(cudaMalloc(&B, sizeof(dtype) * totalB));
  GPU_ERROR(cudaMalloc(&C, sizeof(dtype) * totalC));
  initKernel<<<52, 256>>>(A, totalA);
  initKernel<<<52, 256>>>(B, totalB);
  initKernel<<<52, 256>>>(C, totalC);
}

void deInitMatmul() {
  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(C));
}

double measureMatmul(MatmulFunctionType matmulFunction, size_t M, size_t N,
                     size_t K, int lda, int ldb, int ldc, size_t blockCount,
                     int iters, bool self, dtype beta) {
  //  GPU_ERROR(cudaProfilerStart());
  // GPU_ERROR(cudaDeviceSynchronize());

  bool passed = true;
  double t1 = dtime();
  for (int iter = 0; iter < iters; iter++) {
    if (!self) {
      passed = matmulFunction(blockCount, M, N, K, A, lda, makeDtype(2.0), B,
                              ldb, makeDtype(beta), C, ldc);
    } else if (tsmm_mode) {
      passed = matmulFunction(blockCount, M, N, K, C, ldc, makeDtype(2.0), B,
                              ldb, makeDtype(beta), C, ldc);
    } else if (M == N) {
      passed = matmulFunction(blockCount, M, N, K, A, lda, makeDtype(2.0), A,
                              lda, makeDtype(beta), C, ldc);
    } else {
      passed = false;
    }
  }
  GPU_ERROR(cudaDeviceSynchronize());
  // GPU_ERROR(cudaProfilerStop());
  double t2 = dtime();
  double time = (t2 - t1) / iters;

  //  cout << blockCount << " " << time << "\n";
  if (!passed)
    return -time;
  else
    return time;
}

int main(int argc, char** argv) {
  BenchDB db("benchmarks.db");

  measureMetricInit();

  cudaDeviceProp prop;
  int deviceId;

  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int m1 = 0;
  int m2 = 0;
  int n1 = 0;
  int n2 = 0;
  if (argc == 2) {
    m1 = 1;
    m2 = stoi(argv[1]);
  }
  if (argc >= 3) {
    m1 = stoi(argv[1]);
    m2 = stoi(argv[2]);
  }
  if (argc == 4) {
    cout << "Incomplete set of arguments\n";
    exit(1);
  }
  if (argc == 5) {
    n1 = stoi(argv[3]);
    n2 = stoi(argv[4]);
  }
  if (argc == 1) {
    m1 = m2 = PARM;
    n1 = n2 = PARN;
  }

  size_t maxMatrixSize = 8 * ((size_t)1 << 30) / (2 * sizeof(dtype));
  totalA = maxMatrixSize;

#ifdef TSMM
  auto versions = getEnabledTSMMVersions();
  totalB = 104 * 104;
  totalC = maxMatrixSize;
#endif
#ifdef TSMTTSM
  auto versions = getEnabledTSMTTSMVersions();
  totalB = maxMatrixSize;
  totalC = 104 * 104;
#endif

  initMatmul();

  for (int M = m1; M <= m2; M++) {
    for (int N = n1; N <= n2; N++) {
      if (n1 == 0 && n2 == 0) N = M;

      size_t lda = M;
#ifdef TSMM
      size_t ldb = M;
      size_t ldc = N;
      size_t maxK = maxMatrixSize / max(lda, ldc);
#endif
#ifdef TSMTTSM
      size_t ldb = N;
      size_t ldc = M;
      size_t maxK = maxMatrixSize / max(lda, ldb);
#endif

      for (auto matmulVersion : versions) {
        size_t K = maxK / 100 + 1;
        measureMatmul(matmulVersion.first, M, N, K, lda, ldb, ldc, smCount * 4,
                      1, false, -1.0);
        double resultTime =
            measureMatmul(matmulVersion.first, M, N, K, lda, ldb, ldc,
                          smCount * 4, 1, false, -1.0);

        while (resultTime > 0 && resultTime < 0.005 && K < maxK) {
          K = min(maxK, 2 * K);
          resultTime = measureMatmul(matmulVersion.first, M, N, K, lda, ldb,
                                     ldc, smCount * 4, 1, false, -1.0);
        }

        for (int self = 0; self <= (M == N || tsmm_mode ? 1 : 0); self++) {
          if (self == 1 && tsmm_mode) ldc = max(M, N);
          for (htype beta = (tsmm_mode ? 0.0 : 1.0); beta <= 1.0; beta += 1.0) {
            double bestTime = -1;
            int bestBlockCount = 0;
            for (int blockCount = 1 * smCount; blockCount <= 8 * smCount;
                 blockCount += smCount) {
              int sampleSize = 3;
              vector<double> times(sampleSize);
              for (int t = 0; t < sampleSize; t++) {
                times[t] = measureMatmul(matmulVersion.first, M, N, K, lda, ldb,
                                         ldc, blockCount, 1, (self == 1), beta);
              }
              times.erase(remove_if(begin(times), end(times),
                                    [](double time) { return time < 0; }),
                          end(times));
              sort(times.begin(), times.end());

              if (times.size() != 0 &&
                  (times[sampleSize / 2] < bestTime || bestBlockCount == 0)) {
                bestTime = times[sampleSize / 2];
                bestBlockCount = blockCount;
              }
            }
            double flops = 0;
            double bw = 0;

            if (bestTime > 0) {
              if (tsmm_mode) {
                if (beta < 0.1) {
                  flops = M * K * N * flopsPerCell / bestTime * 1.0e-9;
                  bw = (M * K + N * K + M * N) * sizeof(dtype) / bestTime *
                       1.0e-9;
                } else {
                  flops =
                      (M * K * N + N * K) * flopsPerCell / bestTime * 1.0e-9;
                  if (self == 1) {
                    bw = (M * K + N * K + M * N) * sizeof(dtype) / bestTime *
                         1.0e-9;
                  } else {
                    bw = (M * K + 2 * N * K + M * N) * sizeof(dtype) /
                         bestTime * 1.0e-9;
                  }
                }
              }

              if (tsmttsm_mode) {
                bw = (M + (self == 1 ? 0 : N)) * K * sizeof(dtype) / bestTime /
                     1.0e9;
                flops = M * N * K * flopsPerCell / bestTime * 1.0e-9;
              }
            }
            std::function<double()> measureMatmulFunc =
                std::bind(measureMatmul, matmulVersion.first, M, N, K, lda, ldb,
                          ldc, bestBlockCount, 1, (self == 1), beta);

            if (bestTime < 0) continue;

            double eccBW = 0, DRAMreadBW = 0, DRAMwriteBW = 0, L2readBW = 0,
                   L2writeBW = 0, sharedLoadBW = 0, occupancy = 0,
                   shmemReplays = 0;
            if (matmulVersion.second != "CUBLAS") {
              DRAMreadBW =
                  measureMetric(measureMatmulFunc, "dram_read_throughput") /
                  1.e9;

              DRAMwriteBW =
                  measureMetric(measureMatmulFunc, "dram_write_throughput") /
                  1.e9;

              L2readBW =
                  measureMetric(measureMatmulFunc, "l2_read_throughput") / 1.e9;

              L2writeBW =
                  measureMetric(measureMatmulFunc, "l2_write_throughput") /
                  1.e9;

              sharedLoadBW =
                  measureMetric(measureMatmulFunc, "shared_load_throughput") /
                  1.e9;

              shmemReplays =
                  measureMetric(measureMatmulFunc, "shared_efficiency");

              occupancy =
                  measureMetric(measureMatmulFunc, "achieved_occupancy");

              eccBW = measureMetric(measureMatmulFunc, "ecc_throughput") / 1.e9;
            }
            cout << multype << " " << deviceName << " " << setw(3) << M << " "
                 << setw(3) << N << " " << setw(2) << beta << "    "
                 << (self == 1 ? "A*A" : "A*B") << "  " << setw(8)
                 << matmulVersion.second << " " << setw(9) << K << "  "
                 << setw(8) << bestBlockCount << " " << setprecision(3)
                 << setw(8) << bestTime * 1000.0 << "ms "
                 << " " << setw(5) << setprecision(3) << flops               //
                 << "  " << setw(5) << bw                                    //
                 << " - " << setprecision(2) << setw(5) << occupancy << " "  //
                 << "  " << setprecision(3) << setw(5)
                 << (DRAMreadBW + DRAMwriteBW - eccBW / 2)  //
                 << " " << setw(5) << L2readBW + L2writeBW  //
                 << " " << setw(5) << sharedLoadBW          //
                 << " " << setw(5) << shmemReplays << "  \n";

            cout.flush();
            db.insert({{"multype", "\"" + multype + "\""},
                       {"device", "\"" + deviceName + "\""},
                       {"types", "\"" + types + "\""},
                       {"M", to_string(M)},
                       {"N", to_string(N)},
                       {"name", "\"" + matmulVersion.second + "\""},
                       {"inplace", to_string(self)},
                       {"zerobeta", to_string(beta == 0)},
                       {"branch", "\"" GIT_BRANCH_NAME "\""},
                       {"usr1_name", "\"\""},
                       {"usr1_val", "\"\""}},

                      {{"blockCount", to_string(bestBlockCount)},
                       {"K", to_string(K)},
                       {"time", to_string(bestTime)},
                       {"flops", to_string(flops)},
                       {"bw", to_string(bw)},
                       {"l2bw", to_string(L2readBW + L2writeBW)},
                       {"sharedbw", to_string(sharedLoadBW)},
                       {"occupancy", to_string(occupancy)},
                       {"drambw",
                        to_string((DRAMreadBW + DRAMwriteBW - eccBW / 2))}});
          }
        }
      }
      if (versions.size() > 1) cout << "\n";
    }
  }
  deInitMatmul();
}
