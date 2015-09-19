#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <omp.h>

#include "genv1.cuh"
#include "genv2.cuh"
#include "multi_dispatch.cuh"

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

void cpuMatmul(double *A, double *B, double *result, const size_t M,
               const size_t N, const size_t K) {
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      double sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += A[k * M + m] * B[n * K + k];
      }
      result[n * M + m] = sum;
    }
  }
}

template <int MMAX, int NMAX>
void testMatmul(const size_t M, const size_t N, const size_t K,
                const int blockCount) {
  double *A, *B, *d_temp_storage, *result;

  cout << "Setup, ";
  cout.flush();
  GPU_ERROR(cudaMalloc(&A, sizeof(double) * M * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(double) * N * K));
  GPU_ERROR(cudaMalloc(&result, sizeof(double) * M * N));

  vector<double> hA(M * K);
  vector<double> hB(N * K);
  vector<double> hResult(M * N, 0);
  vector<double> hResult2(M * N, 0);
  vector<double> cpuResult(M * N);

  static int salt = 0;
  srand(time(NULL) + salt++);

  int start = rand();
#pragma omp parallel num_threads(10)
  {
    int randstate = start + omp_get_thread_num();
#pragma omp for
    for (size_t i = 0; i < M * K; i++) {
      randstate = (randstate * 7 + 11) % 101;
      hA[i] = randstate % 3 - 1;
    }
#pragma omp for
    for (size_t i = 0; i < N * K; i++) {
      randstate = (randstate * 7 + 11) % 101;
      hB[i] = randstate % 3 - 1;
    }
  }

  GPU_ERROR(
      cudaMemcpy(A, hA.data(), sizeof(double) * M * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(B, hB.data(), sizeof(double) * N * K, cudaMemcpyDefault));

  size_t temp_storage_bytes = 0;
  d_temp_storage = NULL;
  matmul_dispatch<MMAX, NMAX>::m(temp_storage_bytes, d_temp_storage, A, B,
                                 result, M, N, K, blockCount);

  GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(double) * temp_storage_bytes));

  cout << "GPU, ";
  cout.flush();

  matmul_dispatch<MMAX, NMAX>::m(temp_storage_bytes, d_temp_storage, A, B,
                                 result, M, N, K, blockCount);
  GPU_ERROR(cudaMemcpy(hResult.data(), result, sizeof(double) * M * N,
                       cudaMemcpyDefault));

  matmul_dispatch<MMAX, NMAX>::m(temp_storage_bytes, d_temp_storage, A, B,
                                 result, M, N, K, blockCount);

  GPU_ERROR(cudaMemcpy(hResult2.data(), result, sizeof(double) * M * N,
                       cudaMemcpyDefault));

  GPU_ERROR(cudaDeviceSynchronize());

  cout << "CPU, ";
  cout.flush();
  cpuMatmul(hA.data(), hB.data(), cpuResult.data(), M, N, K);

  bool passed = true;
  for (size_t i = 0; i < N * M; i++) {
    if (hResult[i] != cpuResult[i]) {
      cout << "\e[31mMismatch\e[0m\n";

      for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
          cout << hResult[n * M + m] << " \t";
        }
        cout << "\n";
      }
      cout << "--\n";

      for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
          cout << hResult2[n * M + m] << " \t";
        }
        cout << "\n";
      }
      cout << "--\n";

      for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
          cout << cpuResult[n * M + m] << " \t";
        }
        cout << "\n";
      }
      cout << "--\n\n";

      passed = false;
      break;
    }
  }
  if (passed) cout << "\e[32mPassed\e[0m (" << cpuResult[N * M / 2] << ")\n";

  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(d_temp_storage));
  GPU_ERROR(cudaFree(result));
}

int main(int argc, char **argv) {
  int sampleSize = 1;

  for (size_t M = 1; M <= 8; M++) {
    for (size_t N = 1; N <= 8; N++) {
      size_t K = (size_t)5 * 1024 * 1024 * 1024 / (M + N) / 8 * 0.01;
      for (size_t blockCount = 5 * 13; blockCount <= 6 * 13; blockCount += 13) {
        for (int t = 0; t < sampleSize; t++) {
          cout << M << "xKx" << N << "\t" << blockCount << "\t";
          testMatmul<8, 8>(M, N, K, blockCount);
        }
      }
    }
  }
  cout.flush();
}
