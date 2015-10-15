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
#include "genv3.cuh"
#include "genv4.cuh"
#include "genv5.cuh"
#include "gen_cublas.cuh"
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
#pragma omp parallel for collapse(2)
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      double sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += A[k * M + m] * B[k * N + n];
      }
      result[n * M + m] = sum;
    }
  }
}

void printMatrix(vector<double> m1, vector<double> m2, size_t N, size_t M,
                 string matchColor = "\e[32m",
                 string mismatchColor = "\e[31m") {
  for (size_t n = 0; n < N; n++) {
    for (size_t m = 0; m < M; m++) {
      if (m1[n * M + m] == m2[n * M + m])
        cout << matchColor;
      else
        cout << mismatchColor;

      cout << m1[n * M + m] << "\e[0m\t";
    }
    cout << "\n";
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

  for (size_t i = 0; i < M * K; i++) {
    hA[i] = rand() % 3 - 1;
  }

  for (size_t i = 0; i < N * K; i++) {
    hB[i] = rand() % 3 - 1;
  }

  GPU_ERROR(
      cudaMemcpy(A, hA.data(), sizeof(double) * M * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(B, hB.data(), sizeof(double) * N * K, cudaMemcpyDefault));

  size_t temp_storage_bytes = 0;
  d_temp_storage = NULL;
  //  matmul_dispatch_diagonal<NMAX>::d(temp_storage_bytes, d_temp_storage, A,
  //  B,
  //                                  result, M, N, K, blockCount);
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

      printMatrix(hResult, cpuResult, N, M);
      cout << "--\n";
      printMatrix(hResult2, hResult, N, M, "\e[34m");
      cout << "--\n";
      printMatrix(cpuResult, cpuResult, N, M, "\e[0m");
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

  for (size_t N = 1; N <= 8; N++) {
    for (size_t M = 1; M <= 64; M++) {
      size_t K = (size_t)5 * 1024 * 1024 * 1024 / (M + N) / 8 * 0.01;
      for (size_t blockCount = 2 * 13; blockCount <= 8 * 13; blockCount += 2*13) {
        for (int t = 0; t < sampleSize; t++) {
          cout << M << "xKx" << N << "\t" << blockCount << "\t";
          testMatmul<64, 8>(M, N, K, blockCount);
        }
      }
    }
  }
  cout.flush();
}
