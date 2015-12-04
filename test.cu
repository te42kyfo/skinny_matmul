#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <omp.h>

#include "skyblas.cuh"

#if !defined PARM || !defined PARN
#error "PARM or PARN is not specified! Specify M and N to test for"
#endif

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

void cpuDgemm(const Skyblas::MEMORY_ORDER AOrder,
              const Skyblas::MEMORY_ORDER BOrder, const size_t M,
              const size_t N, const size_t K, const double alpha,
              const double *A, const int lda, const double *B, const int ldb,
              const double beta, double *C, const int ldc) {
#pragma omp parallel for collapse(2)
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      double sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += A[k * lda + m] * B[k * ldb + n];
      }
      C[n * ldc + m] = C[n * ldc + m] * beta + alpha * sum;
    }
  }
}

void printMatrix(vector<double> m1, vector<double> m2, size_t N, size_t M,
                 size_t ld, string matchColor = "\e[32m",
                 string mismatchColor = "\e[31m") {
  for (size_t n = 0; n < N; n++) {
    for (size_t m = 0; m < M; m++) {
      if (m1[n * ld + m] == m2[n * ld + m])
        cout << matchColor;
      else
        cout << mismatchColor;

      cout << m1[n * ld + m] << "\e[0m\t";
    }
    cout << "\n";
  }
}

void testMatmul(Skyblas::MEMORY_ORDER AOrder, Skyblas::MEMORY_ORDER BOrder,
                size_t M, size_t N, size_t K, int lda, int ldb, int ldc,
                size_t blockCount) {
  double *A, *B, *d_temp_storage, *C;

  double alpha = 0.5;
  double beta = 0.5;

  cout << "Setup, ";
  cout.flush();
  GPU_ERROR(cudaMalloc(&A, sizeof(double) * M * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(double) * N * K));
  GPU_ERROR(cudaMalloc(&C, sizeof(double) * M * N));

  vector<double> hA(M * K);
  vector<double> hB(N * K);
  vector<double> hB2(N * K);
  vector<double> hC(M * N, 0);
  vector<double> hC2(M * N, 0);
  vector<double> cpuC(M * N);

  static int salt = 0;
  srand(time(NULL) + salt++);

  for (size_t i = 0; i < M * K; i++) {
    hA[i] = rand() % 3 - 1;
  }

  for (size_t i = 0; i < N * K; i++) {
    hB[i] = rand() % 3 - 1;
  }

  for (size_t i = 0; i < M * N; i++) {
    cpuC[i] = hC[i] = hC2[i] = rand() % 3 - 1;
  }

  GPU_ERROR(
      cudaMemcpy(A, hA.data(), sizeof(double) * M * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(B, hB.data(), sizeof(double) * N * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(C, hC.data(), sizeof(double) * N * M, cudaMemcpyDefault));

  size_t temp_storage_bytes = 0;
  d_temp_storage = NULL;

  Skyblas::dgemm<PARM, PARN>(temp_storage_bytes, d_temp_storage, blockCount,
                             AOrder, BOrder, M, N, K, alpha, A, M, B, N, beta,
                             C, N);

  GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(double) * temp_storage_bytes));

  cout << "GPU, ";
  cout.flush();

  Skyblas::dgemm<PARM, PARN>(temp_storage_bytes, d_temp_storage, blockCount,
                             AOrder, BOrder, M, N, K, alpha, A, M, B, N, beta,
                             C, N);

  GPU_ERROR(
      cudaMemcpy(hC.data(), C, sizeof(double) * M * N, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(hB.data(), B, sizeof(double) * M * N, cudaMemcpyDefault));

  Skyblas::dgemm<PARM, PARN>(temp_storage_bytes, d_temp_storage, blockCount,
                             AOrder, BOrder, M, N, K, alpha, A, M, B, N, beta,
                             C, N);

  GPU_ERROR(
      cudaMemcpy(hC2.data(), C, sizeof(double) * M * N, cudaMemcpyDefault));

  GPU_ERROR(cudaDeviceSynchronize());

  cout << "CPU, ";
  cout.flush();

  cpuDgemm(AOrder, BOrder, M, N, K, alpha, hA.data(), M, hB.data(), N, beta,
           cpuC.data(), N);

  bool passed = true;
  for (size_t i = 0; i < N * M; i++) {
    if (hC[i] != cpuC[i]) {
      cout << "\e[31mMismatch\e[0m\n";

      printMatrix(hC, cpuC, N, M, M);
      cout << "--\n";
      printMatrix(hC2, hC, N, M, M, "\e[34m");
      cout << "--\n";
      printMatrix(cpuC, cpuC, N, M, M, "\e[0m");
      cout << "--\n\n";

      passed = false;
      break;
    }
  }
  if (passed) cout << "\e[32mPassed\e[0m (" << cpuC[N * M / 2] << ")\n";

  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(d_temp_storage));
  GPU_ERROR(cudaFree(C));
}

int main(int argc, char **argv) {
  int sampleSize = 1;

  size_t M = PARM;
  size_t N = PARN;
  size_t K = (size_t)5 * 1024 * 1024 * 1024 / (M + N) / 8 * 0.01;

  for (size_t blockCount = 2 * 13; blockCount <= 8 * 13; blockCount += 2 * 13) {
    for (int t = 0; t < sampleSize; t++) {
      cout << M << "xKx" << N << "\t" << blockCount << "\t";
      testMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, K, M, N, N, blockCount);
    }
  }

  cout.flush();
}
