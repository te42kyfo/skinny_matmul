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
#pragma omp parallel for
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
                 size_t ldc, string matchColor = "\e[32m",
                 string mismatchColor = "\e[31m") {
  for (size_t n = 0; n < N; n++) {
    for (size_t m = 0; m < M; m++) {
      if (m1[n * ldc + m] == m2[n * ldc + m])
        cout << matchColor;
      else
        cout << mismatchColor;

      cout << m1[n * ldc + m] << "\e[0m\t";
    }
    cout << "\n";
  }
}

void testMatmul(Skyblas::MEMORY_ORDER AOrder, Skyblas::MEMORY_ORDER BOrder,
                size_t M, size_t N, size_t K, int lda, int ldb, int ldc,
                size_t blockCount) {
  double *A, *B, *d_temp_storage, *C;

  double alpha = 1.0;
  double beta = 2.0;

  cout << "Setup, ";
  cout.flush();
  GPU_ERROR(cudaMalloc(&A, sizeof(double) * lda * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(double) * ldb * K));
  GPU_ERROR(cudaMalloc(&C, sizeof(double) * ldc * N));

  vector<double> hA(lda * K);
  vector<double> hB(ldb * K);
  vector<double> hB2(ldb * K);
  vector<double> hC(ldc * N, 0);
  vector<double> hC2(ldc * N, 0);
  vector<double> cpuC(ldc * N, 0);

  static int salt = 0;
  srand(time(NULL) + salt++);

  for (size_t i = 0; i < lda * K; i++) {
    hA[i] = rand() % 3 - 1;
  }

  for (size_t i = 0; i < ldb * K; i++) {
    hB[i] = rand() % 3 - 1;
  }

  for (size_t i = 0; i < ldc * N; i++) {
    cpuC[i] = hC[i] = hC2[i] = rand() % 3 - 1;
  }

  GPU_ERROR(
      cudaMemcpy(A, hA.data(), sizeof(double) * lda * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(B, hB.data(), sizeof(double) * ldb * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(C, hC.data(), sizeof(double) * ldc * N, cudaMemcpyDefault));

  size_t temp_storage_bytes = 0;
  d_temp_storage = NULL;

  Skyblas::dgemm<PARM, PARN>(temp_storage_bytes, d_temp_storage, blockCount,
                             AOrder, BOrder, M, N, K, alpha, A, lda, B, ldb,
                             beta, C, ldc);

  GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(double) * temp_storage_bytes));

  cout << "GPU, ";
  cout.flush();

  Skyblas::dgemm<PARM, PARN>(temp_storage_bytes, d_temp_storage, blockCount,
                             AOrder, BOrder, M, N, K, alpha, A, lda, B, ldb,
                             beta, C, ldc);
  GPU_ERROR(
      cudaMemcpy(hC.data(), C, sizeof(double) * ldc * N, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(C, hC2.data(), sizeof(double) * ldc * N, cudaMemcpyDefault));

  Skyblas::dgemm<PARM, PARN>(temp_storage_bytes, d_temp_storage, blockCount,
                             AOrder, BOrder, M, N, K, alpha, A, lda, B, ldb,
                             beta, C, ldc);

  GPU_ERROR(
      cudaMemcpy(hC2.data(), C, sizeof(double) * ldc * N, cudaMemcpyDefault));

  GPU_ERROR(cudaDeviceSynchronize());

  cout << "CPU, ";
  cout.flush();

  cpuDgemm(AOrder, BOrder, M, N, K, alpha, hA.data(), lda, hB.data(), ldb, beta,
           cpuC.data(), ldc);

  bool passed = true;
  for (size_t n = 0; n < N; n++) {
    for (size_t m = 0; m < M; m++) {
      if (hC[n * ldc + m] != cpuC[n * ldc + m]) {
        cout << "\e[31mMismatch\e[0m\n";

        printMatrix(hC, cpuC, N, M, ldc);
        cout << "--\n";
        printMatrix(hC2, hC, N, M, ldc, "\e[34m");
        cout << "--\n";
        printMatrix(cpuC, cpuC, N, M, ldc, "\e[0m");
        cout << "--\n\n";

        passed = false;
        break;
      }
    }
    if (!passed) break;
  }
  if (passed)
    cout << "\e[32mPassed\e[0m (" << cpuC[N / 2 * ldc + M / 2] << ")\n";

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
    size_t lda = M + rand() % (M + 2);
    size_t ldb = N + rand() % (N + 2);
    size_t ldc = M + rand() % (N + 2);
    for (int t = 0; t < sampleSize; t++) {
      cout << M << "xKx" << N << "\t" << lda << "\t" << ldb << "\t" << ldc
           << "\t" << blockCount << "\t";
      testMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, K, lda, ldb, ldc,
                 blockCount);
    }
  }

  cout.flush();
}
