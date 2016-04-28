#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <omp.h>
#include <random>
#include <complex>
#include <string>

#include "tsmm.cuh"

#if !defined PARM || !defined PARN
#error "PARM or PARN is not specified! Specify M and N to test for"
#endif

using namespace std;

#define XSTR(s) STR(s)
#define STR(s) #s

#ifdef FC
typedef complex<float> htype;
typedef cuFloatComplex dtype;
dtype makeDtype(htype v) { return make_cuFloatComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuFloatComplex(v1, v2)
string mode = "float complex";

#elif DC
typedef complex<double> htype;
typedef cuDoubleComplex dtype;
dtype makeDtype(htype v) { return make_cuDoubleComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuDoubleComplex(v1, v2)
string mode = "double complex";

#elif FR
typedef float htype;
typedef float dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) float(v1)
string mode = "float real";

#elif DR
typedef double htype;
typedef double dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) double(v1)
string mode = "double real";

#endif

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

void cpuDgemm(const size_t M, const size_t N, const size_t K, const htype alpha,
              const htype *A, const int lda, const htype *B, const int ldb,
              const htype beta, htype *C, const int ldc) {
#pragma omp parallel for
  for (size_t k = 0; k < K; k++) {
    for (size_t n = 0; n < N; n++) {
      htype sum = 0;
      for (size_t m = 0; m < M; m++) {
        sum += A[k * lda + m] * B[m * ldb + n];
      }
      C[k * ldc + n] = C[k * ldc + n] * beta + alpha * sum;
    }
  }
}

void printMatrix(vector<htype> m1, vector<htype> m2, size_t N, size_t K,
                 size_t ldc, string matchColor = "\e[32m",
                 string mismatchColor = "\e[31m") {
  for (size_t k = 0; k < K; k++) {
    for (size_t n = 0; n < N; n++) {
      if (m1[k * ldc + n] == m2[k * ldc + n])
        cout << matchColor;
      else
        cout << mismatchColor;

      cout << m1[k * ldc + n] << "\e[0m\t";
    }
    cout << "\n";
  }
}

bool testMatmul(size_t M, size_t N, size_t K, int lda, int ldb, int ldc,
                size_t blockCount, bool self) {
  dtype *A, *B, *C;

  htype halpha = 1.0;
  htype hbeta = 0.0;

  dtype dalpha = makeDtype(halpha);
  dtype dbeta = makeDtype(hbeta);

  cout.flush();
  GPU_ERROR(cudaMalloc(&A, sizeof(dtype) * lda * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(dtype) * ldb * M));
  GPU_ERROR(cudaMalloc(&C, sizeof(dtype) * ldc * K));

  vector<htype> hA(lda * K);
  vector<htype> hB(ldb * M);
  vector<htype> hC(ldc * K, 0);
  vector<htype> cpuC(ldc * K, 0);

#pragma omp parallel
  {
    random_device r;
    default_random_engine gen(r());
    uniform_int_distribution<int> dis(-2, 2);
#pragma omp for
    for (size_t i = 0; i < lda * K; i++) {
      hA[i] = 1;//RAND_HTYPE(dis(gen));
    }
#pragma omp for
    for (size_t i = 0; i < ldb * M; i++) {
      hB[i] = 1;//RAND_HTYPE(dis(gen));
    }
#pragma omp for
    for (size_t i = 0; i < ldc * K; i++) {
      hC[i] = cpuC[i] = 1;//RAND_HTYPE(dis(gen));
    }
  }
  GPU_ERROR(
      cudaMemcpy(A, hA.data(), sizeof(htype) * lda * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(B, hB.data(), sizeof(htype) * ldb * M, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(C, hC.data(), sizeof(htype) * ldc * K, cudaMemcpyDefault));

  tsmm<dtype, PARM, PARN>(blockCount, K, dalpha, A, lda, B, ldb, dbeta, C, ldc);

  GPU_ERROR(
      cudaMemcpy(hC.data(), C, sizeof(htype) * ldc * K, cudaMemcpyDefault));

  GPU_ERROR(cudaDeviceSynchronize());

  cpuDgemm(M, N, K, halpha, hA.data(), lda, hB.data(), ldb, hbeta, cpuC.data(),
           ldc);

  bool passed = true;
  for (size_t n = 0; n < N; n++) {
    for (size_t k = 0; k < K; k++) {
      if (hC[k * ldc + n] != cpuC[k * ldc + n]) {
        cout << "\n( " << blockCount << " blocks, " << ((self) ? "A*A" : "A*B")
             << ") ";
        cout << "\e[31mMismatch\e[0m\n";

        printMatrix(hC, cpuC, N, K, ldc);
        cout << "--\n";
        printMatrix(cpuC, cpuC, N, K, ldc, "\e[0m");
        cout << "--\n\n";

        passed = false;
        break;
      }
    }
    if (!passed) break;
  }

  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(C));

  return passed;
}

int main(int argc, char **argv) {
  int sampleSize = 1;

  size_t M = PARM;
  size_t N = PARN;
  size_t K = 10;  // (size_t)5 * 1024 * 1024 * 1024 / (M + N) / 8 * 0.02;

  cout << M << "xKx" << N << "  " << mode << " ";
  bool passed = true;
  for (size_t blockCount = 1 * 13; blockCount <= 1 * 13; blockCount += 2 * 13) {
    for (int t = 0; t < sampleSize; t++) {
      size_t lda = M + rand() % 4;
      size_t ldb = N + rand() % 4;
      size_t ldc = M + rand() % 4;

      passed &= testMatmul(M, N, K, lda, ldb, ldc, blockCount, false);
      cout << ".";
      cout.flush();
    }
  }
  if (passed) cout << "\e[32m Passed \e[0m\n";
  cout.flush();
}
