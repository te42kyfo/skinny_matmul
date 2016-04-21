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

#include "skyblas.cuh"

#if !defined PARM || !defined PARN
#error "PARM or PARN is not specified! Specify M and N to test for"
#endif

using namespace std;

#define PREC FLOAT
#define MODE COMPLEX

#define XSTR(s) STR(s)
#define STR(s) #s


#if PREC == FLOAT && MODE == TRUE
typedef complex<float> htype;
typedef cuFloatComplex dtype;
dtype makeDtype(htype v) { return make_cuFloatComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)

#elif PREC == DOUBLE && MODE == TRUE
typedef complex<double> htype;
typedef cuDoubleComplex dtype;
dtype makeDtype(htype v) { return make_cuDoubleComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)

#elif PREC == FLOAT && MODE == FALSE
typedef float htype;
typedef float dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)

#elif PREC == DOUBLE && MODE == FALSE
typedef double htype;
typedef double dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)

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

void cpuDgemm(const Skyblas::MEMORY_ORDER AOrder,
              const Skyblas::MEMORY_ORDER BOrder, const size_t M,
              const size_t N, const size_t K, const htype alpha, const htype *A,
              const int lda, const htype *B, const int ldb, const htype beta,
              htype *C, const int ldc) {
#pragma omp parallel for
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      htype sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += A[k * lda + m] * B[k * ldb + n];
      }
      C[n * ldc + m] = C[n * ldc + m] * beta + alpha * sum;
    }
  }
}

void printMatrix(vector<htype> m1, vector<htype> m2, size_t N, size_t M,
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

bool testMatmul(Skyblas::MEMORY_ORDER AOrder, Skyblas::MEMORY_ORDER BOrder,
                size_t M, size_t N, size_t K, int lda, int ldb, int ldc,
                size_t blockCount, bool self) {
  dtype *A, *B, *d_temp_storage, *C;

  htype halpha = 1.0;
  htype hbeta = 2.0;

  dtype dalpha = makeDtype(halpha);
  dtype dbeta = makeDtype(hbeta);

  cout.flush();
  GPU_ERROR(cudaMalloc(&A, sizeof(dtype) * lda * K));
  GPU_ERROR(cudaMalloc(&B, sizeof(dtype) * ldb * K));
  GPU_ERROR(cudaMalloc(&C, sizeof(dtype) * ldc * N));

  vector<htype> hA(lda * K);
  vector<htype> hB(ldb * K);
  vector<htype> hB2(ldb * K);
  vector<htype> hC(ldc * N, 0);
  vector<htype> hC2(ldc * N, 0);
  vector<htype> cpuC(ldc * N, 0);

#pragma omp parallel
  {
    random_device r;
    default_random_engine gen(r());
    uniform_int_distribution<int> dis(-2, 2);
#pragma omp for
    for (size_t i = 0; i < lda * K; i++) {
      hA[i] = RAND_HTYPE(dis(gen));
    }
#pragma omp for
    for (size_t i = 0; i < ldb * K; i++) {
      hB[i] = RAND_HTYPE(dis(gen));
    }
#pragma omp for
    for (size_t i = 0; i < ldc * N; i++) {
      hC2[i] = hC[i] = cpuC[i] = RAND_HTYPE(dis(gen));
    }
  }
  GPU_ERROR(
      cudaMemcpy(A, hA.data(), sizeof(htype) * lda * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(B, hB.data(), sizeof(htype) * ldb * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(C, hC.data(), sizeof(htype) * ldc * N, cudaMemcpyDefault));

  size_t temp_storage_bytes = 0;
  d_temp_storage = NULL;

  if (self)
    Skyblas::dgemm<dtype, PARM, PARN>(temp_storage_bytes, d_temp_storage,
                                      blockCount, AOrder, BOrder, M, N, K,
                                      dalpha, A, lda, A, lda, dbeta, C, ldc);
  else
    Skyblas::dgemm<dtype, PARM, PARN>(temp_storage_bytes, d_temp_storage,
                                      blockCount, AOrder, BOrder, M, N, K,
                                      dalpha, A, lda, B, ldb, dbeta, C, ldc);

  GPU_ERROR(cudaMalloc(&d_temp_storage, sizeof(htype) * temp_storage_bytes));

  if (self)
    Skyblas::dgemm<dtype, PARM, PARN>(temp_storage_bytes, d_temp_storage,
                                      blockCount, AOrder, BOrder, M, N, K,
                                      dalpha, A, lda, A, lda, dbeta, C, ldc);
  else
    Skyblas::dgemm<dtype, PARM, PARN>(temp_storage_bytes, d_temp_storage,
                                      blockCount, AOrder, BOrder, M, N, K,
                                      dalpha, A, lda, B, ldb, dbeta, C, ldc);

  GPU_ERROR(
      cudaMemcpy(hC.data(), C, sizeof(htype) * ldc * N, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(C, hC2.data(), sizeof(htype) * ldc * N, cudaMemcpyDefault));

  if (self)
    Skyblas::dgemm<dtype, PARM, PARN>(temp_storage_bytes, d_temp_storage,
                                      blockCount, AOrder, BOrder, M, N, K,
                                      dalpha, A, lda, A, lda, dbeta, C, ldc);
  else
    Skyblas::dgemm<dtype, PARM, PARN>(temp_storage_bytes, d_temp_storage,
                                      blockCount, AOrder, BOrder, M, N, K,
                                      dalpha, A, lda, B, ldb, dbeta, C, ldc);

  GPU_ERROR(
      cudaMemcpy(hC2.data(), C, sizeof(htype) * ldc * N, cudaMemcpyDefault));

  GPU_ERROR(cudaDeviceSynchronize());

  if (self)
    cpuDgemm(AOrder, BOrder, M, N, K, halpha, hA.data(), lda, hA.data(), lda,
             hbeta, cpuC.data(), ldc);
  else
    cpuDgemm(AOrder, BOrder, M, N, K, halpha, hA.data(), lda, hB.data(), ldb,
             hbeta, cpuC.data(), ldc);

  bool passed = true;
  for (size_t n = 0; n < N; n++) {
    for (size_t m = 0; m < M; m++) {
      if (hC[n * ldc + m] != cpuC[n * ldc + m]) {
        cout << "\n( " << blockCount << " blocks, " << ((self) ? "A*A" : "A*B")
             << ") ";
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

  GPU_ERROR(cudaFree(A));
  GPU_ERROR(cudaFree(B));
  GPU_ERROR(cudaFree(d_temp_storage));
  GPU_ERROR(cudaFree(C));

  return passed;
}

int main(int argc, char **argv) {
  int sampleSize = 5;

  size_t M = PARM;
  size_t N = PARN;
  size_t K = (size_t)5 * 1024 * 1024 * 1024 / (M + N) / 8 * 0.02;

  cout << M << "xKx" << N << "  " << XSTR(PREC) << " " << XSTR(MODE) << " ";
  bool passed = true;
  for (size_t blockCount = 1 * 13; blockCount <= 8 * 13; blockCount += 2 * 13) {
    for (int t = 0; t < sampleSize; t++) {
      size_t lda = M + rand() % 4;
      size_t ldb = N + rand() % 4;
      size_t ldc = M + rand() % 4;
      if (M == N)
        passed &= testMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, K, lda, ldb,
                             ldc, blockCount, true);
      passed &= testMatmul(Skyblas::COLUMN, Skyblas::ROW, M, N, K, lda, ldb,
                           ldc, blockCount, false);
      cout << ".";
      cout.flush();
    }
  }
  if (passed) cout << "\e[32m Passed \e[0m\n";
  cout.flush();
}
