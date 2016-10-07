#include <cuda_runtime.h>
#include <sys/time.h>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../cu_complex.h"
#include "../gpu_error.cuh"
#include "cublas.cuh"
#include "fix1.cuh"
#include "fix2.cuh"
#include "fix_blend.cuh"
#include "fix_fb.cuh"
#include "fix_ip_ghost.cuh"
#include "var1.cuh"

#if !defined PARM || !defined PARN
#error "PARM or PARN is not specified! Specify M and N to measure"
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
int flopsPerCell = 8;

#elif DC
typedef complex<double> htype;
typedef cuDoubleComplex dtype;
dtype makeDtype(htype v) { return make_cuDoubleComplex(v.real(), v.imag()); }
#define RAND_HTYPE(gen) htype(gen, gen)
#define MAKE_DTYPE(v1, v2) make_cuDoubleComplex(v1, v2)
string mode = "double complex";
int flopsPerCell = 8;

#elif FR
typedef float htype;
typedef float dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) float(v1)
string mode = "float real";
int flopsPerCell = 2;

#elif DR
typedef double htype;
typedef double dtype;
dtype makeDtype(htype v) { return v; }
#define RAND_HTYPE(gen) htype(gen)
#define MAKE_DTYPE(v1, v2) double(v1)
string mode = "double real";
int flopsPerCell = 2;

#endif

using MatmulFunctionType = function<bool(
    const size_t, const int, const int, const int, const dtype*, const int,
    const dtype, const dtype*, const int, const dtype, dtype*, const int)>;

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

void printMatrix(const vector<htype>& m1, const vector<htype>& m2, size_t N,
                 size_t K, size_t ldc, size_t position = 0,
                 string matchColor = "\e[32m",
                 string mismatchColor = "\e[31m") {
  const size_t range = 5;
  size_t k = position < range ? 0 : position - range;
  cout << " - " << k << " - \n";

  for (; k < K && k < position + range; k++) {
    for (size_t n = 0; n < N; n++) {
      if (m1[k * ldc + n] == m2[k * ldc + n])
        cout << matchColor;
      else
        cout << mismatchColor;

      cout << m1[k * ldc + n] << "\e[0m\t";
    }
    cout << "\n";
  }
  cout << " - " << k << " - \n";
}

vector<htype> hA;
vector<htype> hB;
vector<htype> hC;
vector<htype> hC_test;
vector<htype> hC_reference;

dtype *A_clean, *B_clean, *C_clean;
dtype *A_dirty, *B_dirty, *C_dirty;

void initMatmul(size_t M, size_t N, size_t K, int lda, int ldb, int ldc) {
  hA = vector<htype>(lda * K);
  hB = vector<htype>(ldb * M);
  hC = vector<htype>(ldc * K);
  hC_test = vector<htype>(ldc * K);
  hC_reference = vector<htype>(ldc * K);
  GPU_ERROR(cudaMalloc(&A_clean, sizeof(dtype) * lda * K));
  GPU_ERROR(cudaMalloc(&B_clean, sizeof(dtype) * ldb * M));
  GPU_ERROR(cudaMalloc(&C_clean, sizeof(dtype) * ldc * K));
  GPU_ERROR(cudaMalloc(&A_dirty, sizeof(dtype) * lda * K));
  GPU_ERROR(cudaMalloc(&B_dirty, sizeof(dtype) * ldb * M));
  GPU_ERROR(cudaMalloc(&C_dirty, sizeof(dtype) * ldc * K));

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
    for (size_t i = 0; i < ldb * M; i++) {
      hB[i] = RAND_HTYPE(dis(gen));
    }
#pragma omp for
    for (size_t i = 0; i < ldc * K; i++) {
      hC[i] = RAND_HTYPE(dis(gen));
    }
  }
  GPU_ERROR(cudaMemcpy(A_clean, hA.data(), sizeof(htype) * lda * K,
                       cudaMemcpyDefault));
  GPU_ERROR(cudaMemcpy(B_clean, hB.data(), sizeof(htype) * ldb * M,
                       cudaMemcpyDefault));
  GPU_ERROR(cudaMemcpy(C_clean, hC.data(), sizeof(htype) * ldc * K,
                       cudaMemcpyDefault));
  GPU_ERROR(cudaDeviceSynchronize());
}

void deInitMatmul() {
  GPU_ERROR(cudaFree(A_clean));
  GPU_ERROR(cudaFree(B_clean));
  GPU_ERROR(cudaFree(C_clean));
  GPU_ERROR(cudaFree(A_dirty));
  GPU_ERROR(cudaFree(B_dirty));
  GPU_ERROR(cudaFree(C_dirty));
}

bool cleanMatmul(MatmulFunctionType matmulFunction, size_t M, size_t N,
                 size_t K, int lda, int ldb, int ldc, size_t blockCount,
                 bool self, htype beta, vector<htype>& resultDest) {
  GPU_ERROR(
      cudaMemcpy(A_dirty, A_clean, sizeof(htype) * lda * K, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(B_dirty, B_clean, sizeof(htype) * ldb * M, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(C_dirty, C_clean, sizeof(htype) * ldc * K, cudaMemcpyDefault));
  dtype dalpha = makeDtype(2.0);
  dtype dbeta = makeDtype(beta);
  bool result;
  if (self) {
    result = matmulFunction(blockCount, M, N, K, C_dirty, ldc, dalpha, B_dirty,
                            ldb, dbeta, C_dirty, ldc);
  } else {
    result = matmulFunction(blockCount, M, N, K, A_dirty, lda, dalpha, B_dirty,
                            ldb, dbeta, C_dirty, ldc);
  }
  GPU_ERROR(cudaMemcpy(resultDest.data(), C_dirty, sizeof(htype) * ldc * K,
                       cudaMemcpyDefault));
  return result;
}

bool testMatmul(MatmulFunctionType matmulFunction, size_t M, size_t N, size_t K,
                int lda, int ldb, int ldc, size_t blockCount, bool self,
                htype beta) {
  // matmulFunction does not support parameters, this is a pass
  if (!cleanMatmul(matmulFunction, M, N, K, lda, ldb, ldc, blockCount, self,
                   beta, hC_test))
    return true;
  GPU_ERROR(cudaDeviceSynchronize());
  cleanMatmul(tsmm_cublas<dtype>, M, N, K, lda, ldb, ldc, blockCount, self,
              beta, hC_reference);
  GPU_ERROR(cudaDeviceSynchronize());

  bool passed = true;

  for (size_t k = 0; k < K; k++) {
    for (size_t n = 0; n < N; n++) {
      if (hC_test[k * ldc + n] != hC_reference[k * ldc + n]) {
        cout << "\n( " << blockCount << " blocks, " << ((self) ? "A*A" : "A*B")
             << ", beta=" << beta << ", lda=" << lda << ", ldb=" << ldb
             << ", ldc=" << ldc << ") ";
        cout << "\e[31mMismatch\e[0m at " << k << ", " << n << "; "
             << hC_test[k * ldc + n] << " != " << hC_reference[k * ldc + n]
             << " ";
#ifdef VERBOSE_ERRORS
        cout << "\n";
        printMatrix(hC_test, hC_reference, N, K, ldc, k);
        cout << "\n--\n";
        printMatrix(hC_reference, hC_reference, N, K, ldc, k, "\e[0m");
        cout << "--\n\n";
        cout << K << " Rows\n";
#endif
        passed = false;
        break;
      }
    }
    if (!passed) break;
  }

  return passed;
}

int main(int argc, char** argv) {
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

  vector<pair<MatmulFunctionType, string>> versions;

#if PARM != 0 && PARN != 0
#ifdef FIX_BLEND
  versions.push_back({tsmm_fix_blend<dtype, PARM, PARN>, "FBLEND"});
#endif
#ifdef FIX_FB
  versions.push_back({tsmm_fix_fb<dtype, PARM, PARN>, "FIX_FB"});
#endif
#ifdef FIX1
  versions.push_back({tsmm_fix1<dtype, PARM, PARN>, "FIX_V1"});
#endif
#ifdef FIX2
  versions.push_back({tsmm_fix2<dtype, PARM, PARN>, "FIX_V2"});
#endif
#ifdef CUBLAS
  versions.push_back({tsmm_cublas<dtype>, "CUBLAS"});
#endif
#ifdef VAR1
  versions.push_back({tsmm_var1<dtype>, "VAR_V1"});
#endif
#ifdef FIXIPG
  versions.push_back({tsmm_fix_ip_ghost<dtype, PARM, PARN>, "FIXIPG"});
#endif

#endif

  int maxK = 0.05 * ((size_t)1 << 30) / (2 * sizeof(dtype));
  initMatmul(100, 100, maxK / 208 * 2, 104, 104, 104);

  random_device r;
  default_random_engine gen(r());
  uniform_int_distribution<int> dis(0, 4);

  int sampleSize = 2;

  for (int M = m1; M <= m2; M++) {
    for (int N = n1; N <= n2; N++) {
      if (n1 == 0 && n2 == 0) N = M;
      for (const auto& matmulVersion : versions) {
        cout << M << "xKx" << N << "  " << matmulVersion.second << " " << mode
             << " ";
        if (!matmulVersion.first(0, M, N, 0, A_dirty, M, makeDtype(1.0),
                                 B_dirty, N, makeDtype(1.0), C_dirty, M)) {
          cout << "\e[35m Skipped \e[0m\n";
          continue;
        }
        bool passed = true;

        for (int self = 0; self <= (M == N) ? 1 : 0; self++) {
          for (htype beta = 0.0; beta <= 1.0; beta += 1.0) {
            for (int t = 0; t < sampleSize; t++) {
              for (int blockCount = 1 * 13; blockCount <= 8 * 13;
                   blockCount += 13) {
                size_t lda = M + dis(gen);
                size_t ldb = M + dis(gen);
                size_t ldc = N + dis(gen);
                size_t K = maxK / (lda + ldc);
                bool result = testMatmul(matmulVersion.first, M, N, K, lda, ldb,
                                         ldc, blockCount, (self == 1), beta);
                if (result)
                  cout << ".";
                else
                  cout << "x";
                passed &= result;
                cout.flush();
              }
            }
          }
        }
        if (passed)
          cout << "\e[32m Passed \e[0m\n";
        else
          cout << "\e[31m Failed \e[0m\n";
      }
      if (versions.size() > 1) cout << "\n";
    }
  }
  deInitMatmul();
}
