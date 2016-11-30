#include <cuda_runtime.h>
#include <sys/time.h>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cu_complex.h"
#include "gpu_error.cuh"
#include "types.hpp"
#include "versions.hpp"

#if !defined PARM || !defined PARN
#error "PARM or PARN is not specified! Specify M and N to measure"
#endif

using namespace std;

#ifdef TSMM
bool tsmttsm_mode = false;
bool tsmm_mode = true;
#endif
#ifdef TSMTTSM
bool tsmttsm_mode = true;
bool tsmm_mode = false;
#endif

void printMatrix(const vector<htype>& m1, const vector<htype>& m2, size_t N1,
                 size_t N2, size_t stride, size_t position = 0,
                 string matchColor = "\e[32m",
                 string mismatchColor = "\e[31m") {
  const size_t range = 32;
  size_t n1 = position < range ? 0 : position - range;
  cout << " - " << n1 << " - \n";

  for (; n1 < N1 && n1 < position + range; n1++) {
    for (size_t n2 = 0; n2 < N2; n2++) {
      if (m1[n1 * stride + n2] == m2[n1 * stride + n2])
        cout << matchColor;
      else
        cout << mismatchColor;

      cout << m1[n1 * stride + n2] << "\e[0m\t";
    }
    cout << "\n";
  }
  cout << " - " << n1 << " - \n";
}

vector<htype> hA;
vector<htype> hB;
vector<htype> hC;
vector<htype> hC_test;
vector<htype> hC_reference;
size_t totalA, totalB, totalC;

dtype *A_clean, *B_clean, *C_clean;
dtype *A_dirty, *B_dirty, *C_dirty;
dtype* temp_storage;

void initMatmul() {
  hA = vector<htype>(totalA);
  hB = vector<htype>(totalB);
  hC = vector<htype>(totalC);
  hC_test = vector<htype>(totalC);
  hC_reference = vector<htype>(totalC);
  GPU_ERROR(cudaMalloc(&A_clean, sizeof(dtype) * totalA));
  GPU_ERROR(cudaMalloc(&B_clean, sizeof(dtype) * totalB));
  GPU_ERROR(cudaMalloc(&C_clean, sizeof(dtype) * totalC));
  GPU_ERROR(cudaMalloc(&A_dirty, sizeof(dtype) * totalA));
  GPU_ERROR(cudaMalloc(&B_dirty, sizeof(dtype) * totalB));
  GPU_ERROR(cudaMalloc(&C_dirty, sizeof(dtype) * totalC));

#pragma omp parallel
  {
    random_device r;
    default_random_engine gen(r());
    uniform_int_distribution<int> dis(-2, 2);
#pragma omp for
    for (size_t i = 0; i < totalA; i++) {
      hA[i] = RAND_HTYPE(dis(gen));
    }
#pragma omp for
    for (size_t i = 0; i < totalB; i++) {
      hB[i] = RAND_HTYPE(dis(gen));
    }
#pragma omp for
    for (size_t i = 0; i < totalC; i++) {
      hC[i] = RAND_HTYPE(dis(gen));
    }
  }
  GPU_ERROR(cudaMemcpy(A_clean, hA.data(), sizeof(htype) * totalA,
                       cudaMemcpyDefault));
  GPU_ERROR(cudaMemcpy(B_clean, hB.data(), sizeof(htype) * totalB,
                       cudaMemcpyDefault));
  GPU_ERROR(cudaMemcpy(C_clean, hC.data(), sizeof(htype) * totalC,
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
      cudaMemcpy(A_dirty, A_clean, sizeof(htype) * totalA, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(B_dirty, B_clean, sizeof(htype) * totalB, cudaMemcpyDefault));
  GPU_ERROR(
      cudaMemcpy(C_dirty, C_clean, sizeof(htype) * totalC, cudaMemcpyDefault));
  dtype dalpha = makeDtype(1.0);
  dtype dbeta = makeDtype(beta);
  bool result;

  if (!self) {
    result = matmulFunction(blockCount, M, N, K, A_dirty, lda, dalpha, B_dirty,
                            ldb, dbeta, C_dirty, ldc);
  } else if (tsmm_mode) {
    result = matmulFunction(blockCount, M, N, K, C_dirty, ldc, dalpha, B_dirty,
                            ldb, dbeta, C_dirty, ldc);
  } else if (M == N) {
    result = matmulFunction(blockCount, M, N, K, A_dirty, lda, dalpha, A_dirty,
                            lda, dbeta, C_dirty, ldc);
  } else {
    result = false;
  }

  GPU_ERROR(cudaMemcpy(resultDest.data(), C_dirty, sizeof(htype) * totalC,
                       cudaMemcpyDefault));
  return result;
}

enum class TESTRESULT { PASS, SKIP, FAIL };

TESTRESULT testMatmul(MatmulFunctionType matmulFunction,
                      MatmulFunctionType referenceFunction, size_t M, size_t N,
                      size_t K, int lda, int ldb, int ldc, size_t blockCount,
                      bool self, htype beta) {

  // matmulFunction does not support parameters, this is a pass
  if (!cleanMatmul(matmulFunction, M, N, K, lda, ldb, ldc, blockCount, self,
                   beta, hC_test))
    return TESTRESULT::SKIP;

  GPU_ERROR(cudaDeviceSynchronize());
  cleanMatmul(referenceFunction, M, N, K, lda, ldb, ldc, blockCount, self, beta,
              hC_reference);
  GPU_ERROR(cudaDeviceSynchronize());

  bool passed = true;

#ifdef TSMM
  size_t C1 = K;
  size_t C2 = N;
#endif
#ifdef TSMTTSM
  size_t C1 = M;
  size_t C2 = N;
#endif

  for (size_t c1 = 0; c1 < C1; c1++) {
    for (size_t c2 = 0; c2 < C2; c2++) {
      if (hC_test[c1 * ldc + c2] != hC_reference[c1 * ldc + c2]) {
        cout << "\n( " << blockCount << " blocks, " << ((self) ? "A*A" : "A*B")
             << ", beta=" << beta << ", lda=" << lda << ", ldb=" << ldb
             << ", ldc=" << ldc << ") ";
        cout << "\e[31mMismatch\e[0m at " << c1 << ", " << c2 << "; "
             << hC_test[c1 * ldc + c2] << " != " << hC_reference[c1 * ldc + c2]
             << " ";
#ifdef VERBOSE_ERRORS
        cout << "\n";
        printMatrix(hC_test, hC_reference, C2, C1, ldc, c1);
        cout << "\n--\n";
        printMatrix(hC_reference, hC_reference, C2, C1, ldc, c1);
        cout << "--\n\n";
        cout << K << " Rows\n";
#endif
        passed = false;
        break;
      }
    }
    if (!passed) break;
  }

  if (passed)
    return TESTRESULT::PASS;
  else
    return TESTRESULT::FAIL;
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

  size_t maxMatrixSize = 0.1 * ((size_t)1 << 30) / (2 * sizeof(dtype));
  totalA = maxMatrixSize;

#ifdef TSMM
  auto versions = getEnabledTSMMVersions();
  MatmulFunctionType referenceFunction = tsmm_cublas<dtype>;
  totalB = 104 * 104;
  totalC = maxMatrixSize;
#endif
#ifdef TSMTTSM
  auto versions = getEnabledTSMTTSMVersions();
  MatmulFunctionType referenceFunction = tsmttsm_cublas<dtype>;
  totalB = maxMatrixSize;
  totalC = 104 * 104;
#endif
  initMatmul();

  random_device r;
  default_random_engine gen(r());
  uniform_int_distribution<int> dis(0, 4);

  int sampleSize = 2;

  for (int M = m1; M <= m2; M++) {
    for (int N = n1; N <= n2; N++) {
      if (n1 == 0 && n2 == 0) N = M;

      for (const auto& matmulVersion : versions) {
        cout << M << "xKx" << N << "  " << matmulVersion.second << " " << types
             << " ";
        bool passed = true;

        for (int self = 0; self <= 1; self++) {
          for (htype beta = 0.0; beta <= 1.0; beta += 1.0) {
            for (int t = 0; t < sampleSize; t++) {
              for (int blockCount = 1 * 13; blockCount <= 8 * 13;
                   blockCount += 13) {
                size_t lda = M + dis(gen);
#ifdef TSMM
                size_t ldb = M + dis(gen);
                size_t ldc = (self == 1 ? max(N + dis(gen), M) : N + dis(gen));
                size_t K = maxMatrixSize / max(lda, ldc);
#endif
#ifdef TSMTTSM
                size_t ldb = N + dis(gen);
                size_t ldc = M + dis(gen);
                size_t K = maxMatrixSize / max(lda, ldb);
#endif
                K = uniform_int_distribution<int>(1, K)(gen);
                auto result =
                    testMatmul(matmulVersion.first, referenceFunction, M, N, K,
                               lda, ldb, ldc, blockCount, (self == 1), beta);
                if (result == TESTRESULT::PASS) {
                  cout << "#";
                  passed &= true;
                }
                if (result == TESTRESULT::SKIP) {
                  cout << "\e[35m-\e[0m";
                  passed &= true;
                }
                if (result == TESTRESULT::FAIL) {
                  cout << "\e[31mX\e[0m";
                  passed &= false;
                }
                cout.flush();
              }
            }
          }
        }
        if (passed)
          cout << "\e[32m\e[1m Passed \e[0m\n";
        else
          cout << "\e[31m\e[1m Failed \e[0m\n";
      }
      if (versions.size() > 1) cout << "\n";
    }
  }
  deInitMatmul();
}
