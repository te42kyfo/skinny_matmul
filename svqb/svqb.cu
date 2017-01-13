#include <omp.h>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "../cu_complex.h"
#include "../dtime.hpp"
#include "../gpu_error.cuh"
#include "../tsmm/fix_blend.cuh"
#include "../tsmttsm/specsmall.cuh"
#include "../types.hpp"
#define ADD_
#include "magma.h"
#include "magma_lapack.h"

using namespace std;

#include "mblas_dd.h"
#include "mblas_qd.h"
using HP_TYPE = dd_real;

void WTW(double* dW, double* dS, int M, int K) {
  vector<double> hW(M * K);
  vector<double> hS(M * M);
  vector<HP_TYPE> hpW(M * K);
  vector<HP_TYPE> hpS(M * M);
  GPU_ERROR(
      cudaMemcpy(hW.data(), dW, M * K * sizeof(double), cudaMemcpyDefault));
  for (int i = 0; i < M * K; i++) {
    hpW[i] = hW[i];
  }

  Rgemm("n", "t", M, M, K, 1.0, hpW.data(), M, hpW.data(), M, 0.0, hpS.data(),
        M);

  for (int i = 0; i < M * M; i++) {
    hS[i] = hpS[i].x[0];
  }
  GPU_ERROR(
      cudaMemcpy(dS, hS.data(), M * M * sizeof(double), cudaMemcpyDefault));
}

template <typename T>
T* cudaCreateAndUpload(const vector<T>& hmem) {
  T* resPtr = NULL;
  GPU_ERROR(cudaMalloc(&resPtr, sizeof(T) * hmem.size()));
  GPU_ERROR(cudaMemcpy(resPtr, hmem.data(), sizeof(T) * hmem.size(),
                       cudaMemcpyDefault));
  return resPtr;
}

void printMat(double* d, int M, int N) {
  vector<double> hd(M * N);
  GPU_ERROR(
      cudaMemcpy(hd.data(), d, M * N * sizeof(double), cudaMemcpyDefault));
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      cout << hd[n * M + m] << " ";
    }
    cout << "\n";
  }
}

enum class TSMTTSM_TYPE { BLAS, SPEC, MPACK_DD };

template <int M, typename iT, TSMTTSM_TYPE tsmttsmType>
void svqb(double* dW, double* dS, double* dQ, int K) {
  const int blockCount = 52;
  vector<double> hdinv(M);
  vector<double> hS(M * M);
  vector<double> hTemp(1000 * M);
  vector<int> hiTemp(1000 * M);

  vector<double> hL(M);

  if (tsmttsmType == TSMTTSM_TYPE::BLAS)
    magma_dgemm(MagmaNoTrans, MagmaTrans, M, M, K, 1.0, dW, M, dW, M, 0.0, dS,
                M);
  else if (tsmttsmType == TSMTTSM_TYPE::SPEC)
    SPECSMALL::tsmttsm<double, iT, M, M>(blockCount, M, M, K, dW, M, 1.0, dW, M,
                                         0.0, dS, M);
  else if (tsmttsmType == TSMTTSM_TYPE::MPACK_DD)
    WTW(dW, dS, M, K);

  GPU_ERROR(cudaMemcpy(hS.data(), dS,
                       hS.size() * sizeof(decltype(hS)::value_type),
                       cudaMemcpyDefault));
  // CPU Part
  for (int i = 0; i < M; i++) {
    if (fabs(hS[i * M + i]) > 1.0e-15) {
      hdinv[i] = 1.0 / sqrt(hS[i * M + i]);
    } else {
      hdinv[i] = 0;
    }
  }
  for (int n = 0; n < M; n++) {
    for (int m = 0; m < M; m++) {
      hS[n * M + m] *= hdinv[n] * hdinv[m];
    }
  }
  //  printMat(hS.data(), M, M);
  int info = 0;
  magma_dsyevd(MagmaVec, MagmaUpper, M, hS.data(), M, hL.data(), hTemp.data(),
               hTemp.size(), hiTemp.data(), hiTemp.size(), &info);
  for (int n = 0; n < M; n++) {
    for (int m = 0; m < M; m++) {
      hS[n * M + m] *= hdinv[m] * (1.0 / sqrt(max(1.0e-15, hL[n])));
    }
  }
  GPU_ERROR(cudaMemcpy(dS, hS.data(),
                       hS.size() * sizeof(decltype(hS)::value_type),
                       cudaMemcpyDefault));
  // GPU Part
  if (tsmttsmType == TSMTTSM_TYPE::BLAS)
    magma_dgemm(MagmaTrans, MagmaNoTrans, M, K, M, 1.0, dS, M, dW, M, 0.0, dQ,
                M);
  else
    tsmm_fix_blend<double, M, M>(blockCount, M, M, K, dW, M, 1.0, dS, M, 0.0,
                                 dQ, M);
}

template <bool PQ, int M>
double getL2Error(double* dQ, int K) {
  vector<double> hS(M * M);
  auto dS = cudaCreateAndUpload(hS);
  // if (PQ) {
  SPECSMALL::tsmttsm<double, PseudoQuad, M, M>(52, M, M, K, dQ, M, 1.0, dQ, M,
                                               0.0, dS, M);
  //                                                 } else {*/
  // magma_dgemm(MagmaNoTrans, MagmaTrans, M, M, K, 1.0, dQ, M, dQ, M, 0.0, dS,
  // M);
  //}

  GPU_ERROR(cudaMemcpy(hS.data(), dS,
                       hS.size() * sizeof(decltype(hS)::value_type),
                       cudaMemcpyDefault));

  double l2 = 0.0;
  for (int n = 0; n < M; n++) {
    for (int m = 0; m < M; m++) {
      double iv = (m == n ? 1.0 : 0.0);
      double error = iv - hS[n * M + m];
      l2 += error * error;
    }
  }
  GPU_ERROR(cudaFree(dS));
  return l2;
}

template <bool PQ, int M>
double getMaxError(double* dQ, int K) {
  vector<double> hS(M * M);
  auto dS = cudaCreateAndUpload(hS);
  /*  if (PQ) {
    SPECSMALL::tsmttsm<double, PseudoQuad, M, M>(52, M, M, K, dQ, M, 1.0, dQ, M,
                                                 0.0, dS, M);
                                                 } else {*/
  magma_dgemm(MagmaNoTrans, MagmaTrans, M, M, K, 1.0, dQ, M, dQ, M, 0.0, dS, M);
  //}

  GPU_ERROR(cudaMemcpy(hS.data(), dS,
                       hS.size() * sizeof(decltype(hS)::value_type),
                       cudaMemcpyDefault));

  double maxError = 0.0;
  for (int n = 0; n < M; n++) {
    for (int m = 0; m < M; m++) {
      double iv = (m == n ? 1.0 : 0.0);
      double error = iv - hS[n * M + m];
      maxError = max(fabs(error), maxError);
    }
  }
  GPU_ERROR(cudaFree(dS));
  return maxError;
}

void randInit(double* hW, int N, double pert) {
#pragma omp parallel
  {
    random_device rd;
    mt19937_64 re(rd() * omp_get_thread_num());
    uniform_real_distribution<double> dis(-1.0, 1.0);
#pragma omp for
    for (int i = 0; i < N; i++) {
      hW[i] = 1 + pert * dis(re);
    }
  }
}

int main(int argc, char** argv) {
  magma_init();

  const int maxK = 1000000;
  const int M = 8;

  vector<double> hW(maxK * M, 1.0);
  vector<double> hQ(maxK * M);
  vector<double> hS(M * M);

  double pert = 0.000001;

  auto dW = cudaCreateAndUpload(hW);
  auto dS = cudaCreateAndUpload(hS);
  auto dQ = cudaCreateAndUpload(hQ);

  cout.precision(4);

  using SVQBFunctionType =
      std::function<void(double*, double*, double*, int K)>;

  vector<tuple<string, SVQBFunctionType, vector<double>>> svqbFunctions;
  svqbFunctions.push_back(make_tuple(
      "SPEC PQ", svqb<M, PseudoQuad, TSMTTSM_TYPE::SPEC>, vector<double>()));
  svqbFunctions.push_back(make_tuple(
      "SPEC D", svqb<M, double, TSMTTSM_TYPE::SPEC>, vector<double>()));
  svqbFunctions.push_back(make_tuple(
      "BLAS D", svqb<M, double, TSMTTSM_TYPE::BLAS>, vector<double>()));
  svqbFunctions.push_back(make_tuple(
      "MPACK DD", svqb<M, double, TSMTTSM_TYPE::MPACK_DD>, vector<double>()));

  int K = maxK;
  cout << "\n" << K << "\n";
  for (int i = 0; i < 10; i++) {
    randInit(hW.data(), M * K, pert);
    GPU_ERROR(
        cudaMemcpy(dW, hW.data(), M * K * sizeof(double), cudaMemcpyDefault));

    for (auto& svqbFunc : svqbFunctions) {
      get<1>(svqbFunc)(dW, dS, dQ, K);
      get<2>(svqbFunc).push_back(getL2Error<true, M>(dQ, K));
    }
    cout << ".";
    cout.flush();
  }
  cout << "\n";
  for (auto& svqbFunc : svqbFunctions) {
    double av = accumulate(begin(get<2>(svqbFunc)), end(get<2>(svqbFunc)), 0.0);
    cout << get<0>(svqbFunc) << ": " << av / get<2>(svqbFunc).size() << "\n";
  }
}
