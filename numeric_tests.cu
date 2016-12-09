#include <ieee754.h>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "cu_complex.h"
#include "gpu_error.cuh"
#include "types.hpp"
#include "versions.hpp"

using namespace std;

void dispSplit(double d) {
  ieee754_double v;
  v.d = d;
  cout << bitset<20>(v.ieee.mantissa0) << bitset<32>(v.ieee.mantissa1)
       << "  *2^";
  cout << v.ieee.exponent - 1023 << "\n";
}

int main(int argc, char** argv) {
  int N = 111113;
  vector<double> hda(N);
  vector<double> hdb(N);
  vector<double> hdc(1);
  vector<PseudoQuad> hpqc(1);
  double* dda;
  double* ddb;
  double* ddc;
  double* dpqc;

  GPU_ERROR(cudaMalloc(&dda, sizeof(double) * N));
  GPU_ERROR(cudaMalloc(&ddb, sizeof(double) * N));
  GPU_ERROR(cudaMalloc(&ddc, sizeof(double) * 1));
  GPU_ERROR(cudaMalloc(&dpqc, sizeof(PseudoQuad) * 1));

  double bigVal = ((uint64_t)1 << 52);
  bigVal *= 1024;
  double smallVal1 = 0.1;
  double smallVal2 = 0.1;

  for (int i = 0; i < N; i++) {
    hda[i] = smallVal1;
    hdb[i] = smallVal2;
  }
  hda[0] = bigVal;
  hda[N - 1] = bigVal;

  hdb[0] = bigVal;
  hdb[N - 1] = -bigVal;

  GPU_ERROR(cudaMemcpy(dda, hda.data(), N * sizeof(double), cudaMemcpyDefault));
  GPU_ERROR(cudaMemcpy(ddb, hdb.data(), N * sizeof(double), cudaMemcpyDefault));

  GENV3::tsmttsm<double, PseudoQuad, 1, 1>(54, 1, 1, N, dda, 1, 1.0, ddb, 1, 0.0,
                                       ddc, 1);

  GPU_ERROR(cudaMemcpy(hdc.data(), ddc, sizeof(double), cudaMemcpyDefault));
  cout << hdc[0] << "   ";
  dispSplit(hdc[0]);

  cout << smallVal1 * smallVal2 * (N-2) << "   ";
  dispSplit(smallVal1 * smallVal2 * (N-2));
}
