#pragma once
#include "../eq.cuh"

template <typename T, int NCOLSOUT, int NCOLSIN>
__global__ void fix_ip_ghost_kernel(T *x, const T *const __restrict__ w,
                                    const T alpha, const T beta, int nrows,
                                    int stridex, int stridew) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int m;
  T tmp[NCOLSOUT];

  for (; row < nrows; row += gridDim.x * blockDim.y) {
    tmp[threadIdx.x] = scale<T>(x[row * stridex + threadIdx.x], beta);
    for (m = 0; m < NCOLSIN; m++) {
      tmp[threadIdx.x] = axpy<T, T>(
          tmp[threadIdx.x], alpha,
          scale<T>(x[row * stridex + m], w[threadIdx.x * stridew + m]));
    }
    __syncthreads();
    x[row * stridex + threadIdx.x] = tmp[threadIdx.x];
  }
}

template <typename T, int M, int N>
bool tsmm_fix_ip_ghost(const size_t blockCount, const int varM, const int varN,
                       const int K, const T *A, const int lda, const T alpha,
                       const T *B, const int ldb, const T beta, T *C,
                       const int ldc) {
  if (varM != M || varN != N || A != C) return false;

  const int BLOCKSIZE = 128;

  dim3 block, grid;
  block.x = N;
  block.y = BLOCKSIZE / block.x;
  block.z = 1;
  grid.x = K / block.y + 1;
  grid.y = 1;
  grid.z = 1;

  fix_ip_ghost_kernel<T, N, M><<<grid, block>>>(C, B, alpha, beta, K, ldc, ldb);

  return true;
}
