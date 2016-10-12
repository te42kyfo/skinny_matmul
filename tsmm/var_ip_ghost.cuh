#pragma once
#include "../eq.cuh"

extern __shared__ char shmem[];

template <typename T>
__global__ static void var_ip_ghost_kernel(T *x, const T *const __restrict__ w,
                                           const T alpha, const T beta,
                                           int nrows, int stridex, int stridew,
                                           int NCOLSOUT, int NCOLSIN) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  int row_in_shared = blockDim.x * threadIdx.y;
  int m;
  T *shared = (T *)shmem;

  for (; row < nrows; row += gridDim.x * blockDim.y) {
    shared[threadIdx.x + row_in_shared] =
        scale<T>(x[row * stridex + threadIdx.x], beta);
    for (m = 0; m < NCOLSIN; m++) {
      shared[threadIdx.x + row_in_shared] =
          axpy<T, T>(shared[threadIdx.x + row_in_shared], alpha,
                     scale<T>(__ldg(&x[row * stridex + m]),
                              __ldg(&w[threadIdx.x * stridew + m])));
    }
    __syncthreads();
    x[row * stridex + threadIdx.x] = shared[threadIdx.x + row_in_shared];
  }
}

template <typename T>
bool tsmm_var_ip_ghost(const size_t blockCount, const int varM, const int varN,
                       const int K, const T *A, const int lda, const T alpha,
                       const T *B, const int ldb, const T beta, T *C,
                       const int ldc) {
  if (A != C) return false;

  const int BLOCKSIZE = 128;

  size_t reqSmem = BLOCKSIZE * varM * sizeof(T);

  dim3 block, grid;
  block.x = varN;
  block.y = BLOCKSIZE / block.x;
  block.z = 1;
  grid.x = K / block.y + 1;
  grid.y = 1;
  grid.z = 1;

  var_ip_ghost_kernel<T><<<grid, block, reqSmem>>>(C, B, alpha, beta, K, ldc,
                                                   ldb, varN, varM);

  return true;
}
