#pragma once

template <int MMAX, int NMAX>
struct matmul_dispatch {
  void static m(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
                double *B, double *result, const size_t M, const size_t N,
                const size_t K, const int blockCount) {
    if (M > MMAX || N > NMAX) {
      std::cout << "No instantiated matmul variant for " << M << "xKx" << N
                << "\n";
      exit(1);
    }
    if (M == MMAX) {
      matmul_dispatch<MMAX, NMAX>::n(temp_storage_bytes, d_temp_storage, A, B,
                                     result, M, N, K, blockCount);
    } else {
      matmul_dispatch<MMAX - 1, NMAX>::m(temp_storage_bytes, d_temp_storage, A,
                                         B, result, M, N, K, blockCount);
    }
  }
  void static n(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
                double *B, double *result, const size_t M, const size_t N,
                const size_t K, const int blockCount) {
    if (N == NMAX) {
      GENV1::matmul<MMAX, NMAX>(temp_storage_bytes, d_temp_storage, A, B,
                                result, K, blockCount);
    } else {
      matmul_dispatch<MMAX, NMAX - 1>::n(temp_storage_bytes, d_temp_storage, A,
                                         B, result, M, N, K, blockCount);
    }
  }
};

template <int MMAX>
struct matmul_dispatch<MMAX, 0> {
  void static m(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
                double *B, double *result, const size_t M, const size_t N,
                const size_t K, const int blockCount) {
    std::cout << "Invalid zero or negative Matrix Size\n";
    exit(1);
  }
  void static n(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
                double *B, double *result, const size_t M, const size_t N,
                const size_t K, const int blockCount) {
    std::cout << "Invalid zero or negative Matrix Size\n";
    exit(1);
  }
};

template <int NMAX>
struct matmul_dispatch<0, NMAX> {
  void static m(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
                double *B, double *result, const size_t M, const size_t N,
                const size_t K, const int blockCount) {
    std::cout << "Invalid zero or negative Matrix Size\n";
    exit(1);
  }
  void static n(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
                double *B, double *result, const size_t M, const size_t N,
                const size_t K, const int blockCount) {
    std::cout << "Invalid zero or negative Matrix Size\n";
    exit(1);
  }
};

template <int DMAX>
struct matmul_dispatch_diagonal {
  void static d(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
                double *B, double *result, const size_t M, const size_t N,
                const size_t K, const int blockCount) {
    if (M != N) {
      std::cout << "M != N, can't use diagonal dispatch\n";
    } else if (DMAX == M) {
      GEN_CUBLAS::matmul<DMAX, DMAX>(temp_storage_bytes, d_temp_storage, A, B,
                                result, K, blockCount);
    } else {
      matmul_dispatch_diagonal<DMAX - 1>::d(temp_storage_bytes, d_temp_storage,
                                            A, B, result, M, N, K, blockCount);
    }
  }
};

template <>
struct matmul_dispatch_diagonal<0> {
  void static d(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
                double *B, double *result, const size_t M, const size_t N,
                const size_t K, const int blockCount) {
    std::cout << "Invalid Zero or negative Matrix Size\n";
  }
};
