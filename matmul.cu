#include "genv1.cuh"
#include "genv2.cuh"

void matmul(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
            double *B, double *result, const size_t M, const size_t N,
            const size_t K, const int blockCount) {
  if (M == PARM && N == PARN) {
    GENV2::matmul<PARM, PARN>(temp_storage_bytes, d_temp_storage, A, B, result,
                              K, blockCount);
  } else {
    std::cout << M << "," << N << " does not match instantiated " << PARM << ","
              << PARN << "\n";
  }
}
