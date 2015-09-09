#include "mxn.cuh"

void matmul(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
            double *B, double *result, const size_t K, const int blockCount) {
    MXN::MXN<PARM, PARN>(temp_storage_bytes, d_temp_storage, A, B, result, K,
                         blockCount);
}