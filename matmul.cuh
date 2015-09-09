#pragma once

void matmul(size_t &temp_storage_bytes, double *d_temp_storage, double *A,
            double *B, double *result, const size_t M, const size_t N,
            const size_t K, const int blockCount);