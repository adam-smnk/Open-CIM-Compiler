#include "mlir_interface/memref/memref.hpp"

#include <array>
#include <cstdint>
#include <iostream>

// Functions generated from TC
extern "C" {
void _mlir_ciface_mm(memref::MemRefDescriptor<int32_t, 2> *A,
                     memref::MemRefDescriptor<int32_t, 2> *B,
                     memref::MemRefDescriptor<int32_t, 2> *C);
}

int main() {
  // Matrix-matrix multiplication
  // C[M][N] =  A[M][K] * B[K][N]
  const size_t rank = 2;
  const int32_t M = 3;
  const int32_t N = 3;
  const int32_t K = 3;

  int32_t matA[M][K] = {
      {1, 0, 0},
      {0, 1, 0},
      {0, 0, 1},
  };

  int32_t matB[K][N] = {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };

  int32_t matC[M][N] = {0};

  std::array<int64_t, rank> aDim = {M, K};
  std::array<int64_t, rank> bDim = {K, N};
  std::array<int64_t, rank> cDim = {M, N};

  memref::MemRef<int32_t, rank> A((int32_t *)matA, aDim);
  memref::MemRef<int32_t, rank> B((int32_t *)matB, bDim);
  memref::MemRef<int32_t, rank> C((int32_t *)matC, cDim);

  _mlir_ciface_mm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
}
