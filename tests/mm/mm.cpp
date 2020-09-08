#include "mlir_interface/memref/memref.hpp"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdio.h>

extern "C" {
#include "libs/cim.h"
#include "libs/gic.h"
#include "libs/m5ops.h"
}

// Functions generated from TC
extern "C" {
void _mlir_ciface_mm(memref::MemRefDescriptor<int32_t, 2> *A,
                     memref::MemRefDescriptor<int32_t, 2> *B,
                     memref::MemRefDescriptor<int32_t, 2> *C);
}

int main() {
  enable_caches();
#ifdef ENABLE_INTERRUPTS
  gic_init();
  gic_enable_interrupt(131);
#endif

  printf("\n\nMain starts\n\n");

  // Matrix-matrix multiplication
  // C[M][N] =  A[M][K] * B[K][N]
  const size_t rank = 2;
  const int32_t M = 3;
  const int32_t N = 3;
  const int32_t K = 3;

  int32_t matA[M][K] = {
      {0, 0, 1},
      {0, 1, 0},
      {1, 0, 0},
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

  std::cout << "A Matrix:\n";
  utility::printMatrix(A);

  std::cout << "B Matrix:\n";
  utility::printMatrix(B);

  _mlir_ciface_mm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);

  std::cout << "GEMM result:\n";
  utility::printMatrix(C);

  _mlir_ciface_mm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);

  std::cout << "C += GEMM(A, B):\n";
  utility::printMatrix(C);

  M5_EXIT();
}
