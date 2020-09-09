#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

// Functions generated from TC
extern "C" {
void _mlir_ciface_mm(memref::MemRefDescriptor<int32_t, 2> *A,
                     memref::MemRefDescriptor<int32_t, 2> *B,
                     memref::MemRefDescriptor<int32_t, 2> *C);

void _mlir_ciface_mm_At(memref::MemRefDescriptor<int32_t, 2> *A,
                        memref::MemRefDescriptor<int32_t, 2> *B,
                        memref::MemRefDescriptor<int32_t, 2> *C);

void _mlir_ciface_mm_Bt(memref::MemRefDescriptor<int32_t, 2> *A,
                        memref::MemRefDescriptor<int32_t, 2> *B,
                        memref::MemRefDescriptor<int32_t, 2> *C);

void _mlir_ciface_mm_AtBt(memref::MemRefDescriptor<int32_t, 2> *A,
                          memref::MemRefDescriptor<int32_t, 2> *B,
                          memref::MemRefDescriptor<int32_t, 2> *C);
}

int main() {
  simulator_init();

  // Matrix-matrix multiplication
  // C[M][N] =  A[M][K] * B[K][N]
  const size_t rank = 2;
  const int32_t M = 3;
  const int32_t N = 3;
  const int32_t K = 3;

  int32_t matA[M][K] = {
      {1, 0, 0},
      {1, 0, 0},
      {1, 0, 0},
  };

  int32_t matB[K][N] = {
      {1, 2, 3},
      {1, 2, 3},
      {1, 2, 3},
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
  std::cout << "A * B result:\n";
  utility::printMatrix(C);

  _mlir_ciface_mm_At(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
  std::cout << "A^t * B result:\n";
  utility::printMatrix(C);

  _mlir_ciface_mm_Bt(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
  std::cout << "A * B^t result:\n";
  utility::printMatrix(C);

  _mlir_ciface_mm_AtBt(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
  std::cout << "A^t * B^t result:\n";
  utility::printMatrix(C);

  simulator_terminate();
}
