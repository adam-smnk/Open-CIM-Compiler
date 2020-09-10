#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

#define CIM_PRECISION int8_t

// Functions generated from TC
extern "C" {
void _mlir_ciface_mm(memref::MemRefDescriptor<CIM_PRECISION, 2> *A,
                     memref::MemRefDescriptor<CIM_PRECISION, 2> *B,
                     memref::MemRefDescriptor<CIM_PRECISION, 2> *C);

void _mlir_ciface_mm_At(memref::MemRefDescriptor<CIM_PRECISION, 2> *A,
                        memref::MemRefDescriptor<CIM_PRECISION, 2> *B,
                        memref::MemRefDescriptor<CIM_PRECISION, 2> *C);

void _mlir_ciface_mm_Bt(memref::MemRefDescriptor<CIM_PRECISION, 2> *A,
                        memref::MemRefDescriptor<CIM_PRECISION, 2> *B,
                        memref::MemRefDescriptor<CIM_PRECISION, 2> *C);

void _mlir_ciface_mm_AtBt(memref::MemRefDescriptor<CIM_PRECISION, 2> *A,
                          memref::MemRefDescriptor<CIM_PRECISION, 2> *B,
                          memref::MemRefDescriptor<CIM_PRECISION, 2> *C);
}

int main() {
  simulator_init();

  // Matrix-matrix multiplication
  // C[M][N] =  A[M][K] * B[K][N]
  const size_t rank = 2;
  const int32_t M = 3;
  const int32_t N = 3;
  const int32_t K = 3;

  CIM_PRECISION matA[M][K] = {
      {1, 0, 0},
      {1, 0, 0},
      {1, 0, 0},
  };

  CIM_PRECISION matB[K][N] = {
      {1, 2, 3},
      {1, 2, 3},
      {1, 2, 3},
  };

  CIM_PRECISION matC[M][N] = {0};

  std::array<int64_t, rank> aDim = {M, K};
  std::array<int64_t, rank> bDim = {K, N};
  std::array<int64_t, rank> cDim = {M, N};

  memref::MemRef<CIM_PRECISION, rank> A((CIM_PRECISION *)matA, aDim);
  memref::MemRef<CIM_PRECISION, rank> B((CIM_PRECISION *)matB, bDim);
  memref::MemRef<CIM_PRECISION, rank> C((CIM_PRECISION *)matC, cDim);

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
