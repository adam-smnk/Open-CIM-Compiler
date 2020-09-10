#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>

#define CIM_PRECISION int8_t

// Functions generated from TC
extern "C" {
void _mlir_ciface_vm(memref::MemRefDescriptor<CIM_PRECISION, 1> *A,
                     memref::MemRefDescriptor<CIM_PRECISION, 2> *B,
                     memref::MemRefDescriptor<CIM_PRECISION, 1> *C);
}

int main() {
  simulator_init();

  // Vector-matrix multiplication
  // C[N] =  A[K] * B[K][N]
  const size_t matRank = 2;
  const size_t vecRank = 1;
  const int32_t N = 3;
  const int32_t K = 3;

  CIM_PRECISION matA[K] = {1, 2, 3};

  CIM_PRECISION matB[K][N] = {
      {0, 0, 1},
      {0, 1, 0},
      {1, 0, 0},
  };

  CIM_PRECISION matC[N] = {0};

  std::array<int64_t, vecRank> aDim{K};
  std::array<int64_t, matRank> bDim{K, N};
  std::array<int64_t, vecRank> cDim{N};

  memref::MemRef<CIM_PRECISION, vecRank> A((CIM_PRECISION *)matA, aDim);
  memref::MemRef<CIM_PRECISION, matRank> B((CIM_PRECISION *)matB, bDim);
  memref::MemRef<CIM_PRECISION, vecRank> C((CIM_PRECISION *)matC, cDim);

  std::cout << "A Vector:\n";
  utility::printTensor(A);

  std::cout << "B Matrix:\n";
  utility::printTensor(B);

  _mlir_ciface_vm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);

  std::cout << "GEVM result:\n";
  utility::printTensor(C);

  simulator_terminate();
}
