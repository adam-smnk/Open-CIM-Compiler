#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

#define CIM_PRECISION int8_t

#ifndef DIM_SIZE
#define DIM_SIZE 3
#endif // DIM_SIZE

static const size_t rankA = 2;
static const size_t rankB = 2;
static const size_t rankC = 2;

// Functions generated from TC
extern "C" {
void _mlir_ciface_kernel1mm(memref::MemRefDescriptor<CIM_PRECISION, rankA> *A,
                            memref::MemRefDescriptor<CIM_PRECISION, rankB> *B,
                            memref::MemRefDescriptor<CIM_PRECISION, rankC> *C);
}

int main() {
  simulator_init();

  const int32_t M = DIM_SIZE;
  const int32_t N = DIM_SIZE;
  const int32_t K = DIM_SIZE;

  std::array<int64_t, rankA> aDim = {M, K};
  std::array<int64_t, rankB> bDim = {K, N};
  std::array<int64_t, rankC> cDim = {M, N};

  const int aSize = utility::tensorSize<>(aDim);
  const int bSize = utility::tensorSize<>(bDim);
  const int cSize = utility::tensorSize<>(cDim);

  CIM_PRECISION matA[aSize];
  for (int i = 0; i < aSize; ++i) {
    matA[i] = i;
  }

  CIM_PRECISION matB[bSize];
  for (int i = 0; i < bSize; ++i) {
    matB[i] = i;
  }

  CIM_PRECISION matC[cSize];
  for (int i = 0; i < cSize; ++i) {
    matC[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, rankA> A((CIM_PRECISION *)matA, aDim);
  memref::MemRef<CIM_PRECISION, rankB> B((CIM_PRECISION *)matB, bDim);
  memref::MemRef<CIM_PRECISION, rankC> C((CIM_PRECISION *)matC, cDim);

  std::cout << "A Matrix:\n";
  utility::printTensor(A);

  std::cout << "B Matrix:\n";
  utility::printTensor(B);

  simulator_mark_start();
  _mlir_ciface_kernel1mm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
  simulator_mark_end();

  std::cout << "Result:\n";
  utility::printTensor(C);

  simulator_terminate();
}
