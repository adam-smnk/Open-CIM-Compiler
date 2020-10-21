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

static const size_t rank = 2;

// Functions generated from TC
extern "C" {
void _mlir_ciface_kernel2mm(memref::MemRefDescriptor<CIM_PRECISION, rank> *A,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *B,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *C,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *D,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *E);
}

int main() {
  simulator_init();

  const int32_t M = DIM_SIZE;
  const int32_t N = DIM_SIZE;
  const int32_t K = DIM_SIZE;
  const int32_t P = DIM_SIZE;

  std::array<int64_t, rank> aDim = {M, K};
  std::array<int64_t, rank> bDim = {K, N};
  std::array<int64_t, rank> cDim = {P, M};
  std::array<int64_t, rank> dDim = {M, N};
  std::array<int64_t, rank> eDim = {P, N};

  const int aSize = utility::tensorSize<>(aDim);
  const int bSize = utility::tensorSize<>(bDim);
  const int cSize = utility::tensorSize<>(cDim);
  const int dSize = utility::tensorSize<>(dDim);
  const int eSize = utility::tensorSize<>(eDim);

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
    matC[i] = i;
  }

  CIM_PRECISION matD[dSize];
  for (int i = 0; i < dSize; ++i) {
    matD[i] = 0;
  }

  CIM_PRECISION matE[eSize];
  for (int i = 0; i < eSize; ++i) {
    matE[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, rank> A((CIM_PRECISION *)matA, aDim);
  memref::MemRef<CIM_PRECISION, rank> B((CIM_PRECISION *)matB, bDim);
  memref::MemRef<CIM_PRECISION, rank> C((CIM_PRECISION *)matC, cDim);
  memref::MemRef<CIM_PRECISION, rank> D((CIM_PRECISION *)matD, dDim);
  memref::MemRef<CIM_PRECISION, rank> E((CIM_PRECISION *)matE, eDim);

  std::cout << "A Matrix:\n";
  utility::printTensor(A);

  std::cout << "B Matrix:\n";
  utility::printTensor(B);

  std::cout << "C Matrix:\n";
  utility::printMatrix(C);

  simulator_mark_start();
  _mlir_ciface_kernel2mm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc,
                         &D.memRefDesc, &E.memRefDesc);
  simulator_mark_end();

  std::cout << "Result:\n";
  std::cout << "D = A * B:\n";
  utility::printMatrix(D);

  std::cout << "E = C * D:\n";
  utility::printMatrix(E);

  simulator_terminate();
}
