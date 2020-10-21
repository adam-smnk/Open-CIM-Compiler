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
void _mlir_ciface_kernel3mm(memref::MemRefDescriptor<CIM_PRECISION, rank> *A,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *B,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *C,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *D,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *E,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *F,
                            memref::MemRefDescriptor<CIM_PRECISION, rank> *G);
}

int main() {
  simulator_init();

  const int32_t M = DIM_SIZE;
  const int32_t N = DIM_SIZE;
  const int32_t K = DIM_SIZE;
  const int32_t P = DIM_SIZE;
  const int32_t Q = DIM_SIZE;

  std::array<int64_t, rank> aDim = {M, K};
  std::array<int64_t, rank> bDim = {K, N};
  std::array<int64_t, rank> cDim = {N, Q};
  std::array<int64_t, rank> dDim = {Q, P};
  std::array<int64_t, rank> eDim = {M, N};
  std::array<int64_t, rank> fDim = {N, P};
  std::array<int64_t, rank> gDim = {M, P};

  const int aSize = utility::tensorSize<>(aDim);
  const int bSize = utility::tensorSize<>(bDim);
  const int cSize = utility::tensorSize<>(cDim);
  const int dSize = utility::tensorSize<>(dDim);
  const int eSize = utility::tensorSize<>(eDim);
  const int fSize = utility::tensorSize<>(fDim);
  const int gSize = utility::tensorSize<>(gDim);

  // Inputs
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
    matD[i] = i;
  }

  // Outputs
  CIM_PRECISION matE[eSize];
  for (int i = 0; i < eSize; ++i) {
    matE[i] = 0;
  }
  CIM_PRECISION matF[fSize];
  for (int i = 0; i < fSize; ++i) {
    matF[i] = 0;
  }
  CIM_PRECISION matG[gSize];
  for (int i = 0; i < gSize; ++i) {
    matG[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, rank> A((CIM_PRECISION *)matA, aDim);
  memref::MemRef<CIM_PRECISION, rank> B((CIM_PRECISION *)matB, bDim);
  memref::MemRef<CIM_PRECISION, rank> C((CIM_PRECISION *)matC, cDim);
  memref::MemRef<CIM_PRECISION, rank> D((CIM_PRECISION *)matD, dDim);
  memref::MemRef<CIM_PRECISION, rank> E((CIM_PRECISION *)matE, eDim);
  memref::MemRef<CIM_PRECISION, rank> F((CIM_PRECISION *)matF, fDim);
  memref::MemRef<CIM_PRECISION, rank> G((CIM_PRECISION *)matG, gDim);

  std::cout << "A Matrix:\n";
  utility::printTensor(A);

  std::cout << "B Matrix:\n";
  utility::printTensor(B);

  std::cout << "C Matrix:\n";
  utility::printTensor(C);

  std::cout << "D Matrix:\n";
  utility::printTensor(D);

  simulator_mark_start();
  _mlir_ciface_kernel3mm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc,
                         &D.memRefDesc, &E.memRefDesc, &F.memRefDesc,
                         &G.memRefDesc);
  simulator_mark_end();

  std::cout << "Result:\n";
  std::cout << "E = A * B:\n";
  utility::printTensor(E);

  std::cout << "F = C * D:\n";
  utility::printTensor(F);

  std::cout << "G = C * D:\n";
  utility::printTensor(G);

  simulator_terminate();
}
