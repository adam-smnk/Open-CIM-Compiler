#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

#define CIM_PRECISION int8_t

#ifndef DIM_SIZE
#define DIM_SIZE 3
#endif // DIM_SIZE

static const size_t rankA = 3;
static const size_t rankB = 3;
static const size_t rankC = 2;

// Functions generated from TC
extern "C" {
void _mlir_ciface_tc2x3x3(memref::MemRefDescriptor<CIM_PRECISION, rankA> *A,
                          memref::MemRefDescriptor<CIM_PRECISION, rankB> *B,
                          memref::MemRefDescriptor<CIM_PRECISION, rankC> *C);
}

int main() {
  simulator_init();

  const int32_t A = DIM_SIZE;
  const int32_t B = DIM_SIZE;
  const int32_t C = DIM_SIZE;
  const int32_t D = DIM_SIZE;

  std::array<int64_t, rankA> aDim = {A, C, D};
  std::array<int64_t, rankB> bDim = {D, B, C};
  std::array<int64_t, rankC> cDim = {A, B};

  const int aSize = utility::tensorSize<>(aDim);
  const int bSize = utility::tensorSize<>(bDim);
  const int cSize = utility::tensorSize<>(cDim);

  auto matA = new CIM_PRECISION[aSize];
  for (int i = 0; i < aSize; ++i) {
    matA[i] = i;
  }

  auto matB = new CIM_PRECISION[bSize];
  for (int i = 0; i < bSize; ++i) {
    matB[i] = i;
  }

  auto matC = new CIM_PRECISION[cSize];
  for (int i = 0; i < cSize; ++i) {
    matC[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, rankA> memA((CIM_PRECISION *)matA, aDim);
  memref::MemRef<CIM_PRECISION, rankB> memB((CIM_PRECISION *)matB, bDim);
  memref::MemRef<CIM_PRECISION, rankC> memC((CIM_PRECISION *)matC, cDim);

  std::cout << "A Tensor:\n";
#ifdef BENCH_PRINT
  utility::printMatrix3D(memA);
#else
  utility::printDimensions(memA);
#endif // BENCH_PRINT

  std::cout << "B Tensor:\n";
#ifdef BENCH_PRINT
  utility::printTensor(memB);
#else
  utility::printDimensions(memB);
#endif // BENCH_PRINT

  simulator_mark_start();
  _mlir_ciface_tc2x3x3(&memA.memRefDesc, &memB.memRefDesc, &memC.memRefDesc);
  simulator_mark_end();

  std::cout << "C Tensor:\n";
#ifdef BENCH_PRINT
  utility::printMatrix3D(memC);
#else
  utility::printDimensions(memC);
#endif // BENCH_PRINT

  delete[] matA;
  delete[] matB;
  delete[] matC;

  simulator_terminate();
}
