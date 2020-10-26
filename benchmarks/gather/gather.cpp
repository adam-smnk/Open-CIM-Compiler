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

static const size_t rankA = 1;
static const size_t rankB = 2;
static const size_t rankC = 2;

// Functions generated from TC
extern "C" {
void _mlir_ciface_gather(memref::MemRefDescriptor<CIM_PRECISION, rankA> *X,
                         memref::MemRefDescriptor<CIM_PRECISION, rankB> *I,
                         memref::MemRefDescriptor<CIM_PRECISION, rankC> *Z);
}

int main() {
  simulator_init();

  const int32_t N = DIM_SIZE;
  const int32_t A = DIM_SIZE;
  const int32_t B = DIM_SIZE;

  std::array<int64_t, rankA> xDim = {N};
  std::array<int64_t, rankB> iDim = {A, B};
  std::array<int64_t, rankC> zDim = {A, B};

  const int xSize = utility::tensorSize<>(xDim);
  const int bSize = utility::tensorSize<>(iDim);
  const int zSize = utility::tensorSize<>(zDim);

  CIM_PRECISION bufX[xSize];
  for (int i = 0; i < xSize; ++i) {
    bufX[i] = i;
  }

  CIM_PRECISION bufI[bSize];
  for (int i = 0; i < bSize; ++i) {
    bufI[i] = i % N;
  }

  CIM_PRECISION bufZ[zSize];
  for (int i = 0; i < zSize; ++i) {
    bufZ[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, rankA> X((CIM_PRECISION *)bufX, xDim);
  memref::MemRef<CIM_PRECISION, rankB> I((CIM_PRECISION *)bufI, iDim);
  memref::MemRef<CIM_PRECISION, rankC> Z((CIM_PRECISION *)bufZ, zDim);

  std::cout << "X Input:\n";
#ifdef BENCH_PRINT
  utility::printTensor(X);
#else
  utility::printDimensions(X);
#endif // BENCH_PRINT

  std::cout << "I Indices:\n";
#ifdef BENCH_PRINT
  utility::printTensor(I);
#else
  utility::printDimensions(I);
#endif // BENCH_PRINT

  simulator_mark_start();
  _mlir_ciface_gather(&X.memRefDesc, &I.memRefDesc, &Z.memRefDesc);
  simulator_mark_end();

  std::cout << "Z Output:\n";
#ifdef BENCH_PRINT
  utility::printTensor(Z);
#else
  utility::printDimensions(Z);
#endif // BENCH_PRINT

  simulator_terminate();
}
