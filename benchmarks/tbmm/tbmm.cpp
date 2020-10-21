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

static const size_t rankX = 3;
static const size_t rankY = 3;
static const size_t rankZ = 3;

// Functions generated from TC
extern "C" {
void _mlir_ciface_tbmm(memref::MemRefDescriptor<CIM_PRECISION, rankX> *X,
                       memref::MemRefDescriptor<CIM_PRECISION, rankY> *Y,
                       memref::MemRefDescriptor<CIM_PRECISION, rankZ> *Z);
}

int main() {
  simulator_init();

  const int32_t B = 3;
  const int32_t M = DIM_SIZE;
  const int32_t N = DIM_SIZE;
  const int32_t K = DIM_SIZE;

  std::array<int64_t, rankX> xDim = {B, N, M};
  std::array<int64_t, rankY> yDim = {B, K, M};
  std::array<int64_t, rankZ> zDim = {B, N, K};

  const int xSize = utility::tensorSize<>(xDim);
  const int ySize = utility::tensorSize<>(yDim);
  const int zSize = utility::tensorSize<>(zDim);

  CIM_PRECISION bufX[xSize];
  for (int i = 0; i < xSize; ++i) {
    bufX[i] = i;
  }

  CIM_PRECISION bufY[ySize];
  for (int i = 0; i < ySize; ++i) {
    bufY[i] = i;
  }

  CIM_PRECISION bufZ[zSize];
  for (int i = 0; i < zSize; ++i) {
    bufZ[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, rankX> X((CIM_PRECISION *)bufX, xDim);
  memref::MemRef<CIM_PRECISION, rankY> Y((CIM_PRECISION *)bufY, yDim);
  memref::MemRef<CIM_PRECISION, rankZ> Z((CIM_PRECISION *)bufZ, zDim);

  std::cout << "X Matrix:\n";
#ifdef BENCH_PRINT
  utility::printMatrix3D(X);
#else
  utility::printDimensions(X);
#endif // BENCH_PRINT

  std::cout << "Y Matrix:\n";
#ifdef BENCH_PRINT
  utility::printMatrix3D(Y);
#else
  utility::printDimensions(Y);
#endif // BENCH_PRINT

  simulator_mark_start();
  _mlir_ciface_tbmm(&X.memRefDesc, &Y.memRefDesc, &Z.memRefDesc);
  simulator_mark_end();

  std::cout << "Z Result:\n";
#ifdef BENCH_PRINT
  utility::printMatrix3D(Z);
#else
  utility::printDimensions(Z);
#endif // BENCH_PRINT

  simulator_terminate();
}
