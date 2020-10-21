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

static const size_t rankI = 2;
static const size_t rankMean = 1;
static const size_t rankVar = 1;

// Functions generated from TC
extern "C" {
void _mlir_ciface_moments2_2d_1D(
    memref::MemRefDescriptor<CIM_PRECISION, rankI> *I,
    memref::MemRefDescriptor<CIM_PRECISION, rankMean> *mean,
    memref::MemRefDescriptor<CIM_PRECISION, rankVar> *var);
}

int main() {
  simulator_init();

  const int32_t N = DIM_SIZE;
  const int32_t K = DIM_SIZE;

  std::array<int64_t, rankI> iDim = {N, K};
  std::array<int64_t, rankMean> meanDim = {N};
  std::array<int64_t, rankVar> varDim = {N};

  const int iSize = utility::tensorSize<>(iDim);
  const int meanSize = utility::tensorSize<>(meanDim);
  const int varSize = utility::tensorSize<>(varDim);

  CIM_PRECISION bufI[iSize];
  for (int i = 0; i < iSize; ++i) {
    bufI[i] = i;
  }

  CIM_PRECISION bufMean[meanSize];
  CIM_PRECISION bufVar[varSize];

  memref::MemRef<CIM_PRECISION, rankI> I((CIM_PRECISION *)bufI, iDim);
  memref::MemRef<CIM_PRECISION, rankMean> mean((CIM_PRECISION *)bufMean,
                                               meanDim);
  memref::MemRef<CIM_PRECISION, rankVar> var((CIM_PRECISION *)bufVar, varDim);

  std::cout << "Input I:\n";
#ifdef BENCH_PRINT
  utility::printTensor(I);
#else
  utility::printDimensions(I);
#endif // BENCH_PRINT

  simulator_mark_start();
  _mlir_ciface_moments2_2d_1D(&I.memRefDesc, &mean.memRefDesc, &var.memRefDesc);
  simulator_mark_end();

  std::cout << "Mean:\n";
#ifdef BENCH_PRINT
  utility::printTensor(mean);
#else
  utility::printDimensions(mean);
#endif // BENCH_PRINT

  std::cout << "Var:\n";
#ifdef BENCH_PRINT
  utility::printTensor(var);
#else
  utility::printDimensions(var);
#endif // BENCH_PRINT

  simulator_terminate();
}
