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

// Functions generated from TC
extern "C" {
void _mlir_ciface_wavenet1(
    memref::MemRefDescriptor<CIM_PRECISION, 3> *Data,
    memref::MemRefDescriptor<CIM_PRECISION, 3> *FilterWeight,
    memref::MemRefDescriptor<CIM_PRECISION, 1> *FilterBias,
    memref::MemRefDescriptor<CIM_PRECISION, 3> *GateWeight,
    memref::MemRefDescriptor<CIM_PRECISION, 1> *GateBias,
    memref::MemRefDescriptor<CIM_PRECISION, 2> *ResWeight,
    memref::MemRefDescriptor<CIM_PRECISION, 1> *ResBias,
    memref::MemRefDescriptor<CIM_PRECISION, 2> *SkipWeight,
    memref::MemRefDescriptor<CIM_PRECISION, 1> *SkipBias,
    memref::MemRefDescriptor<CIM_PRECISION, 1> *Dilation,
    memref::MemRefDescriptor<CIM_PRECISION, 3> *FilterOut,
    memref::MemRefDescriptor<CIM_PRECISION, 3> *GateOut,
    memref::MemRefDescriptor<CIM_PRECISION, 3> *NonLin,
    memref::MemRefDescriptor<CIM_PRECISION, 3> *Res,
    memref::MemRefDescriptor<CIM_PRECISION, 3> *Skip);
}

int main() {
  simulator_init();

  const int32_t B = 1;
  const int32_t DILATION_FACTOR = 4;
  const int32_t RECEPTIVE_FIELD = DIM_SIZE;
  const int32_t RESIDUAL_C = DIM_SIZE / 4;
  const int32_t DILATION_C = DIM_SIZE / 4;
  const int32_t SKIP_C = DIM_SIZE / 2;

  auto Data = utility::allocateMemRef<CIM_PRECISION, 3>(
      {B, RESIDUAL_C, RECEPTIVE_FIELD});

  auto FilterWeight =
      utility::allocateMemRef<CIM_PRECISION, 3>({DILATION_C, RESIDUAL_C, 2});
  auto FilterBias = utility::allocateMemRef<CIM_PRECISION, 1>({DILATION_C}, 1);

  auto GateWeight =
      utility::allocateMemRef<CIM_PRECISION, 3>({DILATION_C, RESIDUAL_C, 2});
  auto GateBias = utility::allocateMemRef<CIM_PRECISION, 1>({DILATION_C}, 1);

  auto ResWeight =
      utility::allocateMemRef<CIM_PRECISION, 2>({RESIDUAL_C, DILATION_C});
  auto ResBias = utility::allocateMemRef<CIM_PRECISION, 1>({RESIDUAL_C}, 1);

  auto SkipWeight =
      utility::allocateMemRef<CIM_PRECISION, 2>({SKIP_C, DILATION_C});
  auto SkipBias = utility::allocateMemRef<CIM_PRECISION, 1>({SKIP_C}, 1);

  auto Dilation =
      utility::allocateMemRef<CIM_PRECISION, 1>({DILATION_FACTOR}, 0);

  auto FilterOut = utility::allocateMemRef<CIM_PRECISION, 3>(
      {B, DILATION_C, RECEPTIVE_FIELD}, 0);
  auto GateOut = utility::allocateMemRef<CIM_PRECISION, 3>(
      {B, DILATION_C, RECEPTIVE_FIELD}, 0);
  auto NonLin = utility::allocateMemRef<CIM_PRECISION, 3>(
      {B, DILATION_C, RECEPTIVE_FIELD}, 0);
  auto Res = utility::allocateMemRef<CIM_PRECISION, 3>(
      {B, RESIDUAL_C, RECEPTIVE_FIELD}, 0);
  auto Skip = utility::allocateMemRef<CIM_PRECISION, 3>(
      {B, SKIP_C, RECEPTIVE_FIELD}, 0);

#ifdef BENCH_PRINT
  std::cout << "Data:\n";
  utility::printMatrix3D(Data);
  std::cout << "FilterWeight:\n";
  utility::printMatrix3D(FilterWeight);
  std::cout << "FilterBias:\n";
  utility::printTensor(FilterBias);
  std::cout << "GateWeight:\n";
  utility::printMatrix3D(GateWeight);
  std::cout << "GateBias:\n";
  utility::printTensor(GateBias);
  std::cout << "ResWeight:\n";
  utility::printTensor(ResWeight);
  std::cout << "ResBias:\n";
  utility::printTensor(ResBias);
  std::cout << "SkipWeight:\n";
  utility::printTensor(SkipWeight);
  std::cout << "SkipBias:\n";
  utility::printTensor(SkipBias);
  std::cout << "Dilation:\n";
  utility::printTensor(Dilation);
#else
  std::cout << "Data:\n";
  utility::printDimensions(Data);
  std::cout << "FilterWeight:\n";
  utility::printDimensions(FilterWeight);
  std::cout << "FilterBias:\n";
  utility::printDimensions(FilterBias);
  std::cout << "GateWeight:\n";
  utility::printDimensions(GateWeight);
  std::cout << "GateBias:\n";
  utility::printDimensions(GateBias);
  std::cout << "ResWeight:\n";
  utility::printDimensions(ResWeight);
  std::cout << "ResBias:\n";
  utility::printDimensions(ResBias);
  std::cout << "SkipWeight:\n";
  utility::printDimensions(SkipWeight);
  std::cout << "SkipBias:\n";
  utility::printDimensions(SkipBias);
  std::cout << "Dilation:\n";
  utility::printDimensions(Dilation);
#endif // BENCH_PRINT

  simulator_mark_start();
  _mlir_ciface_wavenet1(
      &Data.memRefDesc, &FilterWeight.memRefDesc, &FilterBias.memRefDesc,
      &GateWeight.memRefDesc, &GateBias.memRefDesc, &ResWeight.memRefDesc,
      &ResBias.memRefDesc, &SkipWeight.memRefDesc, &SkipBias.memRefDesc,
      &Dilation.memRefDesc, &FilterOut.memRefDesc, &GateOut.memRefDesc,
      &NonLin.memRefDesc, &Res.memRefDesc, &Skip.memRefDesc);
  simulator_mark_end();

  std::cout << "### Wavenet results ###\n";
#ifdef BENCH_PRINT
  std::cout << "FilterOut:\n";
  utility::printMatrix3D(FilterOut);
  std::cout << "GateOut:\n";
  utility::printMatrix3D(GateOut);
  std::cout << "NonLin:\n";
  utility::printMatrix3D(NonLin);
  std::cout << "Res:\n";
  utility::printMatrix3D(Res);
  std::cout << "Skip:\n";
  utility::printMatrix3D(Skip);
#else
  std::cout << "FilterOut:\n";
  utility::printDimensions(FilterOut);
  std::cout << "GateOut:\n";
  utility::printDimensions(GateOut);
  std::cout << "NonLin:\n";
  utility::printDimensions(NonLin);
  std::cout << "Res:\n";
  utility::printDimensions(Res);
  std::cout << "Skip:\n";
  utility::printDimensions(Skip);
#endif // BENCH_PRINT

  delete[] Data.memRefDesc.allocated;
  delete[] FilterWeight.memRefDesc.allocated;
  delete[] FilterBias.memRefDesc.allocated;
  delete[] GateWeight.memRefDesc.allocated;
  delete[] GateBias.memRefDesc.allocated;
  delete[] ResWeight.memRefDesc.allocated;
  delete[] ResBias.memRefDesc.allocated;
  delete[] SkipWeight.memRefDesc.allocated;
  delete[] SkipBias.memRefDesc.allocated;
  delete[] Dilation.memRefDesc.allocated;
  delete[] FilterOut.memRefDesc.allocated;
  delete[] GateOut.memRefDesc.allocated;
  delete[] NonLin.memRefDesc.allocated;
  delete[] Res.memRefDesc.allocated;
  delete[] Skip.memRefDesc.allocated;

  simulator_terminate();
}
