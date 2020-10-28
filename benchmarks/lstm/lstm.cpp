#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

#define CIM_PRECISION int8_t

#ifndef DIM_SIZE
#define DIM_SIZE 8
#endif // DIM_SIZE

// Functions generated from TC
extern "C" {
void _mlir_ciface_lstm(memref::MemRefDescriptor<CIM_PRECISION, 1> *CTprev,
                       memref::MemRefDescriptor<CIM_PRECISION, 1> *XH,
                       memref::MemRefDescriptor<CIM_PRECISION, 2> *Weights,
                       memref::MemRefDescriptor<CIM_PRECISION, 1> *States,
                       memref::MemRefDescriptor<CIM_PRECISION, 1> *CT,
                       memref::MemRefDescriptor<CIM_PRECISION, 1> *HT);
}

int main() {
  simulator_init();

  const int32_t H = DIM_SIZE / 4;
  const int32_t EH = DIM_SIZE / 2;
  const int32_t H4 = 4 * H;

  auto CTprev = utility::allocateMemRef<CIM_PRECISION, 1>({H}, 1);
  auto XH = utility::allocateMemRef<CIM_PRECISION, 1>({EH}, 1);
  auto Weights = utility::allocateMemRef<CIM_PRECISION, 2>({EH, H4});

  auto States = utility::allocateMemRef<CIM_PRECISION, 1>({H4}, 0);
  auto CT = utility::allocateMemRef<CIM_PRECISION, 1>({H}, 0);
  auto HT = utility::allocateMemRef<CIM_PRECISION, 1>({H}, 0);

#ifdef BENCH_PRINT
  std::cout << "CTprev:\n";
  utility::printTensor(CTprev);
  std::cout << "XH:\n";
  utility::printTensor(XH);
  std::cout << "Weights:\n";
  utility::printTensor(Weights);
#else
  std::cout << "CTprev:\n";
  utility::printDimensions(CTprev);
  std::cout << "XH:\n";
  utility::printDimensions(XH);
  std::cout << "Weights:\n";
  utility::printDimensions(Weights);
#endif // BENCH_PRINT

  simulator_mark_start();
  _mlir_ciface_lstm(&CTprev.memRefDesc, &XH.memRefDesc, &Weights.memRefDesc,
                    &States.memRefDesc, &CT.memRefDesc, &HT.memRefDesc);
  simulator_mark_end();

  std::cout << "### LSTM results ###\n";
#ifdef BENCH_PRINT
  std::cout << "States:\n";
  utility::printTensor(States);
  std::cout << "CT:\n";
  utility::printTensor(CT);
  std::cout << "HT:\n";
  utility::printTensor(HT);
#else
  std::cout << "States:\n";
  utility::printDimensions(States);
  std::cout << "CT:\n";
  utility::printDimensions(CT);
  std::cout << "HT:\n";
  utility::printDimensions(HT);
#endif // BENCH_PRINT

  delete[] CTprev.memRefDesc.allocated;
  delete[] XH.memRefDesc.allocated;
  delete[] Weights.memRefDesc.allocated;
  delete[] States.memRefDesc.allocated;
  delete[] CT.memRefDesc.allocated;
  delete[] HT.memRefDesc.allocated;

  simulator_terminate();
}
