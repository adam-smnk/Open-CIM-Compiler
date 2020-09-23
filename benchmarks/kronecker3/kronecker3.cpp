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
void _mlir_ciface_kronecker3(memref::MemRefDescriptor<CIM_PRECISION, 2> *W0,
                             memref::MemRefDescriptor<CIM_PRECISION, 2> *W1,
                             memref::MemRefDescriptor<CIM_PRECISION, 2> *W2,
                             memref::MemRefDescriptor<CIM_PRECISION, 4> *X,
                             memref::MemRefDescriptor<CIM_PRECISION, 4> *Y,
                             memref::MemRefDescriptor<CIM_PRECISION, 4> *XW1,
                             memref::MemRefDescriptor<CIM_PRECISION, 4> *XW2);
}

int main() {
  simulator_init();

  const int32_t D0 = DIM_SIZE / 2;
  const int32_t N0 = DIM_SIZE;
  const int32_t D1 = DIM_SIZE / 2;
  const int32_t N1 = DIM_SIZE;
  const int32_t D2 = DIM_SIZE / 2;
  const int32_t N2 = DIM_SIZE;
  const int32_t M = DIM_SIZE / 2;

  std::array<int64_t, 2> w0Dim = {D0, N0};
  std::array<int64_t, 2> w1Dim = {D1, N1};
  std::array<int64_t, 2> w2Dim = {D2, N2};
  std::array<int64_t, 4> xDim = {M, N0, N1, N2};
  std::array<int64_t, 4> yDim = {M, D0, D1, D2};
  std::array<int64_t, 4> xw1Dim = {M, N0, D1, D2};
  std::array<int64_t, 4> xw2Dim = {M, N0, N1, D2};

  const int w0Size = utility::tensorSize<>(w0Dim);
  const int w1Size = utility::tensorSize<>(w1Dim);
  const int w2Size = utility::tensorSize<>(w2Dim);
  const int xSize = utility::tensorSize<>(xDim);
  const int ySize = utility::tensorSize<>(yDim);
  const int xw1Size = utility::tensorSize<>(xw1Dim);
  const int xw2Size = utility::tensorSize<>(xw2Dim);

  // Input
  CIM_PRECISION bufX[xSize];
  for (int i = 0; i < xSize; ++i) {
    bufX[i] = i;
  }

  // Weights
  CIM_PRECISION bufW0[w0Size];
  for (int i = 0; i < w0Size; ++i) {
    bufW0[i] = i;
  }
  CIM_PRECISION bufW1[w1Size];
  for (int i = 0; i < w1Size; ++i) {
    bufW1[i] = i;
  }
  CIM_PRECISION bufW2[w2Size];
  for (int i = 0; i < w2Size; ++i) {
    bufW2[i] = i;
  }

  // Outputs
  CIM_PRECISION bufY[ySize];
  for (int i = 0; i < ySize; ++i) {
    bufY[i] = 0;
  }
  CIM_PRECISION bufXW1[xw1Size];
  for (int i = 0; i < xw1Size; ++i) {
    bufXW1[i] = 0;
  }
  CIM_PRECISION bufXW2[xw2Size];
  for (int i = 0; i < xw2Size; ++i) {
    bufXW2[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, 2> W0((CIM_PRECISION *)bufW0, w0Dim);
  memref::MemRef<CIM_PRECISION, 2> W1((CIM_PRECISION *)bufW1, w1Dim);
  memref::MemRef<CIM_PRECISION, 2> W2((CIM_PRECISION *)bufW2, w2Dim);
  memref::MemRef<CIM_PRECISION, 4> X((CIM_PRECISION *)bufX, xDim);
  memref::MemRef<CIM_PRECISION, 4> Y((CIM_PRECISION *)bufY, yDim);
  memref::MemRef<CIM_PRECISION, 4> XW1((CIM_PRECISION *)bufXW1, xw1Dim);
  memref::MemRef<CIM_PRECISION, 4> XW2((CIM_PRECISION *)bufXW2, xw2Dim);

  std::cout << "Input:\n";
  utility::printTensor(X);

  std::cout << "Weights:\n";
  std::cout << "W0:\n";
  utility::printTensor(W0);
  std::cout << "W1:\n";
  utility::printTensor(W1);
  std::cout << "W2:\n";
  utility::printTensor(W2);

  simulator_mark_start();
  _mlir_ciface_kronecker3(&W0.memRefDesc, &W1.memRefDesc, &W2.memRefDesc,
                          &X.memRefDesc, &Y.memRefDesc, &XW1.memRefDesc,
                          &XW2.memRefDesc);
  simulator_mark_end();

  std::cout << "### Kronecker3 results ###\n";
  std::cout << "Y:\n";
  utility::printTensor(Y);
  std::cout << "XW1:\n";
  utility::printTensor(XW1);
  std::cout << "XW2:\n";
  utility::printTensor(XW2);

  simulator_terminate();
}
