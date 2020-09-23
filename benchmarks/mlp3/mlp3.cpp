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
void _mlir_ciface_mlp3(memref::MemRefDescriptor<CIM_PRECISION, 2> *I,
                       memref::MemRefDescriptor<CIM_PRECISION, 2> *W1,
                       memref::MemRefDescriptor<CIM_PRECISION, 1> *B1,
                       memref::MemRefDescriptor<CIM_PRECISION, 2> *W2,
                       memref::MemRefDescriptor<CIM_PRECISION, 1> *B2,
                       memref::MemRefDescriptor<CIM_PRECISION, 2> *W3,
                       memref::MemRefDescriptor<CIM_PRECISION, 1> *B3,
                       memref::MemRefDescriptor<CIM_PRECISION, 2> *O1,
                       memref::MemRefDescriptor<CIM_PRECISION, 2> *O2,
                       memref::MemRefDescriptor<CIM_PRECISION, 2> *O3);
}

int main() {
  simulator_init();

  const int32_t B = DIM_SIZE;
  const int32_t N = DIM_SIZE;
  const int32_t O = DIM_SIZE;
  const int32_t P = DIM_SIZE;
  const int32_t Q = DIM_SIZE;

  std::array<int64_t, 2> iDim = {B, N};
  std::array<int64_t, 2> w1Dim = {O, N};
  std::array<int64_t, 2> w2Dim = {P, O};
  std::array<int64_t, 2> w3Dim = {Q, P};
  std::array<int64_t, 1> b1Dim = {O};
  std::array<int64_t, 1> b2Dim = {P};
  std::array<int64_t, 1> b3Dim = {Q};
  std::array<int64_t, 2> o1Dim = {B, O};
  std::array<int64_t, 2> o2Dim = {B, P};
  std::array<int64_t, 2> o3Dim = {B, Q};

  const int iSize = utility::tensorSize<>(iDim);
  const int w1Size = utility::tensorSize<>(w1Dim);
  const int w2Size = utility::tensorSize<>(w2Dim);
  const int w3Size = utility::tensorSize<>(w3Dim);
  const int b1Size = utility::tensorSize<>(b1Dim);
  const int b2Size = utility::tensorSize<>(b2Dim);
  const int b3Size = utility::tensorSize<>(b3Dim);
  const int o1Size = utility::tensorSize<>(o1Dim);
  const int o2Size = utility::tensorSize<>(o2Dim);
  const int o3Size = utility::tensorSize<>(o3Dim);

  // Input
  CIM_PRECISION bufI[iSize];
  for (int i = 0; i < iSize; ++i) {
    bufI[i] = i;
  }

  // Weights
  CIM_PRECISION bufW1[w1Size];
  for (int i = 0; i < w1Size; ++i) {
    bufW1[i] = i;
  }
  CIM_PRECISION bufW2[w2Size];
  for (int i = 0; i < w2Size; ++i) {
    bufW2[i] = i;
  }
  CIM_PRECISION bufW3[w3Size];
  for (int i = 0; i < w3Size; ++i) {
    bufW3[i] = i;
  }

  // Biases
  CIM_PRECISION bufB1[b1Size];
  for (int i = 0; i < b1Size; ++i) {
    bufB1[i] = 2;
  }
  CIM_PRECISION bufB2[b2Size];
  for (int i = 0; i < b2Size; ++i) {
    bufB2[i] = 0;
  }
  CIM_PRECISION bufB3[b3Size];
  for (int i = 0; i < b3Size; ++i) {
    bufB3[i] = 1;
  }

  // Outputs
  CIM_PRECISION bufO1[o1Size];
  for (int i = 0; i < o1Size; ++i) {
    bufO1[i] = 0;
  }
  CIM_PRECISION bufO2[o2Size];
  for (int i = 0; i < o2Size; ++i) {
    bufO2[i] = 0;
  }
  CIM_PRECISION bufO3[o3Size];
  for (int i = 0; i < o3Size; ++i) {
    bufO3[i] = 0;
  }

  memref::MemRef<CIM_PRECISION, 2> I((CIM_PRECISION *)bufI, iDim);
  memref::MemRef<CIM_PRECISION, 2> W1((CIM_PRECISION *)bufW1, w1Dim);
  memref::MemRef<CIM_PRECISION, 1> B1((CIM_PRECISION *)bufB1, b1Dim);
  memref::MemRef<CIM_PRECISION, 2> O1((CIM_PRECISION *)bufO1, o1Dim);
  memref::MemRef<CIM_PRECISION, 2> W2((CIM_PRECISION *)bufW2, w2Dim);
  memref::MemRef<CIM_PRECISION, 1> B2((CIM_PRECISION *)bufB2, b2Dim);
  memref::MemRef<CIM_PRECISION, 2> O2((CIM_PRECISION *)bufO2, o2Dim);
  memref::MemRef<CIM_PRECISION, 2> W3((CIM_PRECISION *)bufW3, w3Dim);
  memref::MemRef<CIM_PRECISION, 1> B3((CIM_PRECISION *)bufB3, b3Dim);
  memref::MemRef<CIM_PRECISION, 2> O3((CIM_PRECISION *)bufO3, o3Dim);

  std::cout << "Input:\n";
  utility::printTensor(I);

  std::cout << "Weights:\n";
  std::cout << "W1:\n";
  utility::printTensor(W1);
  std::cout << "W2:\n";
  utility::printTensor(W2);
  std::cout << "W3:\n";
  utility::printTensor(W3);

  std::cout << "Biases:\n";
  std::cout << "B1:\n";
  utility::printTensor(B1);
  std::cout << "B2:\n";
  utility::printTensor(B2);
  std::cout << "B3:\n";
  utility::printTensor(B3);

  simulator_mark_start();
  _mlir_ciface_mlp3(&I.memRefDesc, &W1.memRefDesc, &B1.memRefDesc,
                    &W2.memRefDesc, &B2.memRefDesc, &W3.memRefDesc,
                    &B3.memRefDesc, &O1.memRefDesc, &O2.memRefDesc,
                    &O3.memRefDesc);
  simulator_mark_end();

  std::cout << "### MLP3 results ###\n";
  std::cout << "O1:\n";
  utility::printTensor(O1);
  std::cout << "O2:\n";
  utility::printTensor(O2);
  std::cout << "O3:\n";
  utility::printTensor(O3);

  simulator_terminate();
}
