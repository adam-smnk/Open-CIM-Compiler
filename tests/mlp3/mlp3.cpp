#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

#define MAX_WEIGHT 5

// Functions generated from TC
extern "C" {
void _mlir_ciface_mlp3(memref::MemRefDescriptor<int32_t, 2> *I,
                       memref::MemRefDescriptor<int32_t, 2> *W1,
                       memref::MemRefDescriptor<int32_t, 1> *B1,
                       memref::MemRefDescriptor<int32_t, 2> *W2,
                       memref::MemRefDescriptor<int32_t, 1> *B2,
                       memref::MemRefDescriptor<int32_t, 2> *W3,
                       memref::MemRefDescriptor<int32_t, 1> *B3,
                       memref::MemRefDescriptor<int32_t, 2> *O1,
                       memref::MemRefDescriptor<int32_t, 2> *O2,
                       memref::MemRefDescriptor<int32_t, 2> *O3);
}

int main() {
  simulator_init();

  const int32_t B = 3;
  const int32_t N = 3;
  const int32_t O = 3;
  const int32_t P = 3;
  const int32_t Q = 3;

  // Input
  int32_t bufI[B][N] = {
      {1, 1, 1},
      {2, 2, 2},
      {3, 3, 3},
  };

  // Weights
  int32_t bufW1[O][N];
  for (int i = 0; i < O; ++i) {
    for (int j = 0; j < N; ++j) {
      bufW1[i][j] = (i + j) % MAX_WEIGHT;
    }
  }
  int32_t bufW2[P][O];
  for (int i = P - 1; i >= 0; --i) {
    for (int j = O - 1; j >= 0; --j) {
      bufW2[i][j] = ((i + j) % MAX_WEIGHT) + 1;
    }
  }
  int32_t bufW3[Q][P];
  for (int i = Q - 1; i >= 0; --i) {
    for (int j = 0; j < P; ++j) {
      bufW3[i][j] = (i + j) % MAX_WEIGHT;
    }
  }

  // Biases
  int32_t bufB1[N];
  for (int i = 0; i < N; ++i) {
    bufB1[i] = 2;
  }
  int32_t bufB2[P];
  for (int i = 0; i < P; ++i) {
    bufB2[i] = 0;
  }
  int32_t bufB3[Q];
  for (int i = 0; i < Q; ++i) {
    bufB3[i] = 1;
  }

  // Outputs
  int32_t bufO1[B][O] = {0};
  int32_t bufO2[B][P] = {0};
  int32_t bufO3[B][Q] = {0};

  memref::MemRef<int32_t, 2> I((int32_t *)bufI, {B, N});
  memref::MemRef<int32_t, 2> W1((int32_t *)bufW1, {O, N});
  memref::MemRef<int32_t, 1> B1((int32_t *)bufB1, {O});
  memref::MemRef<int32_t, 2> O1((int32_t *)bufO1, {B, O});
  memref::MemRef<int32_t, 2> W2((int32_t *)bufW2, {P, O});
  memref::MemRef<int32_t, 1> B2((int32_t *)bufB2, {P});
  memref::MemRef<int32_t, 2> O2((int32_t *)bufO2, {B, P});
  memref::MemRef<int32_t, 2> W3((int32_t *)bufW3, {Q, P});
  memref::MemRef<int32_t, 1> B3((int32_t *)bufB3, {Q});
  memref::MemRef<int32_t, 2> O3((int32_t *)bufO3, {B, Q});

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

  _mlir_ciface_mlp3(&I.memRefDesc, &W1.memRefDesc, &B1.memRefDesc,
                    &W2.memRefDesc, &B2.memRefDesc, &W3.memRefDesc,
                    &B3.memRefDesc, &O1.memRefDesc, &O2.memRefDesc,
                    &O3.memRefDesc);

  std::cout << "### MLP3 results ###\n";
  std::cout << "O1:\n";
  utility::printTensor(O1);
  std::cout << "O2:\n";
  utility::printTensor(O2);
  std::cout << "O3:\n";
  utility::printTensor(O3);

  simulator_terminate();
}
