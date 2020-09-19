#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

// Functions generated from TC
extern "C" {
void _mlir_ciface_mlp1(memref::MemRefDescriptor<int32_t, 2> *I,
                       memref::MemRefDescriptor<int32_t, 2> *W1,
                       memref::MemRefDescriptor<int32_t, 1> *B1,
                       memref::MemRefDescriptor<int32_t, 2> *O1);
}

int main() {
  simulator_init();

  const int32_t M = 3;
  const int32_t N = 3;
  const int32_t B = 3;

  int32_t bufI[B][M] = {
      {1, 1, 1},
      {2, 2, 2},
      {3, 3, 3},
  };

  int32_t bufW1[M][N] = {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
  };

  int32_t bufB1[N];
  for (int i = 0; i < N; ++i) {
    bufB1[i] = 2;
  }

  int32_t bufO1[B][N] = {0};

  memref::MemRef<int32_t, 2> I((int32_t *)bufI, {B, M});
  memref::MemRef<int32_t, 2> W1((int32_t *)bufW1, {M, N});
  memref::MemRef<int32_t, 1> B1((int32_t *)bufB1, {N});
  memref::MemRef<int32_t, 2> O1((int32_t *)bufO1, {B, N});

  std::cout << "Inputs:\n";
  utility::printTensor(I);

  std::cout << "Weights:\n";
  utility::printTensor(W1);

  std::cout << "Biases:\n";
  utility::printTensor(B1);

  _mlir_ciface_mlp1(&I.memRefDesc, &W1.memRefDesc, &B1.memRefDesc,
                    &O1.memRefDesc);

  std::cout << "MLP1 result:\n";
  utility::printTensor(O1);

  simulator_terminate();
}
