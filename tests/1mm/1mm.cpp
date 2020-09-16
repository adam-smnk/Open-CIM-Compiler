#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#define ENABLE_CIM

#define CIM_PRECISION int8_t
#define MAT_SIZE 64
#define MAT_RANK 2

static CIM_PRECISION matA[MAT_SIZE][MAT_SIZE];
static CIM_PRECISION matB[MAT_SIZE][MAT_SIZE];
static CIM_PRECISION matC[MAT_SIZE][MAT_SIZE];

// Functions generated from TC
extern "C" {
void _mlir_ciface_kernel1mm(memref::MemRefDescriptor<CIM_PRECISION, 2> *A,
                            memref::MemRefDescriptor<CIM_PRECISION, 2> *B,
                            memref::MemRefDescriptor<CIM_PRECISION, 2> *C);
}

void set_matrix() {
  printf("Setting matrix values\n");
  for (int i = 0; i < MAT_SIZE; ++i) {
    for (int j = 0; j < MAT_SIZE; ++j) {
      matA[i][j] = 0xFF << (8 * (j % 2));
      matB[i][j] = ((j + i) << 4) % 0xFF;
      matC[i][j] = 0;
    }
  }
  printf("Done\n");
}

int main() {
  simulator_init();

  set_matrix();

  memref::MemRef<CIM_PRECISION, MAT_RANK> A((CIM_PRECISION *)matA,
                                            {MAT_SIZE, MAT_SIZE});
  memref::MemRef<CIM_PRECISION, MAT_RANK> B((CIM_PRECISION *)matB,
                                            {MAT_SIZE, MAT_SIZE});
  memref::MemRef<CIM_PRECISION, MAT_RANK> C((CIM_PRECISION *)matC,
                                            {MAT_SIZE, MAT_SIZE});

  std::cout << "A Matrix:\n";
  utility::printMatrix(A);

  std::cout << "B Matrix:\n";
  utility::printMatrix(B);

#ifdef ENABLE_CIM
  _mlir_ciface_kernel1mm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
#else
  utility::computeGemm(A, B, C);
#endif // ENABLE_CIM

  std::cout << "GEMM result:\n";
  utility::printMatrix(C);

  simulator_terminate();
}
