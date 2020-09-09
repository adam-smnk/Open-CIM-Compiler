#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

// Functions generated from TC
extern "C" {
void _mlir_ciface_vm(memref::MemRefDescriptor<int32_t, 1> *A,
                     memref::MemRefDescriptor<int32_t, 2> *B,
                     memref::MemRefDescriptor<int32_t, 1> *C);

void _mlir_ciface_vm_Bt(memref::MemRefDescriptor<int32_t, 1> *A,
                        memref::MemRefDescriptor<int32_t, 2> *B,
                        memref::MemRefDescriptor<int32_t, 1> *C);
}

int main() {
  simulator_init();

  // Vector-matrix multiplication
  // C[N] =  A[K] * B[K][N]
  const size_t matRank = 2;
  const size_t vecRank = 1;
  const int32_t N = 3;
  const int32_t K = 4;

  int32_t matA[K] = {1, 1, 1, 1};

  int32_t matB[K][N] = {
      {1, 2, 3},
      {1, 2, 3},
      {1, 2, 3},
      {1, 2, 3},
  };

  int32_t matBt[N][K] = {
      {1, 1, 1, 1},
      {2, 2, 2, 2},
      {3, 3, 3, 3},
  };

  int32_t matC[N] = {0};

  std::array<int64_t, vecRank> aDim{K};
  std::array<int64_t, matRank> bDim{K, N};
  std::array<int64_t, matRank> btDim{N, K};
  std::array<int64_t, vecRank> cDim{N};

  memref::MemRef<int32_t, vecRank> A((int32_t *)matA, aDim);
  memref::MemRef<int32_t, matRank> B((int32_t *)matB, bDim);
  memref::MemRef<int32_t, matRank> Bt((int32_t *)matBt, btDim);
  memref::MemRef<int32_t, vecRank> C((int32_t *)matC, cDim);

  std::cout << "A Vector:\n";
  utility::printTensor(A);

  std::cout << "B Matrix:\n";
  utility::printTensor(B);

  _mlir_ciface_vm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
  std::cout << "A * B result:\n";
  utility::printTensor(C);

  std::cout << "D Matrix = B^t:\n";
  utility::printTensor(Bt);

  // Clear output matrix
  std::fill(matC, matC + N, 0);

  _mlir_ciface_vm_Bt(&A.memRefDesc, &Bt.memRefDesc, &C.memRefDesc);
  std::cout << "A * D^t result:\n";
  utility::printTensor(C);

  simulator_terminate();
}
