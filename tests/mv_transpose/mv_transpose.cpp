#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

static const size_t rankA = 2;
static const size_t rankB = 1;
static const size_t rankC = 1;

// Functions generated from TC
extern "C" {
void _mlir_ciface_mv(memref::MemRefDescriptor<int32_t, rankA> *A,
                     memref::MemRefDescriptor<int32_t, rankB> *B,
                     memref::MemRefDescriptor<int32_t, rankC> *C);

void _mlir_ciface_mv_At(memref::MemRefDescriptor<int32_t, rankA> *A,
                        memref::MemRefDescriptor<int32_t, rankB> *B,
                        memref::MemRefDescriptor<int32_t, rankC> *C);
}

int main() {
  simulator_init();

  // Matrix-vector multiplication
  // C[N] =  A[N][K] * B[K]
  const int32_t N = 3;
  const int32_t K = 4;

  int32_t matA[N][K] = {
      {1, 1, 1, 1},
      {2, 2, 2, 2},
      {3, 3, 3, 3},
  };

  int32_t matAt[K][N] = {
      {1, 2, 3},
      {1, 2, 3},
      {1, 2, 3},
      {1, 2, 3},
  };

  int32_t matB[K] = {1, 1, 1, 1};

  int32_t matC[N] = {0};

  std::array<int64_t, rankA> aDim{N, K};
  std::array<int64_t, rankA> atDim{K, N};
  std::array<int64_t, rankB> bDim{K};
  std::array<int64_t, rankC> cDim{N};

  memref::MemRef<int32_t, rankA> A((int32_t *)matA, aDim);
  memref::MemRef<int32_t, rankA> At((int32_t *)matAt, atDim);
  memref::MemRef<int32_t, rankB> B((int32_t *)matB, bDim);
  memref::MemRef<int32_t, rankC> C((int32_t *)matC, cDim);

  std::cout << "A Matrix:\n";
  utility::printTensor(A);

  std::cout << "B Vector:\n";
  utility::printTensor(B);

  _mlir_ciface_mv(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
  std::cout << "A * B result:\n";
  utility::printTensor(C);

  std::cout << "D Matrix = A^t:\n";
  utility::printTensor(At);

  // Clear output matrix
  std::fill(matC, matC + N, 0);

  _mlir_ciface_mv_At(&At.memRefDesc, &B.memRefDesc, &C.memRefDesc);
  std::cout << "D^t * B result:\n";
  utility::printTensor(C);

  simulator_terminate();
}
