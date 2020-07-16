#include "mlir_interface/memref/memref.hpp"

#include <array>
#include <cstdint>

// Functions generated from TC
extern "C" {
void _mlir_ciface_vm(memref::MemRefDescriptor<int16_t, 1> *A,
                     memref::MemRefDescriptor<int16_t, 2> *B,
                     memref::MemRefDescriptor<int16_t, 1> *C);
}

int main() {
  // Vector-matrix multiplication
  // C[N] =  A[K] * B[K][N]
  const size_t matRank = 2;
  const size_t vecRank = 1;
  const int32_t N = 16;
  const int32_t K = 8;

  int16_t matA[K] = {0};
  int16_t matB[K][N] = {0};
  int16_t matC[N] = {0};

  std::array<int64_t, vecRank> aDim{K};
  std::array<int64_t, matRank> bDim{K, N};
  std::array<int64_t, vecRank> cDim{N};

  memref::MemRef<int16_t, vecRank> A((int16_t *)matA, aDim);
  memref::MemRef<int16_t, matRank> B((int16_t *)matB, bDim);
  memref::MemRef<int16_t, vecRank> C((int16_t *)matC, cDim);

  _mlir_ciface_vm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
}
