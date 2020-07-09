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

  std::array<int32_t, vecRank> aDim{K};
  std::array<int32_t, matRank> bDim{K, N};
  std::array<int32_t, vecRank> cDim{N};

  memref::MemRef<int16_t, vecRank> A(aDim);
  memref::MemRef<int16_t, matRank> B(bDim);
  memref::MemRef<int16_t, vecRank> C(cDim);

  _mlir_ciface_vm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
}
