#include "mlir_interface/memref/memref.hpp"

#include <array>
#include <cstdint>

// Functions generated from TC
extern "C" {
void _mlir_ciface_mm(memref::MemRefDescriptor<int32_t, 2> *A,
                     memref::MemRefDescriptor<int32_t, 2> *B,
                     memref::MemRefDescriptor<int32_t, 2> *C);
}

int main() {
  // Matrix-matrix multiplication
  // C[M][N] =  A[M][K] * B[K][N]
  const size_t rank = 2;
  const int32_t M = 16;
  const int32_t N = 16;
  const int32_t K = 8;

  std::array<int32_t, rank> aDim{M, K};
  std::array<int32_t, rank> bDim{K, N};
  std::array<int32_t, rank> cDim{M, N};

  memref::MemRef<int32_t, rank> A(aDim);
  memref::MemRef<int32_t, rank> B(bDim);
  memref::MemRef<int32_t, rank> C(cDim);

  _mlir_ciface_mm(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);
}
