#include "mlir_interface/memref/memref.hpp"
#include "utility/utility.hpp"

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
}

int main() {
  // Matrix-vector multiplication
  // C[N] =  A[N][K] * B[K]
  const int32_t N = 3;
  const int32_t K = 3;

  int32_t matA[N][K] = {
      {0, 0, 1},
      {0, 1, 0},
      {1, 0, 0},
  };

  int32_t matB[K] = {1, 2, 3};

  int32_t matC[N] = {0};

  std::array<int64_t, rankA> aDim{N, K};
  std::array<int64_t, rankB> bDim{K};
  std::array<int64_t, rankC> cDim{N};

  memref::MemRef<int32_t, rankA> A((int32_t *)matA, aDim);
  memref::MemRef<int32_t, rankB> B((int32_t *)matB, bDim);
  memref::MemRef<int32_t, rankC> C((int32_t *)matC, cDim);

  std::cout << "A Matrix:\n";
  utility::printTensor(A);

  std::cout << "B Vector:\n";
  utility::printTensor(B);

  _mlir_ciface_mv(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);

  std::cout << "GEMV result:\n";
  utility::printTensor(C);
}
