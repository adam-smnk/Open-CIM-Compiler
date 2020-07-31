#include "mlir_interface/memref/memref.hpp"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

static const size_t rankA = 4;
static const size_t rankB = 5;
static const size_t rankC = 3;

// Functions generated from TC
extern "C" {
void _mlir_ciface_tc4x5x3(memref::MemRefDescriptor<int32_t, rankA> *A,
                          memref::MemRefDescriptor<int32_t, rankB> *B,
                          memref::MemRefDescriptor<int32_t, rankC> *C);
}

int main() {
  const int32_t M = 3;
  const int32_t N = 2;
  const int32_t K = 4;
  const int32_t L = 5;
  const int32_t P = 3;
  const int32_t Q = 4;

  int32_t matA[M * P * K * L];
  for (int i = 0; i < M * P * K * L; ++i) {
    matA[i] = i;
  }

  int32_t matB[P * K * L * N * Q];
  for (int i = 0; i < P * K * L * N * Q; ++i) {
    matB[i] = i;
  }

  int32_t matC[M * N * Q] = {0};

  std::array<int64_t, rankA> aDim = {P, K, L, M};
  std::array<int64_t, rankB> bDim = {K, Q, L, N, P};
  std::array<int64_t, rankC> cDim = {M, N, Q};

  memref::MemRef<int32_t, rankA> A((int32_t *)matA, aDim);
  memref::MemRef<int32_t, rankB> B((int32_t *)matB, bDim);
  memref::MemRef<int32_t, rankC> C((int32_t *)matC, cDim);

  std::cout << "A Matrix:\n";
  utility::printTensor(A);

  std::cout << "B Matrix:\n";
  utility::printTensor(B);

  _mlir_ciface_tc4x5x3(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);

  std::cout << "Contraction using CIM result:\n";
  utility::printTensor(C);

  std::cout << "matC:\n";
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      std::cout << "| ";
      for (int q = 0; q < Q; ++q) {
        std::cout << matC[m * N * Q + n * Q + q] << " ";
      }
      std::cout << "|\n";
    }
    std::cout << "\n";
  }
}
