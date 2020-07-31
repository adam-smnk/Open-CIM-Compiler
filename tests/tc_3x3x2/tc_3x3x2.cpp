#include "mlir_interface/memref/memref.hpp"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

// Example based on Python NumPy tensordot docs:
// https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html

static const size_t rankA = 3;
static const size_t rankB = 3;
static const size_t rankC = 2;

// Functions generated from TC
extern "C" {
void _mlir_ciface_tc3x3x2(memref::MemRefDescriptor<int32_t, rankA> *A,
                          memref::MemRefDescriptor<int32_t, rankB> *B,
                          memref::MemRefDescriptor<int32_t, rankC> *C);
}

int main() {
  const int32_t M = 5;
  const int32_t N = 2;
  const int32_t K = 3;
  const int32_t L = 4;

  int32_t matA[K * L * M];
  for (int i = 0; i < K * L * M; ++i) {
    matA[i] = i;
  }

  int32_t matB[L * K * N];
  for (int i = 0; i < L * K * N; ++i) {
    matB[i] = i;
  }

  int32_t matC[M * N] = {0};

  std::array<int64_t, rankA> aDim = {K, L, M};
  std::array<int64_t, rankB> bDim = {L, K, N};
  std::array<int64_t, rankC> cDim = {M, N};

  memref::MemRef<int32_t, rankA> A((int32_t *)matA, aDim);
  memref::MemRef<int32_t, rankB> B((int32_t *)matB, bDim);
  memref::MemRef<int32_t, rankC> C((int32_t *)matC, cDim);

  std::cout << "A Matrix:\n";
  utility::printTensor(A);

  std::cout << "B Matrix:\n";
  utility::printTensor(B);

  _mlir_ciface_tc3x3x2(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);

  std::cout << "Contraction using CIM result:\n";
  utility::printTensor(C);

  // Clear output matrix
  std::fill(matC, matC + (M * N), 0);

  // Compute TTGT manually
  int32_t flatA[M][K * L] = {};
  int32_t flatB[K * L][N] = {};
  int32_t flatC[M][N] = {};

  // Transpose (flatten) A
  for (int k = 0; k < K; ++k) {
    for (int l = 0; l < L; ++l) {
      for (int m = 0; m < M; ++m) {
        flatA[m][k * L + l] = matA[k * L * M + l * M + m];
      }
    }
  }

  // Transpose (flatten) B
  for (int l = 0; l < L; ++l) {
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        flatB[k * L + l][n] = matB[l * K * N + k * N + n];
      }
    }
  }

  // GEMM
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      uint32_t sum = 0;

      for (int k = 0; k < K * L; ++k) {
        sum += flatA[m][k] * flatB[k][n];
      }

      flatC[m][n] = sum;
    }
  }

  // Transpose (unflatten) C
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      matC[m * N + n] = flatC[m][n];
    }
  }

  std::cout << "Manual contraction result:\n";
  utility::printTensor(C);
}
