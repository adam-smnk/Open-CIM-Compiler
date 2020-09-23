#include "mlir_interface/memref/memref.hpp"
#include "simulator_interface/cim_sim.h"
#include "utility/utility.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

#define CIM_PRECISION int8_t

// Example based on Python NumPy tensordot docs:
// https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html

static const size_t rankA = 3;
static const size_t rankB = 3;
static const size_t rankC = 2;

// Functions generated from TC
extern "C" {
void _mlir_ciface_tc3x3x2(memref::MemRefDescriptor<CIM_PRECISION, rankA> *A,
                          memref::MemRefDescriptor<CIM_PRECISION, rankB> *B,
                          memref::MemRefDescriptor<CIM_PRECISION, rankC> *C);
}

int main() {
  simulator_init();

  const int32_t M = 5;
  const int32_t N = 2;
  const int32_t K = 3;
  const int32_t L = 4;

  CIM_PRECISION matA[K * L * M];
  for (int i = 0; i < K * L * M; ++i) {
    matA[i] = i;
  }

  CIM_PRECISION matB[L * K * N];
  for (int i = 0; i < L * K * N; ++i) {
    matB[i] = i;
  }

  CIM_PRECISION matC[M * N] = {0};

  std::array<int64_t, rankA> aDim = {K, L, M};
  std::array<int64_t, rankB> bDim = {L, K, N};
  std::array<int64_t, rankC> cDim = {M, N};

  memref::MemRef<CIM_PRECISION, rankA> A((CIM_PRECISION *)matA, aDim);
  memref::MemRef<CIM_PRECISION, rankB> B((CIM_PRECISION *)matB, bDim);
  memref::MemRef<CIM_PRECISION, rankC> C((CIM_PRECISION *)matC, cDim);

  std::cout << "A Matrix:\n";
  utility::printTensor(A);

  std::cout << "B Matrix:\n";
  utility::printTensor(B);

  _mlir_ciface_tc3x3x2(&A.memRefDesc, &B.memRefDesc, &C.memRefDesc);

  std::cout << "Contraction using CIM result:\n";
  utility::printTensor(C);

  // Compute TTGT manually
  CIM_PRECISION flatA[M][K * L] = {};
  CIM_PRECISION flatB[K * L][N] = {};
  CIM_PRECISION flatC[M][N] = {};
  CIM_PRECISION outputC[M * N] = {0};

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
      outputC[m * N + n] = flatC[m][n];
    }
  }
  memref::MemRef<CIM_PRECISION, rankC> manualC((CIM_PRECISION *)outputC, cDim);

  std::cout << "Manual contraction result:\n";
  utility::printTensor(manualC);

  std::cout << "Contration results are equal: ";
  if (std::equal(std::begin(matC), std::end(matC), std::begin(outputC))) {
    std::cout << "TRUE\n";
  } else {
    std::cout << "FALSE\n";
  }

  simulator_terminate();
}
