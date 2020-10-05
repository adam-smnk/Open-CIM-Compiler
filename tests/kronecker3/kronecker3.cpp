#include "mlir_interface/memref/memref.hpp"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

#define MAX_WEIGHT 5

// Functions generated from TC
extern "C" {
void _mlir_ciface_kronecker3(memref::MemRefDescriptor<int32_t, 2> *W0,
                             memref::MemRefDescriptor<int32_t, 2> *W1,
                             memref::MemRefDescriptor<int32_t, 2> *W2,
                             memref::MemRefDescriptor<int32_t, 4> *X,
                             memref::MemRefDescriptor<int32_t, 4> *Y,
                             memref::MemRefDescriptor<int32_t, 4> *XW1,
                             memref::MemRefDescriptor<int32_t, 4> *XW2);
}

int main() {
  const int32_t D0 = 2;
  const int32_t N0 = 1;
  const int32_t D1 = 2;
  const int32_t N1 = 2;
  const int32_t D2 = 2;
  const int32_t N2 = 2;
  const int32_t M = 1;

  // Input
  int32_t bufX[M][N0][N1][N2];
  for (int i = 0; i < M * N0 * N1 * N2; ++i) {
    ((int32_t *)bufX)[i] = i;
  }
  for (int i = 0; i < N1; i++) {
    for (int j = 0; j < N2; j++) {
      std::cout << bufX[0][0][i][j] << " ";
    }
    std::cout << "\n";
  }

  // Weights
  int32_t bufW0[D0 * N0];
  for (int i = 0; i < D0 * N0; ++i) {
    bufW0[i] = i;
  }
  int32_t bufW1[D1 * N1];
  for (int i = 0; i < D1 * N1; ++i) {
    bufW1[i] = i;
  }
  int32_t bufW2[D2 * N2];
  for (int i = 0; i < D2 * N2; ++i) {
    bufW2[i] = i;
  }

  // Outputs
  int32_t bufY[M][D0][D1][D2] = {0};
  int32_t bufXW1[M][N0][D1][D2] = {0};
  int32_t bufXW2[M][N0][N1][D2] = {0};

  memref::MemRef<int32_t, 2> W0((int32_t *)bufW0, {D0, N0});
  memref::MemRef<int32_t, 2> W1((int32_t *)bufW1, {D1, N1});
  memref::MemRef<int32_t, 2> W2((int32_t *)bufW2, {D2, N2});
  memref::MemRef<int32_t, 4> X((int32_t *)bufX, {M, N0, N1, N2});
  memref::MemRef<int32_t, 4> Y((int32_t *)bufY, {M, D0, D1, D2});
  memref::MemRef<int32_t, 4> XW1((int32_t *)bufXW1, {M, N0, D1, D2});
  memref::MemRef<int32_t, 4> XW2((int32_t *)bufXW2, {M, N0, N1, D2});

  std::cout << "Input:\n";
  utility::printTensor(X);

  std::cout << "Weights:\n";
  std::cout << "W0:\n";
  utility::printTensor(W0);
  std::cout << "W1:\n";
  utility::printTensor(W1);
  std::cout << "W2:\n";
  utility::printTensor(W2);

  _mlir_ciface_kronecker3(&W0.memRefDesc, &W1.memRefDesc, &W2.memRefDesc,
                          &X.memRefDesc, &Y.memRefDesc, &XW1.memRefDesc,
                          &XW2.memRefDesc);

  std::cout << "### Kronecker3 results ###\n";
  std::cout << "Y:\n";
  utility::printTensor(Y);
  for (int i = 0; i < D1; i++) {
    for (int j = 0; j < D2; j++) {
      std::cout << bufY[0][0][i][j] << " ";
    }
    std::cout << "\n";
  }
  for (int i = 0; i < D1; i++) {
    for (int j = 0; j < D2; j++) {
      std::cout << bufY[0][1][i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "XW1:\n";
  utility::printTensor(XW1);
  for (int i = 0; i < N1; i++) {
    for (int j = 0; j < N2; j++) {
      std::cout << bufXW1[0][0][i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "XW2:\n";
  utility::printTensor(XW2);
  for (int i = 0; i < N1; i++) {
    for (int j = 0; j < N2; j++) {
      std::cout << bufXW2[0][0][i][j] << " ";
    }
    std::cout << "\n";
  }
}
