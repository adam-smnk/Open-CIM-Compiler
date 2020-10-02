#include "mlir_interface/memref/memref.hpp"
#include "utility/utility.hpp"

#include <array>
#include <cstdint>
#include <iostream>

// Functions generated from TC
extern "C" {
void _mlir_ciface_conv1d(memref::MemRefDescriptor<int32_t, 3> *A,
                         memref::MemRefDescriptor<int32_t, 3> *B,
                         memref::MemRefDescriptor<int32_t, 3> *C);
}

int main() {
  // Conv1D
  // C[N][K][H] =  A[N][C][H] * B[K][C][KH]
  const size_t rank = 3;
  const int32_t N = 1;
  const int32_t K = 1;
  const int32_t C = 1;
  const int32_t H = 3;
  const int32_t KH = 2;

  int32_t matA[N][C][H];
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        matA[n][c][h] = h;
      }
    }
  }

  int32_t matB[K][C][KH];
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      for (int kh = 0; kh < KH; ++kh) {
        matB[k][c][kh] = 1;
      }
    }
  }

  int32_t matC[N][K][H];
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int h = 0; h < H; ++h) {
        matC[n][k][h] = 0;
      }
    }
  }

  memref::MemRef<int32_t, rank> A((int32_t *)matA, {N, C, H});
  memref::MemRef<int32_t, rank> B((int32_t *)matB, {K, C, KH});
  memref::MemRef<int32_t, rank> memC((int32_t *)matC, {N, K, H});

  std::cout << "A input:\n";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      std::cout << "N: " << n << " C: " << c << "\n";
      memref::MemRef<int32_t, 1> mat((int32_t *)matA[n][c], {H});
      utility::printTensor(mat);
    }
  }

  std::cout << "B input:\n";
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      std::cout << "K: " << k << " C: " << c << "\n";
      memref::MemRef<int32_t, 1> mat((int32_t *)matB[k][c], {KH});
      utility::printTensor(mat);
    }
  }

  _mlir_ciface_conv1d(&A.memRefDesc, &B.memRefDesc, &memC.memRefDesc);

  std::cout << "C output:\n";
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      std::cout << "N: " << n << " K: " << k << "\n";
      memref::MemRef<int32_t, 1> mat((int32_t *)matC[n][k], {H});
      utility::printTensor(mat);
    }
  }
}
